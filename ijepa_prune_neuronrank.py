#!/usr/bin/env python3
"""
I-JEPA selective layer pruning with NeuronRank-style MLP channel scoring (single file).

What it does
------------
- Loads a Hugging Face IJepaModel (ViT-style encoder).
- Lets you target a subset of transformer blocks (layers) to prune.
- Collects post-activation outputs of each target block's MLP.fc1 on a small calibration set.
- Computes a simple NeuronRank-style score per intermediate channel ("fires often on informative tokens").
- Structurally removes the lowest-scoring channels from fc1 **and** the corresponding input channels of fc2.
- (Optional) Runs a quick k-NN probe on Imagenette to check that representations are still useful.
- Saves the pruned model.

Usage examples
--------------
# Dry-run: score channels for last 8 blocks, prune 30%, no eval, save to out dir
python ijepa_prune_neuronrank.py \
  --model-id facebook/ijepa_vith14_1k \
  --layers last8 \
  --prune-ratio 0.30 \
  --calib-ds imagenette --calib-samples 2000 \
  --save-dir ./ijepa_pruned_last8_r30

# With eval: k-NN on Imagenette (small split), 30% prune on blocks 24-31
python ijepa_prune_neuronrank.py \
  --model-id facebook/ijepa_vith14_1k \
  --layers 24-31 \
  --prune-ratio 0.30 \
  --calib-ds imagenette --calib-samples 3000 \
  --eval knn --eval-train 2000 --eval-val 1000 \
  --save-dir ./ijepa_pruned_24_31_r30

Dependencies
------------
- torch >= 2.1
- transformers >= 4.56
- datasets
- pillow
- torch-pruning >= 1.2
- **optional for linear probe**: scikit-learn (if you prefer sklearn’s LogisticRegression, otherwise the script uses k-NN)

Notes
-----
-----
- We prune **MLP hidden width** (fc1 out channels / fc2 in channels). Attention heads are left untouched here.
- Calibration set can be Imagenette (auto-downloaded) or a local ImageFolder via `--calib-path`.
- This is a minimal, readable reference — not hyper-optimized.
"""

from __future__ import annotations
import argparse
import math
import os
import random
import re
from dataclasses import dataclass
import copy
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

try:
    import torch_pruning as tp
except Exception as e:
    raise SystemExit("Please install torch-pruning: pip install torch-pruning")

from transformers import AutoModel, AutoProcessor
from datasets import load_dataset

EPS = 1e-12

# ------------------------------
# Utilities
# ------------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class MLPHandle:
    name: str
    mlp: nn.Module
    fc1: nn.Linear
    fc2: nn.Linear
    idx: int  # block index inferred by discovery order


def find_mlp_modules(model: nn.Module) -> List[MLPHandle]:
    """Discover MLP modules in various layouts and return them in traversal order.

    Supports:
    - Direct `fc1`/`fc2` attributes (timm-like VisionTransformer blocks)
    - Hugging Face ViT-style blocks with `intermediate.dense` and `output.dense`
    - Fallback: modules exposing `.mlp.fc1` / `.mlp.fc2`
    """
    mlps: List[MLPHandle] = []

    # Pattern 1: Direct fc1/fc2 on a module
    for name, module in model.named_modules():
        if hasattr(module, "fc1") and hasattr(module, "fc2"):
            fc1 = getattr(module, "fc1")
            fc2 = getattr(module, "fc2")
            if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
                mlps.append(MLPHandle(name=name, mlp=module, fc1=fc1, fc2=fc2, idx=len(mlps)))
    if mlps:
        return mlps

    # Pattern 2: HF ViT-style: <block>.intermediate.dense and <block>.output.dense
    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())
    seen_blocks = set()
    for name, module in list(name_to_module.items()):
        if name.endswith(".intermediate") and hasattr(module, "dense"):
            base = name[: -len(".intermediate")]
            inter_dense = getattr(module, "dense")
            out_mod = name_to_module.get(base + ".output")
            out_dense = getattr(out_mod, "dense", None) if out_mod is not None else None
            if isinstance(inter_dense, nn.Linear) and isinstance(out_dense, nn.Linear):
                if base not in seen_blocks:
                    mlps.append(MLPHandle(name=base, mlp=None, fc1=inter_dense, fc2=out_dense, idx=len(mlps)))
                    seen_blocks.add(base)
    if mlps:
        return mlps

    # Pattern 3: Nested mlp with fc1/fc2
    for name, module in model.named_modules():
        if hasattr(module, "mlp"):
            mlp = getattr(module, "mlp")
            fc1 = getattr(mlp, "fc1", None)
            fc2 = getattr(mlp, "fc2", None)
            if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
                mlps.append(MLPHandle(name=f"{name}.mlp", mlp=mlp, fc1=fc1, fc2=fc2, idx=len(mlps)))
    if mlps:
        return mlps

    raise RuntimeError("Could not find any MLPs to prune (tried fc1/fc2, intermediate/output.dense, and mlp.fc1/fc2). IJepaModel layout may have changed.")


def parse_layer_selection(spec: str, total: int) -> List[int]:
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return list(range(total))
    if spec.startswith("last"):
        k = int(spec.replace("last", ""))
        return list(range(max(total - k, 0), total))
    if spec == "even":
        return [i for i in range(total) if i % 2 == 0]
    if spec == "odd":
        return [i for i in range(total) if i % 2 == 1]
    if spec.startswith("list:"):
        return [int(x) for x in spec.split(":", 1)[1].split(",") if x]
    if "-" in spec:  # e.g., 24-31
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    # single index
    return [int(spec)]


def _resolve_imagefolder_root_and_split(spec: str) -> Tuple[str, str]:
    """Resolve a local ImageFolder root and split indicator from a user spec.

    Currently accepts either a plain path to the dataset root (containing
    'train' and 'val'/'validation' subfolders), or a path followed by an
    optional ':split' suffix. We return the expanded root path and the split
    string (defaulting to 'train').
    """
    if ":" in spec:
        root, split = spec.split(":", 1)
        root = os.path.expanduser(root)
        split = split.strip() or "train"
        return root, split
    return os.path.expanduser(spec), "train"


def auto_download_imagenette(dataset_dir: str = "~/Datasets") -> str:
    """Auto-download Imagenette2-320 if not present.
    
    Returns the path to the extracted dataset.
    """
    import subprocess
    import tarfile
    from urllib.request import urlretrieve
    
    dataset_dir = os.path.expanduser(dataset_dir)
    target_path = os.path.join(dataset_dir, "imagenette2-320")
    
    # Check if already exists
    if os.path.isdir(target_path):
        train_dir = os.path.join(target_path, "train")
        if os.path.isdir(train_dir):
            print(f"✓ Found existing Imagenette2-320 at: {target_path}")
            return target_path
    
    # Download
    os.makedirs(dataset_dir, exist_ok=True)
    tgz_path = os.path.join(dataset_dir, "imagenette2-320.tgz")
    
    if not os.path.exists(tgz_path):
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        print(f"Downloading Imagenette2-320 from {url}...")
        print(f"  Saving to: {tgz_path}")
        try:
            urlretrieve(url, tgz_path)
            print(f"✓ Downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download Imagenette2-320: {e}")
    
    # Extract
    print(f"Extracting {tgz_path}...")
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dataset_dir)
        print(f"✓ Extracted to: {target_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract Imagenette2-320: {e}")
    
    # Verify
    if not os.path.isdir(os.path.join(target_path, "train")):
        raise RuntimeError(f"Extraction succeeded but 'train' folder not found at {target_path}")
    
    return target_path


# ------------------------------
# Benchmark-style NeuronRank (pre-activation, TF-IDF)
# ------------------------------
@torch.no_grad()
def collect_activation_statistics_ijepa(
    model: nn.Module,
    processor: AutoProcessor,
    dataset_name: str,
    num_images: int,
    target_mlps: List[MLPHandle],
    batch_size: int = 16,
    activation_threshold: float = 0.05,
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect pre-activation statistics for NeuronRank (benchmark-style).
    
    Returns dict: {module_name: {'mean_abs_activation', 'doc_freq', 'sample_count'}}
    """
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    # Register hooks on fc1 modules (pre-activation)
    for h in target_mlps:
        fc1 = h.fc1
        name = h.name + ".intermediate.dense" if h.name.startswith("encoder.layer") else h.name + ".fc1"
        
        def make_hook(layer_name: str):
            def hook(_module, inputs, _output):
                if not inputs:
                    return
                features = inputs[0]  # [B, T, in_features]
                if features is None:
                    return
                features = features.detach()
                # Flatten batch & sequence: [B*T, in_features]
                if features.dim() == 3:
                    flattened = features.reshape(-1, features.shape[-1]).abs()
                elif features.dim() == 2:
                    flattened = features.abs()
                else:
                    flattened = features.abs()
                
                flattened = flattened.to(dtype=torch.float32, device='cpu')
                present = (flattened > activation_threshold).to(dtype=torch.float32)
                
                layer_stats = stats.setdefault(layer_name, {
                    'sum_abs_activation': torch.zeros(flattened.size(-1), dtype=torch.float32),
                    'doc_freq': torch.zeros(flattened.size(-1), dtype=torch.float32),
                    'sample_count': 0,
                })
                layer_stats['sum_abs_activation'] += flattened.sum(dim=0)
                layer_stats['doc_freq'] += present.sum(dim=0)
                layer_stats['sample_count'] += flattened.size(0)
            return hook
        
        handles.append(fc1.register_forward_hook(make_hook(name)))
    
    # Load calibration dataset (local folder)
    model.eval().to(device)
    
    # Auto-download if "imagenette" keyword is used
    if dataset_name == "imagenette":
        root = auto_download_imagenette()
    else:
        root = os.path.expanduser(dataset_name)
    
    # Assume local ImageFolder structure
    train_dir = os.path.join(root, "train") if os.path.isdir(os.path.join(root, "train")) else root
    paths: List[Path] = []
    for p in Path(train_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No images found under: {train_dir}")
    random.shuffle(paths)
    paths = paths[: max(1, min(num_images, len(paths)))]
    
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        _ = model(**inputs)
        for im in imgs:
            try: im.close()
            except Exception: pass
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Compute mean
    for layer_name, layer_stats in stats.items():
        count = layer_stats['sample_count']
        if count > 0:
            layer_stats['mean_abs_activation'] = layer_stats['sum_abs_activation'] / count
        else:
            layer_stats['mean_abs_activation'] = torch.zeros_like(layer_stats['sum_abs_activation'])
        layer_stats['doc_freq'] = layer_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        del layer_stats['sum_abs_activation']
    
    return stats


@torch.no_grad()
def compute_neuronrank_scores_ijepa(
    target_mlps: List[MLPHandle],
    activation_stats: Dict[str, Dict[str, torch.Tensor]],
    *,
    tf_power: float = 1.0,
    idf_power: float = 1.0,
    idf_add: float = 1.0,
    idf_smooth: float = 1.0,
    weight_power: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """Compute per-channel NeuronRank scores (benchmark-style TF-IDF).
    
    Returns {block_idx: scores[out_channels]} for fc1 output channels.
    """
    score_map: Dict[int, torch.Tensor] = {}
    
    for h in target_mlps:
        name = h.name + ".intermediate.dense" if h.name.startswith("encoder.layer") else h.name + ".fc1"
        stats = activation_stats.get(name)
        if not stats:
            continue
        
        sample_count = int(stats.get('sample_count', 0))
        if sample_count == 0:
            continue
        
        mean_abs_activation = stats.get('mean_abs_activation')
        doc_freq = stats.get('doc_freq')
        if mean_abs_activation is None or doc_freq is None:
            continue
        
        mean_abs_activation = mean_abs_activation.to(torch.float32)
        doc_freq = doc_freq.to(torch.float32)
        
        weight = h.fc1.weight.detach().to(dtype=torch.float32, device='cpu')  # [out, in]
        if weight.numel() == 0:
            continue
        
        # TF component: per-input-channel mean activation
        tf_component = mean_abs_activation.clamp(min=0.0).pow(tf_power)  # [in]
        
        # IDF component: per-input-channel log(N / DF)
        smooth = idf_smooth if idf_smooth > 0 else 0.0
        numerator_value = float(sample_count) + smooth + EPS
        numerator = torch.tensor(numerator_value, dtype=torch.float32)
        denominator = doc_freq + smooth + EPS
        idf_component = torch.log(numerator / denominator)
        if idf_add != 0.0:
            idf_component = idf_component + idf_add
        idf_component = idf_component.clamp(min=0.0).pow(idf_power)  # [in]
        
        # Weight component: per-weight magnitude
        weight_component = weight.abs().pow(weight_power)  # [out, in]
        
        # Combine: for each output channel j, aggregate over input channels i
        # score[j] = sum_i( |W[j,i]|^p * TF[i] * IDF[i] )
        tf_idf = tf_component * idf_component  # [in]
        per_weight_scores = weight_component * tf_idf.unsqueeze(0)  # [out, in]
        channel_scores = per_weight_scores.sum(dim=1)  # [out]
        
        score_map[h.idx] = channel_scores
    
    return score_map


# ------------------------------
# NeuronRank scoring (DEPRECATED: old quantile-based method)
# ------------------------------
class ChannelQuantile:
    """Reservoir sampler + per-channel quantile estimator (simple, bounded memory)."""
    def __init__(self, hidden: int, max_tokens: int = 20000, q: float = 0.9, device: str = "cpu"):
        self.hidden = hidden
        self.max_tokens = max_tokens
        self.q = q
        self.device = device
        self.buf = None  # [T, hidden]

    @torch.no_grad()
    def ingest(self, x: torch.Tensor):
        # x: [tokens, hidden]
        x = x.detach()
        if self.buf is None:
            self.buf = x[: self.max_tokens].clone().to(self.device)
            return
        # concatenate (bounded)
        need = self.max_tokens - self.buf.shape[0]
        if need > 0:
            take = min(need, x.shape[0])
            self.buf = torch.cat([self.buf, x[:take].to(self.device)], dim=0)
        # if full, randomly replace a chunk (reservoir)
        else:
            idx = torch.randint(0, self.buf.shape[0], (min(x.shape[0], self.buf.shape[0] // 4),), device=self.device)
            rep = x[torch.randint(0, x.shape[0], (idx.numel(),))].to(self.device)
            self.buf[idx] = rep

    @torch.no_grad()
    def threshold(self) -> torch.Tensor:
        if self.buf is None or self.buf.numel() == 0:
            return torch.zeros(self.hidden, device=self.device)
        return torch.quantile(self.buf, self.q, dim=0)


@torch.no_grad()
def collect_neuronrank_scores(
    model: nn.Module,
    processor: AutoProcessor,
    dataset_name: str,
    num_images: int,
    target_mlps: List[MLPHandle],
    batch_size: int = 16,
    quantile_q: float = 0.90,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Runs a calibration set through the model, grabbing post-GELU activations of each target MLP.fc1,
    and computes a DF-frequency score per channel. Returns {block_idx: scores[hidden] (0..1)}.
    """
    # Build hooks that capture fc1 outputs; we apply GELU ourselves to get post-activation
    gelu = nn.GELU()
    buffers: Dict[int, List[torch.Tensor]] = {h.idx: [] for h in target_mlps}
    quantiles: Dict[int, ChannelQuantile] = {}

    def make_hook(block_idx: int):
        def hook(_mod, _inp, out):
            # out: [B, tokens, hidden] or [B, hidden]? For ViT MLP.fc1 it's [B, tokens, intermediate]
            a = gelu(out)
            # collapse batch and tokens
            if a.dim() == 3:
                a2 = a.flatten(0, 1)  # [B*T, hidden]
            else:
                a2 = a
            buffers[block_idx].append(a2.detach().cpu())
            if block_idx not in quantiles:
                quantiles[block_idx] = ChannelQuantile(hidden=a2.shape[-1], max_tokens=20000, q=quantile_q, device="cpu")
            quantiles[block_idx].ingest(a2.cpu())
        return hook

    handles = []
    for h in target_mlps:
        handles.append(h.fc1.register_forward_hook(make_hook(h.idx)))

    # Load calibration dataset (Imagenette 320px by default)
    if dataset_name == "imagenette":
        ds = load_dataset("frgfm/imagenette", "320px", split="train")
        # Reduce to num_images
        ds = ds.shuffle(seed=123).select(range(min(num_images, len(ds))))
        get_image = lambda rec: rec["image"]
        get_label = lambda rec: rec.get("label", 0)
    else:
        # Assume local folder processed by datasets ImageFolder
        ds = load_dataset("imagefolder", data_dir=dataset_name, split="train")
        ds = ds.shuffle(seed=123).select(range(min(num_images, len(ds))))
        get_image = lambda rec: rec["image"]
        get_label = lambda rec: rec.get("label", 0)

    model.eval().to(device)

    loader_idxs = list(range(len(ds)))
    for i in range(0, len(loader_idxs), batch_size):
        batch = [ds[j] for j in loader_idxs[i : i + batch_size]]
        images = [get_image(rec).convert("RGB") if isinstance(get_image(rec), Image.Image) else Image.fromarray(get_image(rec).numpy()) for rec in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        _ = model(**inputs)

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute DF scores using quantile thresholds
    scores: Dict[int, torch.Tensor] = {}
    for h in target_mlps:
        block = h.idx
        if not buffers[block]:
            raise RuntimeError(f"No activations captured for block {block}")
        act = torch.cat(buffers[block], dim=0)  # [tokens, hidden]
        thr = quantiles[block].threshold()  # [hidden]
        # hits: token j is active if act[:, j] > thr[j]
        hits = (act > thr).float()
        df = hits.mean(dim=0)  # frequency in [0,1]
        scores[block] = df
    return scores


# ------------------------------
# Structural pruning (fc1 out / fc2 in)
# ------------------------------
class IJepaWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).last_hidden_state


def apply_structured_mlp_pruning(
    model: nn.Module,
    processor: AutoProcessor,
    target_mlps: List[MLPHandle],
    scores: Dict[int, torch.Tensor],
    prune_ratio: float,
    device: str = "cuda",
):
    # Prepare a wrapper and dependency graph with a dummy image
    wrapper = IJepaWrapper(model)
    wrapper.eval().to(device)

    # Create a dummy example input matching processor's expected size
    size = model.config.image_size
    dummy = torch.zeros(1, 3, size, size, device=device)

    DG = tp.DependencyGraph().build_dependency(wrapper, example_inputs=(dummy,))

    total_pruned = 0
    for h in target_mlps:
        sc = scores[h.idx]
        hidden = sc.numel()
        k = int(round(prune_ratio * hidden))
        if k <= 0:
            continue
        # prune the lowest-scoring channels
        idxs = torch.argsort(sc)[:k].tolist()
        group = DG.get_pruning_group(h.fc1, tp.prune_linear_out_channels, idxs=idxs)
        if DG.check_pruning_group(group):
            group.prune()
            total_pruned += len(idxs)
        else:
            print(f"[warn] Skipped block {h.idx}: pruning group invalid")
    return total_pruned


# ------------------------------
# Lightweight k-NN probe (HF Imagenette) — optional
# ------------------------------
@torch.no_grad()
def knn_probe(
    model: nn.Module,
    processor: AutoProcessor,
    train_n: int = 2000,
    val_n: int = 1000,
    k: int = 20,
    device: str = "cuda",
) -> float:
    model.eval().to(device)
    ds_train = load_dataset("frgfm/imagenette", "320px", split="train").shuffle(seed=123).select(range(train_n))
    ds_val = load_dataset("frgfm/imagenette", "320px", split="validation").shuffle(seed=123).select(range(val_n))

    def embed_batch(images: List[Image.Image]) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model(**inputs).last_hidden_state.mean(dim=1)  # [B, hidden]
        out = F.normalize(out, p=2, dim=1)
        return out

    bank_embeds, bank_labels = [], []
    bs = 32
    for i in range(0, len(ds_train), bs):
        batch = ds_train[i : i + bs]
        images = [rec["image"].convert("RGB") for rec in batch]
        labels = [int(rec["label"]) for rec in batch]
        bank_embeds.append(embed_batch(images))
        bank_labels.extend(labels)
    bank = torch.cat(bank_embeds, dim=0)
    bank_labels = torch.tensor(bank_labels, device=device)

    correct = 0
    total = 0
    for i in range(0, len(ds_val), bs):
        batch = ds_val[i : i + bs]
        images = [rec["image"].convert("RGB") for rec in batch]
        labels = torch.tensor([int(rec["label"]) for rec in batch], device=device)
        q = embed_batch(images)
        sims = q @ bank.T
        topk = sims.topk(k=min(k, bank.shape[0]), dim=1).indices
        preds = []
        for row in topk:
            labs = bank_labels[row]
            # torch.mode not supported on MPS, move to CPU
            pred = torch.mode(labs.cpu()).values.item()
            preds.append(pred)
        preds = torch.tensor(preds, device=device)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)

# ------------------------------
# k-NN probe (local ImageFolder) — fast, no Arrow caching
# ------------------------------
@torch.no_grad()
def knn_probe_local(
    model: nn.Module,
    processor: AutoProcessor,
    root: str,
    train_n: int = 2000,
    val_n: int = 1000,
    k: int = 20,
    device: str = "cuda",
) -> float:
    model.eval().to(device)
    root = os.path.expanduser(root)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "validation")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(root, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise RuntimeError(f"Expected 'train' and 'val/validation' under {root}")

    def list_images_with_labels(split_dir: str) -> List[Tuple[Path, int]]:
        cls_dirs = sorted([d for d in Path(split_dir).iterdir() if d.is_dir()])
        cls_to_idx = {d.name: i for i, d in enumerate(cls_dirs)}
        items: List[Tuple[Path, int]] = []
        for cls_name, idx in cls_to_idx.items():
            d = Path(split_dir) / cls_name
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    items.append((p, idx))
        return items

    train_items = list_images_with_labels(train_dir)
    val_items = list_images_with_labels(val_dir)
    random.shuffle(train_items)
    random.shuffle(val_items)
    train_items = train_items[: max(1, min(train_n, len(train_items)))]
    val_items = val_items[: max(1, min(val_n, len(val_items)))]

    def embed_paths(paths: List[Path]) -> torch.Tensor:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model(**inputs).last_hidden_state.mean(dim=1)
        out = F.normalize(out, p=2, dim=1)
        for im in imgs:
            try: im.close()
            except Exception: pass
        return out

    # Build bank from train items
    bank_embeds, bank_labels = [], []
    bs = 32
    for i in range(0, len(train_items), bs):
        batch = train_items[i : i + bs]
        bank_embeds.append(embed_paths([p for p, _ in batch]))
        bank_labels.extend([lbl for _, lbl in batch])
    bank = torch.cat(bank_embeds, dim=0)
    bank_labels = torch.tensor(bank_labels, device=device)

    # Evaluate on val items
    correct = 0
    total = 0
    for i in range(0, len(val_items), bs):
        batch = val_items[i : i + bs]
        q = embed_paths([p for p, _ in batch])
        labels = torch.tensor([lbl for _, lbl in batch], device=device)
        sims = q @ bank.T
        topk = sims.topk(k=min(k, bank.shape[0]), dim=1).indices
        preds = []
        for row in topk:
            labs = bank_labels[row]
            # torch.mode not supported on MPS, move to CPU
            pred = torch.mode(labs.cpu()).values.item()
            preds.append(pred)
        preds = torch.tensor(preds, device=device)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)

# ------------------------------
# Magnitude (L2) channel scores for fc1 rows
# ------------------------------
@torch.no_grad()
def compute_magnitude_scores(target_mlps: List[MLPHandle]) -> Dict[int, torch.Tensor]:
    scores: Dict[int, torch.Tensor] = {}
    for h in target_mlps:
        w = h.fc1.weight.detach()  # [out, in]
        scores[h.idx] = torch.norm(w, dim=1)  # per-out-channel L2
    return scores

# ------------------------------
# Unstructured pruning (PyTorch prune) helpers
# ------------------------------
def collect_fc1_modules_for_blocks(model: nn.Module, target_indices: List[int]) -> List[Tuple[str, nn.Linear]]:
    mods: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        is_fc1 = name.endswith(".mlp.fc1") or name.endswith(".intermediate.dense")
        if not is_fc1:
            continue
        m = re.search(r"encoder\.layer\.(\d+)", name)
        if not m:
            continue
        if int(m.group(1)) in target_indices:
            mods.append((name, mod))
    return mods


def collect_linear_modules_for_unstructured(
    model: nn.Module,
    target_indices: List[int],
    include_attn: bool = False,
    include_mlp: bool = True,
) -> Tuple[List[nn.Linear], List[str]]:
    """Collect nn.Linear modules in the chosen encoder blocks.
    - include_mlp: intermediate.dense / output.dense (and timm .mlp.fc1/fc2)
    - include_attn: q/k/v and attention output projection
    Returns (modules, names) in discovery order.
    """
    modules: List[nn.Linear] = []
    names: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.search(r"encoder\.layer\.(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in target_indices:
            continue
        is_mlp = (
            ".intermediate.dense" in name or ".output.dense" in name or
            name.endswith(".mlp.fc1") or name.endswith(".mlp.fc2")
        )
        is_attn = (".attention.attention." in name) or name.endswith(".attention.output.dense")
        if (include_mlp and is_mlp) or (include_attn and is_attn):
            modules.append(mod)
            names.append(name)
    return modules, names

def apply_unstructured_pruning(
    modules: List[nn.Linear],
    amount: float,
    scope: str = "global",
    remove_reparam: bool = True,
) -> Tuple[int, int]:
    """Apply L1 unstructured pruning.
    - scope="global": global_unstructured across all modules
    - scope="layer": per-module L1 pruning with the same amount
    Returns (zeros, total) across all pruned tensors after optional prune.remove().
    """
    if not modules:
        return 0, 0
    if scope == "global":
        params_to_prune = [(m, "weight") for m in modules]
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    else:
        for m in modules:
            prune.l1_unstructured(m, name="weight", amount=amount)
    if remove_reparam:
        for m in modules:
            try:
                prune.remove(m, "weight")
            except Exception:
                pass
    zeros = 0
    tot = 0
    for m in modules:
        W = m.weight.detach()
        zeros += (W == 0).sum().item()
        tot += W.numel()
    return zeros, tot

# ------------------------------
# Main
# ------------------------------

def main():
    # Auto-detect device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    
    p = argparse.ArgumentParser(description="Selective NeuronRank pruning for I-JEPA (ViT encoder)")
    p.add_argument("--model-id", type=str, default="facebook/ijepa_vith14_1k")
    p.add_argument("--device", type=str, default=default_device, help=f"Device to use (auto-detected: {default_device})")
    p.add_argument("--layers", type=str, default="last8", help="which blocks to prune: e.g., 'last8', '24-31', 'all', 'odd', 'list:0,2,5'")
    p.add_argument("--prune-ratio", type=float, default=0.30, help="fraction of MLP hidden channels to remove in selected blocks")
    p.add_argument("--calib-ds", type=str, default="imagenette", help="'imagenette' or a local folder for datasets.ImageFolder")
    p.add_argument("--calib-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    # NeuronRank hyperparameters (benchmark-style)
    p.add_argument("--nr-activation-threshold", type=float, default=0.05, help="Pre-activation threshold for document frequency (benchmark NR)")
    p.add_argument("--nr-tf-power", type=float, default=1.0, help="Exponent for TF (mean activation) component")
    p.add_argument("--nr-idf-power", type=float, default=1.0, help="Exponent for IDF component")
    p.add_argument("--nr-idf-add", type=float, default=1.0, help="Constant added to IDF before exponentiation")
    p.add_argument("--nr-idf-smooth", type=float, default=1.0, help="Smoothing for IDF numerator/denominator")
    p.add_argument("--nr-weight-power", type=float, default=1.0, help="Exponent for weight magnitude component")
    p.add_argument("--use-benchmark-nr", action="store_true", help="Use benchmark-style NeuronRank (pre-activation TF-IDF) instead of quantile DF")
    
    # Activation stats caching
    p.add_argument("--stats-cache-dir", type=str, default="./activation_stats_cache", help="Directory to cache activation statistics")
    p.add_argument("--force-recollect-stats", action="store_true", help="Force re-collection of activation statistics even if cache exists")
    
    # Old quantile-based NR (deprecated but kept for comparison)
    p.add_argument("--quantile-q", type=float, default=0.90, help="[DEPRECATED] channel-wise activation quantile for DF threshold")
    p.add_argument("--unstructured-method", type=str, choices=["mb", "nrp", "both"], default="both", help="Unstructured criterion: magnitude-based (mb), NeuronRank-based (nrp), or run both")
    p.add_argument("--nrp-quantile-in", type=float, default=0.90, help="Quantile for fc1 input activation threshold (NRP)")
    p.add_argument("--nrp-quantile-out", type=float, default=0.90, help="Quantile for fc1 post-GELU output threshold (NRP)")
    p.add_argument("--nrp-beta", type=float, default=0.0, help="Optional magnitude mixing: score *= |W|**beta (0 = ignore |W|)")
    p.add_argument("--eval", type=str, choices=["none", "knn"], default="none")
    p.add_argument("--compare-mb", action="store_true", help="Also prune a fresh model with magnitude-based channel scores and report k-NN")
    p.add_argument("--eval-train", type=int, default=2000)
    p.add_argument("--eval-val", type=int, default=1000)
    p.add_argument("--save-dir", type=str, default="./ijepa_pruned")
    p.add_argument("--seed", type=int, default=42)
    # Processor speed control
    p.add_argument("--use-fast-processor", dest="use_fast_processor", action="store_true", help="Use fast image processor if available (default)")
    p.add_argument("--no-fast-processor", dest="use_fast_processor", action="store_false", help="Force slow image processor")
    p.set_defaults(use_fast_processor=True)
    p.add_argument("--mode", type=str, choices=["structured", "unstructured"], default="structured", help="Structured (channel) vs unstructured (element) pruning")
    p.add_argument("--unstructured-scope", type=str, choices=["global", "layer"], default="global", help="Global magnitude or per-layer magnitude pruning")
    p.add_argument("--include-attn", action="store_true", help="Include attention (q/k/v and attn out) in unstructured pruning")
    p.add_argument("--no-mlp", action="store_true", help="Exclude MLP (fc1/fc2) from unstructured pruning")
    p.add_argument("--remove-reparam", action="store_true", help="Call prune.remove() to make zeros permanent on weights")

    args = p.parse_args()
    seed_all(args.seed)

    print(f"Loading model: {args.model_id}")
    # Prefer fast processor to avoid slow-processor warning, with safe fallback
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, use_fast=args.use_fast_processor)
    except TypeError:
        # Older transformers may not accept use_fast for this processor; fall back silently
        processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)

    # Discover all MLPs and choose targets
    mlps = find_mlp_modules(model)
    target_indices = sorted(set([i for i in parse_layer_selection(args.layers, total=len(mlps)) if 0 <= i < len(mlps)]))
    target_mlps = [m for m in mlps if m.idx in target_indices]
    print(f"Discovered {len(mlps)} blocks with MLPs; targeting indices: {target_indices}")

    # Baseline params
    base_params = count_params(model)
    print(f"\nParams (before): {base_params/1e6:.2f}M")

    # Optional baseline evaluation BEFORE pruning
    acc_base = None
    if args.eval != "none":
        if args.calib_ds == "imagenette":
            eval_root = auto_download_imagenette()
        else:
            eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
        print(f"Running baseline k-NN (local ImageFolder at {eval_root})…")
        acc_base = knn_probe_local(model, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
        print(f"Baseline k-NN@20: top-1 = {acc_base*100:.2f}%\n")

    # === Unstructured path ===
    if args.mode == "unstructured":
        target_indices = sorted(set([i for i in parse_layer_selection(args.layers, total=len(mlps)) if 0 <= i < len(mlps)]))

        # Always compare on fc1 only for fairness (NRP defined on fc1)
        fc1_list = collect_fc1_modules_for_blocks(model, target_indices)  # [(name, module)]
        print(f"Unstructured fc1 targets: {len(fc1_list)} modules in blocks {target_indices}")
        if fc1_list:
            print("  e.g.", ", ".join([n for n,_ in fc1_list[:min(4, len(fc1_list))]]))

        # Containers for results
        acc_un_nrp = None
        acc_un_mb = None

        # ---- NRP variant (benchmark-style TF-IDF) ----
        if args.unstructured_method in ("nrp", "both"):
            print("\n[Unstructured-NRP] Collecting per-weight TF-IDF scores (benchmark-style)…")
            model_nrp = copy.deepcopy(model)
            # Re-collect fc1 modules on the copied model
            fc1_list_nrp = collect_fc1_modules_for_blocks(model_nrp, target_indices)
            
            # Collect activation statistics (with caching)
            import hashlib
            cache_key_str = f"{args.model_id}_unstr_{args.layers}_{args.calib_ds}_{args.calib_samples}_{args.nr_activation_threshold}"
            cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
            cache_dir = os.path.expanduser(args.stats_cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"activation_stats_{cache_key}.pt")
            
            activation_stats_unstr = None
            if not args.force_recollect_stats and os.path.exists(cache_file):
                try:
                    print(f"  Loading cached activation statistics from: {cache_file}")
                    activation_stats_unstr = torch.load(cache_file, map_location='cpu')
                    print(f"  ✓ Loaded cached stats (threshold={args.nr_activation_threshold})")
                except Exception as e:
                    print(f"  Warning: Failed to load cache ({e}), re-collecting...")
                    activation_stats_unstr = None
            
            if activation_stats_unstr is None:
                activation_stats_unstr = collect_activation_stats_for_fc1_modules(
                    model=model_nrp,
                    processor=processor,
                    dataset_name=args.calib_ds,
                    num_images=args.calib_samples,
                    target_fc1=fc1_list_nrp,
                    batch_size=args.batch_size,
                    activation_threshold=args.nr_activation_threshold,
                    device=args.device,
                )
                try:
                    torch.save(activation_stats_unstr, cache_file)
                    print(f"  ✓ Saved activation statistics to cache: {cache_file}")
                except Exception as e:
                    print(f"  Warning: Failed to save cache ({e})")
            
            # Compute TF-IDF scores
            score_map = compute_unstructured_tfidf_scores(
                target_fc1=fc1_list_nrp,
                activation_stats=activation_stats_unstr,
                tf_power=args.nr_tf_power,
                idf_power=args.nr_idf_power,
                idf_add=args.nr_idf_add,
                idf_smooth=args.nr_idf_smooth,
                weight_power=args.nr_weight_power,
            )
            zeros, tot = apply_unstructured_from_scores(score_map, amount=args.prune_ratio, scope=args.unstructured_scope, remove_reparam=args.remove_reparam)
            spars = 100.0 * zeros / max(1, tot)
            print(f"[Unstructured-NRP] Applied {args.prune_ratio:.2f} ({args.unstructured_scope}) → sparsity {spars:.2f}% over fc1 weights")
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    acc_un_nrp = knn_probe(model_nrp, processor, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
                else:
                    root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                    acc_un_nrp = knn_probe_local(model_nrp, processor, root=root, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
                print(f"[Unstructured-NRP] k-NN@20: top-1 = {acc_un_nrp*100:.2f}%  |  Δ vs base = {(acc_un_nrp - (acc_base or acc_un_nrp)) * 100:.2f}%")
            # Save
            out_dir_nrp = args.save_dir.rstrip("/") + "_unstr_nrp"
            os.makedirs(out_dir_nrp, exist_ok=True)
            print(f"Saving unstructured-NRP model to: {out_dir_nrp}")
            model_nrp.save_pretrained(out_dir_nrp)

        # ---- Magnitude variant ----
        if args.unstructured_method in ("mb", "both"):
            print("\n[Unstructured-MB] Applying L1 global/per-layer pruning on fc1…")
            model_mb = copy.deepcopy(model)
            # Collect fc1 modules on the copied model
            fc1_list_mb = collect_fc1_modules_for_blocks(model_mb, target_indices)
            mods_mb = [m for _, m in fc1_list_mb]
            zeros_mb, tot_mb = apply_unstructured_pruning(mods_mb, amount=args.prune_ratio, scope=args.unstructured_scope, remove_reparam=args.remove_reparam)
            spars_mb = 100.0 * zeros_mb / max(1, tot_mb)
            print(f"[Unstructured-MB] Applied {args.prune_ratio:.2f} ({args.unstructured_scope}) → sparsity {spars_mb:.2f}% over fc1 weights")
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    acc_un_mb = knn_probe(model_mb, processor, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
                else:
                    root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                    acc_un_mb = knn_probe_local(model_mb, processor, root=root, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
                print(f"[Unstructured-MB]  k-NN@20: top-1 = {acc_un_mb*100:.2f}%  |  Δ vs base = {(acc_un_mb - (acc_base or acc_un_mb)) * 100:.2f}%")
            # Save
            out_dir_mb = args.save_dir.rstrip("/") + "_unstr_mb"
            os.makedirs(out_dir_mb, exist_ok=True)
            print(f"Saving unstructured-MB model to: {out_dir_mb}")
            model_mb.save_pretrained(out_dir_mb)

        # ---- Summary ----
        if args.eval != "none":
            print("\n=== Unstructured Summary (k-NN@20 top-1) ===")
            if acc_base is not None: print(f"Base : {acc_base*100:.2f}%")
            if acc_un_nrp is not None: print(f"NRP  : {acc_un_nrp*100:.2f}%  (Δ {((acc_un_nrp - acc_base) if acc_base is not None else 0)*100:.2f}%)")
            if acc_un_mb  is not None: print(f"MB   : {acc_un_mb*100:.2f}%  (Δ {((acc_un_mb  - acc_base) if acc_base is not None else 0)*100:.2f}%)")
        print("Done.")
        return

    # Calibrate NeuronRank scores
    if args.use_benchmark_nr:
        # Build cache key based on model, layers, dataset, samples, and activation threshold
        import hashlib
        cache_key_str = f"{args.model_id}_{args.layers}_{args.calib_ds}_{args.calib_samples}_{args.nr_activation_threshold}"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_dir = os.path.expanduser(args.stats_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"activation_stats_{cache_key}.pt")
        
        # Try to load cached stats
        activation_stats = None
        if not args.force_recollect_stats and os.path.exists(cache_file):
            try:
                print(f"Loading cached activation statistics from: {cache_file}")
                activation_stats = torch.load(cache_file, map_location='cpu')
                print(f"✓ Loaded cached stats (collected with threshold={args.nr_activation_threshold})")
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}), re-collecting...")
                activation_stats = None
        
        # Collect if not cached or forced
        if activation_stats is None:
            print("Collecting NeuronRank scores (benchmark-style: pre-activation TF-IDF)…")
            activation_stats = collect_activation_statistics_ijepa(
                model=model,
                processor=processor,
                dataset_name=args.calib_ds,
                num_images=args.calib_samples,
                target_mlps=target_mlps,
                batch_size=args.batch_size,
                activation_threshold=args.nr_activation_threshold,
                device=args.device,
            )
            # Save to cache
            try:
                torch.save(activation_stats, cache_file)
                print(f"✓ Saved activation statistics to cache: {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to save cache ({e})")
        
        scores = compute_neuronrank_scores_ijepa(
            target_mlps=target_mlps,
            activation_stats=activation_stats,
            tf_power=args.nr_tf_power,
            idf_power=args.nr_idf_power,
            idf_add=args.nr_idf_add,
            idf_smooth=args.nr_idf_smooth,
            weight_power=args.nr_weight_power,
        )
    else:
        print("Collecting NeuronRank scores (quantile-based DF, post-activation)…")
        scores = collect_neuronrank_scores(
            model=model,
            processor=processor,
            dataset_name=args.calib_ds,
            num_images=args.calib_samples,
            target_mlps=target_mlps,
            batch_size=args.batch_size,
            quantile_q=args.quantile_q,
            device=args.device,
        )

    # Apply structured pruning (NRP)
    print(f"Applying structured pruning (NRP): ratio={args.prune_ratio:.2f} on {len(target_mlps)} blocks…")
    pruned = apply_structured_mlp_pruning(
        model=model,
        processor=processor,
        target_mlps=target_mlps,
        scores=scores,
        prune_ratio=args.prune_ratio,
        device=args.device,
    )
    new_params = count_params(model)
    print(f"Pruned channels: ~{pruned} (aggregated across blocks)")
    print(f"{''}\nParams (after NRP): {new_params/1e6:.2f}M  |  Δ = {(base_params - new_params)/1e6:.2f}M\n")

    # Evaluate NRP
    acc_nrp = None
    if args.eval != "none":
        if args.calib_ds == "imagenette":
            print("Evaluating NRP model with k-NN (HF Imagenette)…")
            acc_nrp = knn_probe(model, processor, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
        else:
            root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
            print(f"Evaluating NRP model with k-NN (local ImageFolder at {root})…")
            acc_nrp = knn_probe_local(model, processor, root=root, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
        print(f"NRP k-NN@20: top-1 = {acc_nrp*100:.2f}%  |  Δ vs base = {(acc_nrp - (acc_base or acc_nrp)) * 100:.2f}%\n")

    # Optional magnitude-based comparison
    if args.compare_mb:
        print("\n=== Magnitude-based pruning comparison ===")
        model_mb = AutoModel.from_pretrained(args.model_id)
        mlps_mb = find_mlp_modules(model_mb)
        target_mb = [m for m in mlps_mb if m.idx in target_indices]
        mb_scores = compute_magnitude_scores(target_mb)
        print(f"Applying structured pruning (MBP): ratio={args.prune_ratio:.2f} on {len(target_mb)} blocks…")
        _ = apply_structured_mlp_pruning(
            model=model_mb,
            processor=processor,
            target_mlps=target_mb,
            scores=mb_scores,
            prune_ratio=args.prune_ratio,
            device=args.device,
        )
        if args.eval != "none":
            if args.calib_ds == "imagenette":
                acc_mb = knn_probe(model_mb, processor, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
            else:
                root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                acc_mb = knn_probe_local(model_mb, processor, root=root, train_n=args.eval_train, val_n=args.eval_val, device=args.device)
            print(f"MBP k-NN@20: top-1 = {acc_mb*100:.2f}%  |  Δ vs base = {(acc_mb - (acc_base or acc_mb)) * 100:.2f}%")
        # Save MBP too
        mb_dir = args.save_dir.rstrip("/") + "_mbp"
        os.makedirs(mb_dir, exist_ok=True)
        print(f"Saving magnitude-pruned model to: {mb_dir}")
        model_mb.save_pretrained(mb_dir)

    # Save NRP model
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nSaving NRP-pruned model to: {args.save_dir}")
    model.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)

    # Summary
    if args.eval != "none":
        print("\n=== Summary (k-NN@20 top-1) ===")
        if acc_base is not None: print(f"Base : {acc_base*100:.2f}%")
        if acc_nrp is not None:  print(f"NRP  : {acc_nrp*100:.2f}%  (Δ {((acc_nrp - acc_base) if acc_base is not None else 0)*100:.2f}%)")
        if args.compare_mb and 'acc_mb' in locals():
            print(f"MBP  : {acc_mb*100:.2f}%  (Δ {((acc_mb - acc_base) if acc_base is not None else 0)*100:.2f}%)")

    print("Done.")


# ------------------------------
# Unstructured NeuronRank per-weight scoring (benchmark-style TF-IDF)
# ------------------------------
@torch.no_grad()
def collect_activation_stats_for_fc1_modules(
    model: nn.Module,
    processor: AutoProcessor,
    dataset_name: str,
    num_images: int,
    target_fc1: List[Tuple[str, nn.Linear]],
    batch_size: int = 16,
    activation_threshold: float = 0.05,
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect pre-activation statistics for fc1 modules (benchmark-style).
    
    Returns dict: {module_name: {'mean_abs_activation', 'doc_freq', 'sample_count'}}
    """
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    # Register hooks on fc1 modules (pre-activation)
    for name, fc1 in target_fc1:
        def make_hook(layer_name: str):
            def hook(_module, inputs, _output):
                if not inputs:
                    return
                features = inputs[0]  # [B, T, in_features]
                if features is None:
                    return
                features = features.detach()
                # Flatten batch & sequence: [B*T, in_features]
                if features.dim() == 3:
                    flattened = features.reshape(-1, features.shape[-1]).abs()
                elif features.dim() == 2:
                    flattened = features.abs()
                else:
                    flattened = features.abs()
                
                flattened = flattened.to(dtype=torch.float32, device='cpu')
                present = (flattened > activation_threshold).to(dtype=torch.float32)
                
                layer_stats = stats.setdefault(layer_name, {
                    'sum_abs_activation': torch.zeros(flattened.size(-1), dtype=torch.float32),
                    'doc_freq': torch.zeros(flattened.size(-1), dtype=torch.float32),
                    'sample_count': 0,
                })
                layer_stats['sum_abs_activation'] += flattened.sum(dim=0)
                layer_stats['doc_freq'] += present.sum(dim=0)
                layer_stats['sample_count'] += flattened.size(0)
            return hook
        
        handles.append(fc1.register_forward_hook(make_hook(name)))
    
    # Load local images
    model.eval().to(device)
    
    # Auto-download if "imagenette" keyword is used
    if dataset_name == "imagenette":
        root = auto_download_imagenette()
    else:
        root = os.path.expanduser(dataset_name)
    
    train_dir = os.path.join(root, "train") if os.path.isdir(os.path.join(root, "train")) else root
    paths: List[Path] = []
    for p in Path(train_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No images found under: {train_dir}")
    random.shuffle(paths)
    paths = paths[: max(1, min(num_images, len(paths)))]
    
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        _ = model(**inputs)
        for im in imgs:
            try: im.close()
            except Exception: pass
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Compute mean
    for layer_name, layer_stats in stats.items():
        count = layer_stats['sample_count']
        if count > 0:
            layer_stats['mean_abs_activation'] = layer_stats['sum_abs_activation'] / count
        else:
            layer_stats['mean_abs_activation'] = torch.zeros_like(layer_stats['sum_abs_activation'])
        layer_stats['doc_freq'] = layer_stats['doc_freq'].clamp_(min=0.0, max=float(count))
        del layer_stats['sum_abs_activation']
    
    return stats


@torch.no_grad()
def compute_unstructured_tfidf_scores(
    target_fc1: List[Tuple[str, nn.Linear]],
    activation_stats: Dict[str, Dict[str, torch.Tensor]],
    *,
    tf_power: float = 1.0,
    idf_power: float = 1.0,
    idf_add: float = 1.0,
    idf_smooth: float = 1.0,
    weight_power: float = 1.0,
) -> Dict[nn.Linear, torch.Tensor]:
    """Compute per-weight TF-IDF NeuronRank scores for fc1 modules.
    
    Returns {module: scores[out, in]} matching weight shape.
    """
    score_map: Dict[nn.Linear, torch.Tensor] = {}
    
    for name, fc1 in target_fc1:
        stats = activation_stats.get(name)
        if not stats:
            continue
        
        sample_count = int(stats.get('sample_count', 0))
        if sample_count == 0:
            continue
        
        mean_abs_activation = stats.get('mean_abs_activation')
        doc_freq = stats.get('doc_freq')
        if mean_abs_activation is None or doc_freq is None:
            continue
        
        mean_abs_activation = mean_abs_activation.to(torch.float32)
        doc_freq = doc_freq.to(torch.float32)
        
        weight = fc1.weight.detach().to(dtype=torch.float32, device='cpu')  # [out, in]
        if weight.numel() == 0:
            continue
        
        # TF component: per-input-channel mean activation
        tf_component = mean_abs_activation.clamp(min=0.0).pow(tf_power)  # [in]
        
        # IDF component: per-input-channel log(N / DF)
        smooth = idf_smooth if idf_smooth > 0 else 0.0
        numerator_value = float(sample_count) + smooth + EPS
        numerator = torch.tensor(numerator_value, dtype=torch.float32)
        denominator = doc_freq + smooth + EPS
        idf_component = torch.log(numerator / denominator)
        if idf_add != 0.0:
            idf_component = idf_component + idf_add
        idf_component = idf_component.clamp(min=0.0).pow(idf_power)  # [in]
        
        # Weight component: per-weight magnitude
        weight_component = weight.abs().pow(weight_power)  # [out, in]
        
        # Combine: per-weight scores = |W[j,i]|^p * TF[i] * IDF[i]
        tf_idf = tf_component * idf_component  # [in]
        per_weight_scores = weight_component * tf_idf.unsqueeze(0)  # [out, in]
        
        score_map[fc1] = per_weight_scores
    
    return score_map


# ------------------------------
# Apply masks from per-weight scores
# ------------------------------
def apply_unstructured_from_scores(
    score_map: Dict[nn.Linear, torch.Tensor],
    amount: float,
    scope: str = "global",
    remove_reparam: bool = True,
) -> Tuple[int, int]:
    mods = list(score_map.keys())
    if not mods:
        return 0, 0
    zeros = 0
    tot = 0
    if scope == "global":
        all_scores = torch.cat([score_map[m].reshape(-1) for m in mods])
        k = int(amount * all_scores.numel())
        if k <= 0:
            thr = -float("inf")
        elif k >= all_scores.numel():
            thr = float("inf")
        else:
            thr = torch.kthvalue(all_scores, k).values.item()
        for m in mods:
            s = score_map[m]
            mask = (s > thr).to(m.weight.device)
            prune.custom_from_mask(m, name="weight", mask=mask)
            if remove_reparam:
                try: prune.remove(m, "weight")
                except Exception: pass
            W = m.weight.detach()
            zeros += (W == 0).sum().item()
            tot += W.numel()
    else:  # per-layer
        for m in mods:
            s = score_map[m].reshape(-1)
            k = int(amount * s.numel())
            if k <= 0:
                thr = -float("inf")
            elif k >= s.numel():
                thr = float("inf")
            else:
                thr = torch.kthvalue(s, k).values.item()
            mask = (score_map[m] > thr).to(m.weight.device)
            prune.custom_from_mask(m, name="weight", mask=mask)
            if remove_reparam:
                try: prune.remove(m, "weight")
                except Exception: pass
            W = m.weight.detach()
            zeros += (W == 0).sum().item()
            tot += W.numel()
    return zeros, tot


if __name__ == "__main__":
    main()