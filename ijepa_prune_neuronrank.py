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
- **optional for linear probe**: scikit-learn (if you prefer sklearn's LogisticRegression, otherwise the script uses k-NN)

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
from torch.utils.hooks import RemovableHandle

try:
    import torch_pruning as tp
except Exception as e:
    raise SystemExit("Please install torch-pruning: pip install torch-pruning")

from transformers import AutoModel, AutoProcessor
from datasets import load_dataset

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Graph visualization will be skipped.")

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


def estimate_model_size_gb(model: nn.Module) -> float:
    """Estimate model size in GB (assumes float32)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**3)


def count_target_mlp_params(target_mlps: List) -> int:
    """Count parameters in target MLP layers (fc1 + fc2)."""
    total = 0
    for mlp_handle in target_mlps:
        # fc1 weight + bias
        total += mlp_handle.fc1.weight.numel()
        if mlp_handle.fc1.bias is not None:
            total += mlp_handle.fc1.bias.numel()
        # fc2 weight + bias
        total += mlp_handle.fc2.weight.numel()
        if mlp_handle.fc2.bias is not None:
            total += mlp_handle.fc2.bias.numel()
    return total


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


def auto_download_imagenette(dataset_dir: str = "./datasets") -> str:
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
            # Use filter for Python 3.14+ compatibility
            try:
                tar.extractall(path=dataset_dir, filter='data')
            except TypeError:
                # Older Python versions don't support filter parameter
                tar.extractall(path=dataset_dir)
        print(f"✓ Extracted to: {target_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract Imagenette2-320: {e}")
    
    # Verify
    if not os.path.isdir(os.path.join(target_path, "train")):
        raise RuntimeError(f"Extraction succeeded but 'train' folder not found at {target_path}")
    
    return target_path


# ------------------------------
# Post-Activation Class-Variance NeuronRank (output neurons, class-aware)
# ------------------------------
@torch.no_grad()
def collect_postactivation_stats(
    model: nn.Module,
    processor: AutoProcessor,
    dataset_name: str,
    num_images: int,
    target_mlps: List[MLPHandle],
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Collect post-GELU activation statistics for variance-based scoring.
    
    Returns dict: {module_name: {
        'mean_abs_activation': [out_channels],
        'sample_count': int,
        'per_class': {class_id: {'mean_abs_activation', 'sample_count'}}
    }}
    """
    stats: Dict[str, Dict] = {"_meta": {"collect_per_class": True}}
    gelu = nn.GELU()
    handles: List[RemovableHandle] = []
    current_batch_label_tensor: Optional[torch.Tensor] = None
    
    # Register hooks on fc1 modules (post-activation)
    for h in target_mlps:
        name = h.name + ".intermediate.dense" if h.name.startswith("encoder.layer") else h.name + ".fc1"
        
        def make_hook(layer_name: str):
            def hook(_module, _inputs, output):
                nonlocal current_batch_label_tensor
                # Apply GELU to get post-activation
                a = gelu(output)  # [B, T, out_features]
                
                # Flatten batch & sequence
                if a.dim() == 3:
                    a_flat = a.flatten(0, 1)  # [B*T, out_features]
                    tokens_per_sample = a.shape[1]
                elif a.dim() == 2:
                    a_flat = a
                    tokens_per_sample = 1
                else:
                    a_flat = a
                    tokens_per_sample = 1
                
                a_flat = a_flat.to(dtype=torch.float32, device='cpu')
                abs_act = a_flat.abs()
                
                # Overall pooled stats
                layer_stats = stats.setdefault(layer_name, {
                    'sum_abs_activation': torch.zeros(a_flat.size(-1), dtype=torch.float32),
                    'sample_count': 0,
                    'per_class': {},
                })
                layer_stats['sum_abs_activation'] += abs_act.sum(dim=0)
                layer_stats['sample_count'] += a_flat.size(0)

                if layer_stats['per_class'] is not None and current_batch_label_tensor is not None:
                    labels_tensor = current_batch_label_tensor
                    if tokens_per_sample > 1:
                        labels_tensor = labels_tensor.repeat_interleave(tokens_per_sample)
                    if labels_tensor.numel() != a_flat.size(0):
                        return
                    per_class_stats = layer_stats['per_class']
                    labels_unique = labels_tensor.unique(sorted=True)
                    for cls_id in labels_unique.tolist():
                        cls_mask = labels_tensor == cls_id
                        if not torch.any(cls_mask):
                            continue
                        cls_values = abs_act[cls_mask]
                        cls_entry = per_class_stats.setdefault(cls_id, {
                            'sum_abs_activation': torch.zeros(a_flat.size(-1), dtype=torch.float32),
                            'sample_count': 0,
                        })
                        cls_entry['sum_abs_activation'] += cls_values.sum(dim=0)
                        cls_entry['sample_count'] += cls_values.size(0)
            return hook
        
        handles.append(h.fc1.register_forward_hook(make_hook(name)))
    
    # Load calibration dataset
    model.eval().to(device)
    
    if dataset_name == "imagenette":
        root = auto_download_imagenette()
    else:
        root = os.path.expanduser(dataset_name)
    
    train_dir = os.path.join(root, "train") if os.path.isdir(os.path.join(root, "train")) else root
    
    # Load images with labels for per-class stats
    from pathlib import Path
    cls_dirs = sorted([d for d in Path(train_dir).iterdir() if d.is_dir()])
    cls_to_idx = {d.name: i for i, d in enumerate(cls_dirs)}
    
    items: List[Tuple[Path, int]] = []
    for cls_name, idx in cls_to_idx.items():
        d = Path(train_dir) / cls_name
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                items.append((p, idx))
    
    if not items:
        raise RuntimeError(f"No images found under: {train_dir}")
    
    # Sample deterministically
    items = sorted(items, key=lambda x: str(x[0]))
    sample_rng = random.Random(42)
    sample_rng.shuffle(items)
    items = items[:min(num_images, len(items))]
    
    # Process batches
    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p, _ in batch_items]
        labels = [lbl for _, lbl in batch_items]
        
        current_batch_label_tensor = torch.tensor(labels, dtype=torch.long)
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        _ = model(**inputs)
        current_batch_label_tensor = None
        
        for im in imgs:
            try: im.close()
            except Exception: pass
    
    for handle in handles:
        handle.remove()

    # Compute means
    for layer_name, layer_stats in stats.items():
        if layer_name == "_meta":
            continue
        count = layer_stats['sample_count']
        if count > 0:
            layer_stats['mean_abs_activation'] = layer_stats['sum_abs_activation'] / count
        else:
            layer_stats['mean_abs_activation'] = torch.zeros_like(layer_stats['sum_abs_activation'])
        del layer_stats['sum_abs_activation']

        if layer_stats['per_class']:
            for cls_id, cls_stats in layer_stats['per_class'].items():
                cls_count = cls_stats['sample_count']
                if cls_count > 0:
                    cls_stats['mean_abs_activation'] = cls_stats['sum_abs_activation'] / cls_count
                else:
                    cls_stats['mean_abs_activation'] = torch.zeros_like(cls_stats['sum_abs_activation'])
                del cls_stats['sum_abs_activation']
    
    return stats


@torch.no_grad()
def compute_variance_magnitude_scores(
    target_mlps: List[MLPHandle],
    activation_stats: Dict[str, Dict],
    discrimination_weight: float = 1.0,
    beta: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """
    Computes a hybrid score for each channel by combining class-discriminative
    variance with the magnitude (L2 norm) of its corresponding weights.
    
    Score = (Variance^discrimination_weight) * (beta × ||W||₂)
    
    This rewards channels that are both selective and have large weights.
    
    Returns {block_idx: scores[out_channels]} for structured pruning.
    """
    score_map: Dict[int, torch.Tensor] = {}

    for h in target_mlps:
        name = h.name + ".intermediate.dense" if h.name.startswith("encoder.layer") else h.name + ".fc1"
        stats = activation_stats.get(name)

        if not stats or not stats.get('per_class'):
            print(f"[warn] No per-class stats for {name}, cannot compute score. Skipping.")
            continue

        per_class_stats = stats['per_class']
        class_means = [
            s['mean_abs_activation']
            for s in per_class_stats.values()
            if s.get('sample_count', 0) > 0 and 'mean_abs_activation' in s
        ]

        if not class_means:
            num_channels = h.fc1.out_features
            score_map[h.idx] = torch.zeros(num_channels)
            continue
            
        class_means_tensor = torch.stack(class_means, dim=0)

        # 1. Discrimination Score: Variance across classes
        variance_score = torch.var(class_means_tensor, dim=0, unbiased=False)
        
        # 2. Impact Score: L2 norm of fc1 outgoing weights
        weights = h.fc1.weight.detach().to(dtype=torch.float32, device='cpu')
        magnitude_score = torch.norm(weights, p=2, dim=1)
        
        # Normalize both scores to [0, 1] to ensure fair contribution
        var_min, var_max = variance_score.min(), variance_score.max()
        if var_max > var_min:
            variance_score = (variance_score - var_min) / (var_max - var_min)

        mag_min, mag_max = magnitude_score.min(), magnitude_score.max()
        if mag_max > mag_min:
            magnitude_score = (magnitude_score - mag_min) / (mag_max - mag_min)
            
        # Apply discrimination weight: Score = (Variance^weight) * Magnitude
        if discrimination_weight != 1.0:
            variance_score = variance_score.pow(discrimination_weight)
        
        # Apply beta multiplier to magnitude score
        if beta != 1.0:
            magnitude_score = magnitude_score * beta
            
        # Final Score = Weighted Discrimination * Impact
        channel_scores = variance_score * magnitude_score
        
        score_map[h.idx] = channel_scores

    return score_map


@torch.no_grad()
def compute_variance_magnitude_scores_fc2(
    target_mlps: List[MLPHandle],
    activation_stats: Dict[str, Dict],
    discrimination_weight: float = 1.0,
    beta: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """
    Computes a hybrid score for each channel by combining class-discriminative
    variance with the magnitude (L2 norm) of its OUTGOING weights in fc2.
    
    Score = (Variance^discrimination_weight) * (beta × ||W_fc2||₂)
    
    This variant uses the fc2 weights (downstream impact) instead of fc1 weights.
    For each intermediate channel j, we take the L2 norm of column j in fc2.weight,
    which represents how strongly that channel influences the next layer.
    
    Returns {block_idx: scores[out_channels]} for structured pruning.
    """
    score_map: Dict[int, torch.Tensor] = {}

    for h in target_mlps:
        name = h.name + ".intermediate.dense" if h.name.startswith("encoder.layer") else h.name + ".fc1"
        stats = activation_stats.get(name)

        if not stats or not stats.get('per_class'):
            print(f"[warn] No per-class stats for {name}, cannot compute score. Skipping.")
            continue

        per_class_stats = stats['per_class']
        class_means = [
            s['mean_abs_activation']
            for s in per_class_stats.values()
            if s.get('sample_count', 0) > 0 and 'mean_abs_activation' in s
        ]

        if not class_means:
            num_channels = h.fc1.out_features
            score_map[h.idx] = torch.zeros(num_channels)
            continue
            
        class_means_tensor = torch.stack(class_means, dim=0)

        # 1. Discrimination Score: Variance across classes (same as original)
        variance_score = torch.var(class_means_tensor, dim=0, unbiased=False)
        
        # 2. Impact Score: L2 norm of fc2 COLUMN weights (downstream impact)
        # fc2.weight shape: [out_features, intermediate_channels]
        # We want the norm of each column (dimension 0), representing the outgoing impact
        weights_fc2 = h.fc2.weight.detach().to(dtype=torch.float32, device='cpu')
        magnitude_score = torch.norm(weights_fc2, p=2, dim=0)  # Note: dim=0 for columns
        
        # Normalize both scores to [0, 1] to ensure fair contribution
        var_min, var_max = variance_score.min(), variance_score.max()
        if var_max > var_min:
            variance_score = (variance_score - var_min) / (var_max - var_min)

        mag_min, mag_max = magnitude_score.min(), magnitude_score.max()
        if mag_max > mag_min:
            magnitude_score = (magnitude_score - mag_min) / (mag_max - mag_min)
            
        # Apply discrimination weight: Score = (Variance^weight) * Magnitude
        if discrimination_weight != 1.0:
            variance_score = variance_score.pow(discrimination_weight)
        
        # Apply beta multiplier to magnitude score
        if beta != 1.0:
            magnitude_score = magnitude_score * beta
            
        # Final Score = Weighted Discrimination * Impact
        channel_scores = variance_score * magnitude_score
        
        score_map[h.idx] = channel_scores

    return score_map


@torch.no_grad()
def compute_postactivation_tfidf_scores(
    target_mlps: List[MLPHandle],
    activation_stats: Dict[str, Dict],
    *,
    tf_power: float = 1.0,
    idf_power: float = 1.0,
    idf_add: float = 1.0,
    idf_smooth: float = 1.0,
    weight_power: float = 0.0,  # Optional weight mixing
    use_per_class: bool = False,
    per_class_agg: str = "max",
    per_class_power: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """Compute per-channel scores from post-activation TF-IDF stats.
    
    Returns {block_idx: scores[out_channels]} for structured pruning.
    """
    score_map: Dict[int, torch.Tensor] = {}
    cache_meta = activation_stats.get("_meta", {}) if isinstance(activation_stats, dict) else {}
    stats_have_per_class = bool(cache_meta.get("collect_per_class", False))
    
    per_class_agg = (per_class_agg or "max").lower()

    for h in target_mlps:
        name = h.name + ".intermediate.dense" if h.name.startswith("encoder.layer") else h.name + ".fc1"
        stats = activation_stats.get(name)
        if not stats:
            continue
        
        if use_per_class and stats_have_per_class and not stats.get('per_class'):
            print(f"[warn] Block {h.idx} lacks per-class stats in cache; falling back to global TF-IDF")

        sample_count = int(stats.get('sample_count', 0))
        if sample_count == 0:
            continue
        
        mean_abs_activation = stats.get('mean_abs_activation')
        doc_freq = stats.get('doc_freq')
        if mean_abs_activation is None or doc_freq is None:
            continue
        
        mean_abs_activation = mean_abs_activation.to(torch.float32)
        doc_freq = doc_freq.to(torch.float32)
        
        # TF component: per-output-channel mean activation (how strong it fires)
        tf_component = mean_abs_activation.clamp(min=0.0).pow(tf_power)  # [out]
        
        # IDF component: per-output-channel rarity (how selective it is)
        smooth = idf_smooth if idf_smooth > 0 else 0.0
        numerator_value = float(sample_count) + smooth + EPS
        numerator = torch.tensor(numerator_value, dtype=torch.float32)
        denominator = doc_freq + smooth + EPS
        idf_component = torch.log(numerator / denominator)
        if idf_add != 0.0:
            idf_component = idf_component + idf_add
        idf_component = idf_component.clamp(min=0.0).pow(idf_power)  # [out]
        
        # Base score: TF × IDF
        channel_scores = tf_component * idf_component  # [out]
        
        # Optional per-class discrimination
        if use_per_class and stats_have_per_class and stats.get('per_class'):
            per_class_entries = [v for v in stats['per_class'].values() if v.get('sample_count', 0) > 0]
            if per_class_entries:
                per_class_tfidf: List[torch.Tensor] = []
                doc_freq_total = doc_freq + idf_smooth + EPS
                for cls_stats in per_class_entries:
                    cls_count = float(cls_stats.get('sample_count', 0))
                    if cls_count <= 0:
                        continue
                    cls_mean = cls_stats.get('mean_abs_activation')
                    cls_df = cls_stats.get('doc_freq')
                    if cls_mean is None or cls_df is None:
                        continue
                    cls_mean = cls_mean.to(torch.float32)
                    cls_df = cls_df.to(torch.float32)
                    tf_cls = cls_mean.clamp(min=0.0).pow(tf_power)
                    # Selectivity relative to global frequency: log((global DF + smooth)/(class DF + smooth))
                    cls_denominator = cls_df + idf_smooth + EPS
                    selectivity = torch.log(torch.clamp(doc_freq_total / cls_denominator, min=1.0))
                    if idf_add != 0.0:
                        selectivity = selectivity + idf_add
                    selectivity = selectivity.clamp(min=0.0).pow(idf_power)
                    per_class_tfidf.append(tf_cls * selectivity)

                if per_class_tfidf:
                    cls_stack = torch.stack(per_class_tfidf, dim=0)
                    if per_class_agg == "max":
                        discrim = cls_stack.max(dim=0).values
                    elif per_class_agg == "mean":
                        discrim = cls_stack.mean(dim=0)
                    elif per_class_agg == "sum":
                        discrim = cls_stack.sum(dim=0)
                    elif per_class_agg == "var":
                        discrim = cls_stack.var(dim=0, unbiased=False)
                    elif per_class_agg == "std":
                        discrim = cls_stack.var(dim=0, unbiased=False).sqrt()
                    elif per_class_agg == "median":
                        discrim = cls_stack.median(dim=0).values
                    elif per_class_agg == "max_minus_mean":
                        discrim = cls_stack.max(dim=0).values - cls_stack.mean(dim=0)
                    else:
                        raise ValueError(f"Unknown per_class_agg='{per_class_agg}'")
                    discrim = discrim.clamp(min=0.0).pow(per_class_power)
                    # Blend by geometric mean to keep scale stable.
                    channel_scores = torch.sqrt(channel_scores.clamp(min=0.0) * (discrim + EPS))

        # Optional: mix with weight magnitude
        if weight_power > 0.0:
            weight = h.fc1.weight.detach().to(dtype=torch.float32, device='cpu')  # [out, in]
            weight_norms = torch.norm(weight, dim=1).pow(weight_power)  # [out]
            channel_scores = channel_scores * weight_norms
        
        score_map[h.idx] = channel_scores
    
    return score_map


# ------------------------------
# Benchmark-style NeuronRank (pre-activation, TF-IDF) - DEPRECATED
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

    model.eval().to(device)

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

# ------------------------------
# Structural pruning (fc1 out / fc2 in) — explicit pairing via Torch-Pruning
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
    """Prune the same intermediate channel across fc1 (out) and fc2 (in) for every
    target MLP block, using Torch-Pruning's DependencyGraph groups.

    For each block b with score vector s_b (length = hidden/intermediate size),
    we select the lowest `prune_ratio` fraction of indices and create a pruning
    group anchored at fc1 (prune_linear_out_channels), then explicitly **add**
    the paired fc2 in-channels (prune_linear_in_channels) to the same group.

    This makes the coupling explicit and robust to non-trivial graphs, matching
    Torch-Pruning's recommended usage (DepGraph + group.add_pruning_target).
    """
    # Wrap model to create a stable forward for DG tracing
    wrapper = IJepaWrapper(model)
    wrapper.eval().to(device)

    # Create a dummy example input with the processor's expected size
    size = getattr(model.config, "image_size", 224)
    dummy = torch.zeros(1, 3, size, size, device=device)

    # Torch-Pruning's documented pattern is:
    #   group = DG.get_pruning_group(...); group.prune()
    # and it auto-infers coupled layers (no add_* API on Group). See https://github.com/VainF/Torch-Pruning/wiki/Structured-Pruning-of-Transformers
    DG = tp.DependencyGraph().build_dependency(wrapper, example_inputs=(dummy,))

    total_pruned = 0
    for h in target_mlps:
        sc = scores.get(h.idx)
        if sc is None:
            print(f"[warn] No scores for block {h.idx}; skipping.")
            continue
        hidden = sc.numel()
        k = int(round(prune_ratio * hidden))
        if k <= 0:
            continue
        # lowest-score indices to remove
        idxs = torch.argsort(sc)[:k].tolist()

        # Build a pruning group anchored at fc1 out-channels
        try:
            group = DG.get_pruning_group(h.fc1, tp.prune_linear_out_channels, idxs=idxs)
        except Exception as e:
            print(f"[warn] Failed to create pruning group for block {h.idx}: {e}")
            continue

        if DG.check_pruning_group(group):
            group.prune()
            total_pruned += len(idxs)
        else:
            print(f"[warn] Skipped block {h.idx}: invalid pruning group (dependency check failed)")
# ------------------------------
# Attention head pruning support
# ------------------------------

# --- Attention handle dataclass and finder ---
from dataclasses import dataclass
import re

@dataclass
class AttnHandle:
    name: str
    attn_mod: nn.Module  # e.g., encoder.layer.X.attention.attention
    query: nn.Linear
    key: nn.Linear
    value: nn.Linear
    out: nn.Linear       # encoder.layer.X.attention.output.dense
    idx: int             # block index
    num_heads: int
    head_dim: int

def find_attention_modules(model: nn.Module) -> List[AttnHandle]:
    """Discover attention (Q,K,V + output.dense) modules per encoder block (ViT/I-JEPA style)."""
    name_to_mod: Dict[str, nn.Module] = dict(model.named_modules())
    attns: List[AttnHandle] = []
    idx = 0
    # Scan encoder.layer.N.attention
    for n, m in list(name_to_mod.items()):
        if not n.endswith(".attention"):
            continue
        # attention container has `.attention` (qkv) and `.output.dense`
        attn_inner = name_to_mod.get(n + ".attention")
        out_mod = name_to_mod.get(n + ".output")
        if attn_inner is None or out_mod is None:
            continue
        q = getattr(attn_inner, "query", None)
        k = getattr(attn_inner, "key", None)
        v = getattr(attn_inner, "value", None)
        o = getattr(out_mod, "dense", None)
        if not (isinstance(q, nn.Linear) and isinstance(k, nn.Linear) and isinstance(v, nn.Linear) and isinstance(o, nn.Linear)):
            continue
        # Resolve heads and head_dim from config and current layer widths
        all_size = q.out_features
        num_heads = getattr(getattr(getattr(model, "config", object()), "num_attention_heads", None), None)  # dummy to ensure getattr chain
        # Fallback: derive from module if model.config not accessible here
        num_heads = getattr(getattr(attn_inner, "num_attention_heads", None), "__int__", lambda: None)() or getattr(model.config, "num_attention_heads", None)
        if num_heads is None or num_heads <= 0:
            # try to infer by dividing out_features by known head size from module
            head_size = getattr(attn_inner, "attention_head_size", None) or (all_size // max(1, getattr(model.config, "num_attention_heads", 1)))
            num_heads = max(1, all_size // max(1, head_size))
        head_dim = all_size // num_heads
        # Determine block index from name (encoder.layer.N)
        mobj = re.search(r"encoder\.layer\.(\d+)", n)
        bidx = int(mobj.group(1)) if mobj else idx
        attns.append(AttnHandle(name=n + ".attention", attn_mod=attn_inner, query=q, key=k, value=v, out=o, idx=bidx, num_heads=num_heads, head_dim=head_dim))
        idx += 1
    return attns

# --- Head-activation scoring collector ---
@torch.no_grad()
def collect_head_activation_scores(
    model: nn.Module,
    processor: AutoProcessor,
    dataset_name: str,
    num_images: int,
    target_attn: List[AttnHandle],
    batch_size: int = 16,
    device: str = "cuda",
    mode: str = "act_l1",
) -> Dict[int, torch.Tensor]:
    """Compute per-head scores by hooking Q/K/V outputs and averaging |.| (or L2) per head across tokens.
    Returns {block_idx: scores[num_heads]}.
    """
    # Storage per block: accumulators for Q,K,V per head
    acc_q: Dict[int, torch.Tensor] = {}
    acc_k: Dict[int, torch.Tensor] = {}
    acc_v: Dict[int, torch.Tensor] = {}
    cnt: Dict[int, int] = {}

    handles: List[RemovableHandle] = []

    def make_hook(bidx: int, which: str, head_dim: int, num_heads: int):
        def hook(_m, _inp, out):
            # out: [B, T, hidden] or [B, hidden]
            if out.dim() == 2:
                # assume [B, hidden] → add T=1
                out_ = out.unsqueeze(1)
            else:
                out_ = out
            B, T, H = out_.shape
            # reshape to heads
            x = out_.reshape(B*T, num_heads, head_dim)  # [BT, H, D]
            if mode == "act_l2":
                head_stat = x.pow(2).mean(dim=(0, 2))  # [H]
            else:
                head_stat = x.abs().mean(dim=(0, 2))   # [H]
            if which == "q":
                acc = acc_q
            elif which == "k":
                acc = acc_k
            else:
                acc = acc_v
            if bidx not in acc:
                acc[bidx] = head_stat.detach().cpu()
            else:
                acc[bidx] += head_stat.detach().cpu()
            cnt[bidx] = cnt.get(bidx, 0) + 1
        return hook

    for ah in target_attn:
        handles.append(ah.query.register_forward_hook(make_hook(ah.idx, "q", ah.head_dim, ah.num_heads)))
        handles.append(ah.key.register_forward_hook(make_hook(ah.idx, "k", ah.head_dim, ah.num_heads)))
        handles.append(ah.value.register_forward_hook(make_hook(ah.idx, "v", ah.head_dim, ah.num_heads)))

    # Load a small local dataset (reusing existing loader logic)
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

    model.eval().to(device)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        _ = model(**inputs)
        for im in imgs:
            try: im.close()
            except Exception: pass

    for h in handles:
        h.remove()

    # Combine Q/K/V stats → simple average
    scores: Dict[int, torch.Tensor] = {}
    for bidx in sorted(cnt.keys()):
        q = acc_q.get(bidx)
        k = acc_k.get(bidx)
        v = acc_v.get(bidx)
        # default zeros if any missing
        H = q.numel() if q is not None else (k.numel() if k is not None else v.numel())
        zero = torch.zeros(H)
        q = q if q is not None else zero
        k = k if k is not None else zero
        v = v if v is not None else zero
        s = (q + k + v) / max(1, cnt.get(bidx, 1)) / 3.0
        scores[bidx] = s
    return scores

# --- Attention-head pruning function ---
def apply_attention_head_pruning(
    model: nn.Module,
    processor: AutoProcessor,
    target_attn: List[AttnHandle],
    head_scores: Dict[int, torch.Tensor],
    head_ratio: float,
    device: str = "cuda",
) -> int:
    """Prune `head_ratio` fraction of heads per selected attention block.
    Returns total number of heads pruned across blocks.
    """
    if head_ratio <= 0.0:
        return 0
    # Build DepGraph once
    wrapper = IJepaWrapper(model)
    wrapper.eval().to(device)
    size = getattr(model.config, "image_size", 224)
    dummy = torch.zeros(1, 3, size, size, device=device)
    DG = tp.DependencyGraph().build_dependency(wrapper, example_inputs=(dummy,))

    total_heads = 0
    pruned_heads = 0
    for ah in target_attn:
        sc = head_scores.get(ah.idx)
        if sc is None or sc.numel() == 0:
            continue
        H = ah.num_heads
        total_heads += H
        k_heads = int(round(head_ratio * H))
        if k_heads <= 0:
            continue
        # lowest-score head indices
        head_ids = torch.argsort(sc)[:k_heads].tolist()
        # expand to channel indices for q,k,v out_features
        idxs = []
        for h in head_ids:
            start = h * ah.head_dim
            idxs.extend(list(range(start, start + ah.head_dim)))

        ok = True
        # Query
        try:
            gq = DG.get_pruning_group(ah.query, tp.prune_linear_out_channels, idxs=idxs)
            if DG.check_pruning_group(gq):
                gq.prune()
            else:
                ok = False
        except Exception as e:
            print(f"[warn] Head prune (Q) failed at block {ah.idx}: {e}")
            ok = False
        # Key
        try:
            gk = DG.get_pruning_group(ah.key, tp.prune_linear_out_channels, idxs=idxs)
            if DG.check_pruning_group(gk):
                gk.prune()
            else:
                ok = False
        except Exception as e:
            print(f"[warn] Head prune (K) failed at block {ah.idx}: {e}")
            ok = False
        # Value
        try:
            gv = DG.get_pruning_group(ah.value, tp.prune_linear_out_channels, idxs=idxs)
            if DG.check_pruning_group(gv):
                gv.prune()
            else:
                ok = False
        except Exception as e:
            print(f"[warn] Head prune (V) failed at block {ah.idx}: {e}")
            ok = False
        # Output projection (in-channels)
        try:
            go = DG.get_pruning_group(ah.out, tp.prune_linear_in_channels, idxs=idxs)
            if DG.check_pruning_group(go):
                go.prune()
            else:
                ok = False
        except Exception as e:
            print(f"[warn] Head prune (O) failed at block {ah.idx}: {e}")
            ok = False

        if ok:
            pruned_heads += k_heads
            # Update static attributes so reshape stays valid
            try:
                new_all = ah.query.out_features
                old_dim = ah.head_dim
                new_heads = max(1, new_all // max(1, old_dim))
                if hasattr(ah.attn_mod, "num_attention_heads"):
                    ah.attn_mod.num_attention_heads = new_heads
                if hasattr(ah.attn_mod, "all_head_size"):
                    ah.attn_mod.all_head_size = new_all
                if hasattr(ah.attn_mod, "attention_head_size"):
                    ah.attn_mod.attention_head_size = old_dim
            except Exception as e:
                print(f"[warn] Could not update attention head attrs at block {ah.idx}: {e}")
        else:
            print(f"[warn] Skipped head pruning at block {ah.idx}: at least one sub-group invalid")

    return pruned_heads

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
    eval_seed: int = 123,
) -> Tuple[float, float]:
    """Returns (accuracy, avg_inference_time_ms)"""
    import time
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
    
    # Sort for deterministic ordering across different filesystems
    train_items = sorted(train_items, key=lambda x: str(x[0]))
    val_items = sorted(val_items, key=lambda x: str(x[0]))
    
    # Use dedicated RNG with eval_seed for reproducibility
    eval_rng = random.Random(eval_seed)
    eval_rng.shuffle(train_items)
    eval_rng.shuffle(val_items)
    train_items = train_items[: max(1, min(train_n, len(train_items)))]
    val_items = val_items[: max(1, min(val_n, len(val_items)))]

    inference_times = []
    
    def embed_paths(paths: List[Path], measure_time: bool = False) -> torch.Tensor:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        
        if measure_time:
            # Warmup
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        
        out = model(**inputs).last_hidden_state.mean(dim=1)
        
        if measure_time:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            inference_times.append(elapsed / len(paths))  # per image
        
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

    # Evaluate on val items (measure inference time)
    correct = 0
    total = 0
    for i in range(0, len(val_items), bs):
        batch = val_items[i : i + bs]
        q = embed_paths([p for p, _ in batch], measure_time=True)
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
    
    accuracy = correct / max(1, total)
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    return accuracy, avg_inference_time

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


@torch.no_grad()
def compute_wanda_scores(
    model: nn.Module,
    processor: AutoProcessor,
    dataset_name: str,
    num_images: int,
    target_mlps: List[MLPHandle],
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Wanda pruning scores: |Weight| × √(Input Activation Norm²)
    
    For each MLP.fc1 layer:
    - Capture input activations X during forward pass
    - Compute activation norm: ||X||₂² per input channel
    - Score for output channel i: ||W[i, :]||₂ × √(||X||₂²)
    
    Paper: "A Simple and Effective Pruning Approach for Large Language Models"
    Sun et al. (2023) - https://arxiv.org/abs/2306.11695
    """
    # Storage for activation norms (squared L2 norm per input channel)
    activation_norms: Dict[int, torch.Tensor] = {}  # {block_idx: [in_channels]}
    sample_counts: Dict[int, int] = {}
    
    def make_hook(block_idx: int):
        def hook(_mod, inp, _out):
            # inp is a tuple, first element is the actual input tensor
            x = inp[0]  # [B, tokens, in_channels] or [B, in_channels]
            
            # Flatten batch and tokens dimensions
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])  # [B*T, in_channels]
            
            # Compute squared L2 norm per input channel: sum over all tokens/batch
            # This is: ∑_tokens x[token, channel]²
            channel_norm_sq = torch.sum(x ** 2, dim=0)  # [in_channels]
            
            # Accumulate across batches
            if block_idx not in activation_norms:
                activation_norms[block_idx] = channel_norm_sq.cpu()
                sample_counts[block_idx] = x.shape[0]
            else:
                activation_norms[block_idx] += channel_norm_sq.cpu()
                sample_counts[block_idx] += x.shape[0]
        return hook
    
    # Register hooks on fc1 inputs
    handles = []
    for h in target_mlps:
        handles.append(h.fc1.register_forward_hook(make_hook(h.idx)))
    
    # Load calibration dataset
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
    
    model.eval().to(device)
    
    print(f"Collecting Wanda statistics from {len(paths)} images...")
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
    
    # Compute Wanda scores: |W| × √(activation_norm²)
    scores: Dict[int, torch.Tensor] = {}
    for h in target_mlps:
        block_idx = h.idx
        if block_idx not in activation_norms:
            raise RuntimeError(f"No activations captured for block {block_idx}")
        
        # Weight matrix: [out_channels, in_channels]
        W = h.fc1.weight.detach().cpu()  # [out, in]
        
        # Activation norms: [in_channels]
        act_norm_sq = activation_norms[block_idx]  # Already summed over all tokens
        
        # Average over samples
        act_norm_sq = act_norm_sq / sample_counts[block_idx]
        
        # Wanda score per output channel:
        # score[i] = ||W[i, :]||_2 × sqrt(sum_j ||X[:, j]||²)
        # Simplified: score[i] = sum_j |W[i,j]| × sqrt(act_norm_sq[j])
        
        # Weight magnitude per output channel, per input feature
        W_abs = torch.abs(W)  # [out, in]
        
        # Broadcast activation norms: [1, in] → [out, in]
        act_sqrt = torch.sqrt(act_norm_sq.clamp(min=1e-12))  # [in]
        
        # Wanda metric: element-wise product, then sum over input dimension
        # Score[out_i] = sum_j |W[out_i, in_j]| × sqrt(act_norm[in_j])
        wanda_metric = W_abs * act_sqrt.unsqueeze(0)  # [out, in] × [1, in]
        scores[block_idx] = torch.sum(wanda_metric, dim=1)  # [out]
    
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
    p.add_argument("--prune-ratio", type=float, nargs='+', default=[0.30], help="fraction(s) of MLP hidden channels to remove in selected blocks (can specify multiple: 0.90 0.92 0.95)")
    p.add_argument("--calib-ds", type=str, default="imagenette", help="'imagenette' (auto-download) or a local folder path for datasets.ImageFolder")
    p.add_argument("--calib-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    # NeuronRank hyperparameters (benchmark-style)
    p.add_argument("--nr-activation-threshold", type=float, default=0.05, help="Pre-activation threshold for document frequency (benchmark NR)")
    p.add_argument("--nr-tf-power", type=float, default=1.0, help="Exponent for TF (mean activation) component")
    p.add_argument("--nr-idf-power", type=float, default=1.0, help="Exponent for IDF component")
    p.add_argument("--nr-idf-add", type=float, default=1.0, help="Constant added to IDF before exponentiation")
    p.add_argument("--nr-idf-smooth", type=float, default=1.0, help="Smoothing for IDF numerator/denominator")
    p.add_argument("--nr-weight-power", type=float, default=1.0, help="Exponent for weight magnitude component")
    p.add_argument("--use-benchmark-nr", action="store_true", help="[DEPRECATED] Use benchmark-style NeuronRank (pre-activation TF-IDF)")
    p.add_argument("--use-postact-tfidf", action="store_true", help="Use post-activation TF-IDF NeuronRank (measures output neurons directly, recommended)")
    p.add_argument("--nr-use-per-class", dest="nr_use_per_class", action="store_true", help="Use per-class TF-IDF discrimination when post-activation stats include class labels")
    p.add_argument("--no-nr-use-per-class", dest="nr_use_per_class", action="store_false", help="Disable per-class TF-IDF discrimination (fallback to global stats only)")
    p.set_defaults(nr_use_per_class=True)
    p.add_argument("--nr-per-class-agg", type=str, default="var", choices=["max", "mean", "sum", "var", "std", "median", "max_minus_mean"], help="Aggregation over per-class TF-IDF scores")
    p.add_argument("--nr-per-class-power", type=float, default=1.0, help="Exponent applied to aggregated per-class discrimination score")
    p.add_argument("--nr-discrimination-weight", type=float, default=1.0, help="Weight for discrimination term in hybrid score: Score = (Variance^weight) * Magnitude")
    p.add_argument("--beta", type=float, default=1.0, help="Multiplier for magnitude/impact score: Score = Discrimination × (beta × ||W||₂)")
    
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
    p.add_argument("--compare-wanda", action="store_true", help="Also prune a fresh model with Wanda (weights×activations) channel scores and report k-NN")
    p.add_argument("--eval-train", type=int, default=2000)
    p.add_argument("--eval-val", type=int, default=1000)
    p.add_argument("--save-dir", type=str, default="./ijepa_pruned")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=123, help="Random seed for k-NN evaluation sampling (for reproducibility)")
    p.add_argument("--eval-seeds", type=int, nargs='+', default=None, help="Multiple eval seeds to average over for stable results (e.g., --eval-seeds 123 456 789)")
    # Processor speed control
    p.add_argument("--use-fast-processor", dest="use_fast_processor", action="store_true", help="Use fast image processor if available (default)")
    p.add_argument("--no-fast-processor", dest="use_fast_processor", action="store_false", help="Force slow image processor")
    p.set_defaults(use_fast_processor=True)
    p.add_argument("--mode", type=str, choices=["structured", "unstructured"], default="structured", help="Structured (channel) vs unstructured (element) pruning")
    p.add_argument("--unstructured-scope", type=str, choices=["global", "layer"], default="global", help="Global magnitude or per-layer magnitude pruning")
    p.add_argument("--include-attn", action="store_true", help="Include attention (q/k/v and attn out) in unstructured pruning")
    p.add_argument("--no-mlp", action="store_true", help="Exclude MLP (fc1/fc2) from unstructured pruning")
    p.add_argument("--remove-reparam", action="store_true", help="Call prune.remove() to make zeros permanent on weights")
    # Save control
    p.add_argument("--save-models", action="store_true", help="Save pruned models to disk (disabled by default to save space)")
    p.add_argument("--save-graph", action="store_true", help="Save comparison graph to disk (disabled by default)")

    # --- Add attention-head pruning CLI flags ---
    p.add_argument("--prune-heads", action="store_true", help="Enable attention head pruning before MLP pruning")
    p.add_argument("--head-ratio", type=float, default=0.0, help="Fraction of attention heads to remove in selected blocks (0.0 disables)")
    p.add_argument("--head-score", type=str, default="act_l1", choices=["act_l1", "act_l2"], help="Head scoring: mean |Q,K,V| (L1) or L2 over tokens")

    args = p.parse_args()
    
    # Print configuration at the very start
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Model seed: {args.seed}")
    print(f"Eval seed: {args.eval_seed}")
    if not args.save_models:
        print("⚠️  Model saving DISABLED (use --save-models to enable)")
    if not args.save_graph:
        print("⚠️  Graph saving DISABLED (use --save-graph to enable)")
    print("="*80)
    
    seed_all(args.seed)

    print(f"\nLoading model: {args.model_id}")
    # Prefer fast processor to avoid slow-processor warning, with safe fallback
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, use_fast=args.use_fast_processor)
    except TypeError:
        # Older transformers may not accept use_fast for this processor; fall back silently
        print("[warn] `use_fast=True` failed for this processor, falling back.")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)

    # Discover all MLPs and choose targets
    mlps = find_mlp_modules(model)
    target_indices = sorted(set([i for i in parse_layer_selection(args.layers, total=len(mlps)) if 0 <= i < len(mlps)]))
    target_mlps = [m for m in mlps if m.idx in target_indices]
    print(f"Discovered {len(mlps)} blocks with MLPs; targeting indices: {target_indices}")

    # Optional: attention head pruning BEFORE MLP pruning (now that args/model/targets exist)
    if args.prune_heads and args.head_ratio > 0.0:
        print(f"\n[Head Pruning] Collecting head scores and pruning {args.head_ratio:.2f} of heads per selected block…")
        attn_all = find_attention_modules(model)
        target_attn = [a for a in attn_all if a.idx in target_indices]
        if not target_attn:
            print("[warn] No attention modules found for requested blocks; skipping head pruning.")
        else:
            head_scores = collect_head_activation_scores(
                model=model,
                processor=processor,
                dataset_name=args.calib_ds,
                num_images=args.calib_samples,
                target_attn=target_attn,
                batch_size=args.batch_size,
                device=args.device,
                mode=args.head_score,
            )
            pruned_heads = apply_attention_head_pruning(
                model=model,
                processor=processor,
                target_attn=target_attn,
                head_scores=head_scores,
                head_ratio=args.head_ratio,
                device=args.device,
            )
            print(f"[Head Pruning] Pruned {pruned_heads} heads total across {len(target_attn)} blocks.")

    # Baseline params and size
    base_params = count_params(model)
    base_size_gb = estimate_model_size_gb(model)
    print(f"\n{'='*60}")
    print(f"BASELINE MODEL SIZE")
    print(f"{'='*60}")
    print(f"Total parameters: {base_params/1e6:.2f}M ({base_params:,})")
    print(f"Estimated size: {base_size_gb:.3f} GB")
    print(f"{'='*60}\n")

    # Optional baseline evaluation BEFORE pruning
    acc_base = None
    time_base = None
    if args.eval != "none":
        if args.calib_ds == "imagenette":
            eval_root = auto_download_imagenette()
        else:
            eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
        
        # Use multiple eval seeds if specified for stability
        eval_seeds = args.eval_seeds if args.eval_seeds else [args.eval_seed]
        
        if len(eval_seeds) > 1:
            print(f"Running baseline k-NN (local ImageFolder at {eval_root}) with {len(eval_seeds)} seeds for stability…")
            accs = []
            times = []
            for seed in eval_seeds:
                acc, inf_time = knn_probe_local(model, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=seed)
                accs.append(acc)
                times.append(inf_time)
                print(f"  Seed {seed}: {acc*100:.2f}%")
            acc_base = sum(accs) / len(accs)
            time_base = sum(times) / len(times)
            std_base = (sum((a - acc_base)**2 for a in accs) / len(accs)) ** 0.5
            print(f"Baseline k-NN@20: top-1 = {acc_base*100:.2f}% ± {std_base*100:.2f}%")
            print(f"Baseline inference time: {time_base:.2f} ms/image\n")
        else:
            print(f"Running baseline k-NN (local ImageFolder at {eval_root})…")
            acc_base, time_base = knn_probe_local(model, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=eval_seeds[0])
            print(f"Baseline k-NN@20: top-1 = {acc_base*100:.2f}%")
            print(f"Baseline inference time: {time_base:.2f} ms/image\n")

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
            zeros, tot = apply_unstructured_from_scores(score_map, amount=args.prune_ratio[0], scope=args.unstructured_scope, remove_reparam=args.remove_reparam)
            spars = 100.0 * zeros / max(1, tot)
            print(f"[Unstructured-NRP] Applied {args.prune_ratio[0]:.2f} ({args.unstructured_scope}) → sparsity {spars:.2f}% over fc1 weights")
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    eval_root = auto_download_imagenette()
                else:
                    eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                acc_un_nrp, time_un_nrp = knn_probe_local(model_nrp, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=args.eval_seed)
                print(f"[Unstructured-NRP] k-NN@20: top-1 = {acc_un_nrp*100:.2f}%  |  Δ vs base = {(acc_un_nrp - (acc_base or acc_un_nrp)) * 100:.2f}%  |  Inference: {time_un_nrp:.2f} ms")
            # Save
            if args.save_models:
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
            zeros_mb, tot_mb = apply_unstructured_pruning(mods_mb, amount=args.prune_ratio[0], scope=args.unstructured_scope, remove_reparam=args.remove_reparam)
            spars_mb = 100.0 * zeros_mb / max(1, tot_mb)
            print(f"[Unstructured-MB] Applied {args.prune_ratio[0]:.2f} ({args.unstructured_scope}) → sparsity {spars_mb:.2f}% over fc1 weights")
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    eval_root = auto_download_imagenette()
                else:
                    eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                acc_un_mb, time_un_mb = knn_probe_local(model_mb, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=args.eval_seed)
                print(f"[Unstructured-MB]  k-NN@20: top-1 = {acc_un_mb*100:.2f}%  |  Δ vs base = {(acc_un_mb - (acc_base or acc_un_mb)) * 100:.2f}%  |  Inference: {time_un_mb:.2f} ms")
            # Save
            if args.save_models:
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
    if args.use_postact_tfidf:
        # Post-activation TF-IDF (NEW: measures output neurons directly)
        print("Collecting NeuronRank scores (Variance x Magnitude method)...")
        import hashlib
        cache_key_str = f"postact_var_mag_{args.model_id}_{args.layers}_{args.calib_ds}_{args.calib_samples}"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_dir = os.path.expanduser(args.stats_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"activation_stats_{cache_key}.pt")
        
        activation_stats = None
        if not args.force_recollect_stats and os.path.exists(cache_file):
            try:
                print(f"Loading cached activation statistics from: {cache_file}")
                activation_stats = torch.load(cache_file, map_location='cpu')
                print(f"✓ Loaded cached stats (post-activation, threshold={args.nr_activation_threshold})")
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}), re-collecting...")
                activation_stats = None
        
        if activation_stats is None:
            activation_stats = collect_postactivation_stats(
                model=model,
                processor=processor,
                dataset_name=args.calib_ds,
                num_images=args.calib_samples,
                target_mlps=target_mlps,
                batch_size=args.batch_size,
                device=args.device,
            )
            try:
                torch.save(activation_stats, cache_file)
                print(f"✓ Saved activation statistics to cache: {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to save cache ({e})")
        
        print(f"  Computing scores based on (Variance^{args.nr_discrimination_weight} * {args.beta}×Weight_Magnitude)...")
        scores = compute_variance_magnitude_scores(
            target_mlps=target_mlps,
            activation_stats=activation_stats,
            discrimination_weight=args.nr_discrimination_weight,
            beta=args.beta,
        )
        
        # ALSO compute fc2 variant for comparison
        print(f"  [COMPARISON] Also computing fc2-based scores (using downstream weights, beta={args.beta})...")
        scores_fc2 = compute_variance_magnitude_scores_fc2(
            target_mlps=target_mlps,
            activation_stats=activation_stats,
            discrimination_weight=args.nr_discrimination_weight,
            beta=args.beta,
        )
    elif args.use_benchmark_nr:
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

    # Track results for comparison table
    results_list = []  # (ratio, nrp_acc, mb_acc, wanda_acc, fc2_acc, nrp_target_params, mb_target_params, wanda_target_params, fc2_target_params, nrp_time, mb_time, wanda_time, fc2_time)
    base_target_params = count_target_mlp_params(target_mlps)
    print(f"Target MLP params (before pruning): {base_target_params/1e6:.2f}M")
    
    # Check if we have fc2 scores available
    has_fc2_scores = 'scores_fc2' in locals()

    # Apply structured pruning (NRP) for each prune ratio
    for prune_ratio in args.prune_ratio:
        print(f"\n{'='*80}")
        print(f"Pruning with ratio: {prune_ratio:.2f}")
        print(f"{'='*80}")
        
        # Deep copy model for this ratio
        model_pruned = copy.deepcopy(model)
        mlps_pruned = find_mlp_modules(model_pruned)
        target_mlps_pruned = [m for m in mlps_pruned if m.idx in target_indices]
        
        print(f"Applying structured pruning (NRP): ratio={prune_ratio:.2f} on {len(target_mlps_pruned)} blocks…")
        pruned = apply_structured_mlp_pruning(
            model=model_pruned,
            processor=processor,
            target_mlps=target_mlps_pruned,
            scores=scores,
            prune_ratio=prune_ratio,
            device=args.device,
        )
        new_params = count_params(model_pruned)
        new_size_gb = estimate_model_size_gb(model_pruned)
        params_reduction = (base_params - new_params) / base_params * 100
        size_reduction = (base_size_gb - new_size_gb) / base_size_gb * 100
        
        print(f"Pruned channels: ~{pruned} (aggregated across blocks)")
        print(f"\n{'─'*60}")
        print(f"Model Size After Pruning:")
        print(f"  Params: {new_params/1e6:.2f}M  |  Δ = -{(base_params - new_params)/1e6:.2f}M ({params_reduction:.1f}% reduction)")
        print(f"  Size:   {new_size_gb:.3f} GB  |  Δ = -{(base_size_gb - new_size_gb):.3f} GB ({size_reduction:.1f}% reduction)")
        print(f"{'─'*60}\n")

        # Evaluate NRP
        acc_nrp = None
        time_nrp = None
        nrp_target_params = count_target_mlp_params(target_mlps_pruned)
        if args.eval != "none":
            if args.calib_ds == "imagenette":
                eval_root = auto_download_imagenette()
            else:
                eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
            print(f"Evaluating NRP model with k-NN (local ImageFolder at {eval_root})…")
            acc_nrp, time_nrp = knn_probe_local(model_pruned, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=args.eval_seed)
            speedup = (time_base / time_nrp) if (time_base and time_nrp) else 0
            print(f"NRP k-NN@20: top-1 = {acc_nrp*100:.2f}%  |  Δ vs base = {(acc_nrp - (acc_base or acc_nrp)) * 100:.2f}%  |  Inference: {time_nrp:.2f} ms ({speedup:.2f}x speedup)\n")

        # Optional magnitude-based comparison
        acc_mb = None
        time_mb = None
        mb_target_params = None
        if args.compare_mb:
            print("\n=== Magnitude-based pruning comparison ===")
            model_mb = AutoModel.from_pretrained(args.model_id)
            mlps_mb = find_mlp_modules(model_mb)
            target_mb = [m for m in mlps_mb if m.idx in target_indices]
            mb_scores = compute_magnitude_scores(target_mb)
            print(f"Applying structured pruning (MBP): ratio={prune_ratio:.2f} on {len(target_mb)} blocks…")
            _ = apply_structured_mlp_pruning(
                model=model_mb,
                processor=processor,
                target_mlps=target_mb,
                scores=mb_scores,
                prune_ratio=prune_ratio,
                device=args.device,
            )
            mb_target_params = count_target_mlp_params(target_mb)
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    eval_root = auto_download_imagenette()
                else:
                    eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                acc_mb, time_mb = knn_probe_local(model_mb, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=args.eval_seed)
                speedup_mb = (time_base / time_mb) if (time_base and time_mb) else 0
                print(f"MBP k-NN@20: top-1 = {acc_mb*100:.2f}%  |  Δ vs base = {(acc_mb - (acc_base or acc_mb)) * 100:.2f}%  |  Inference: {time_mb:.2f} ms ({speedup_mb:.2f}x speedup)")
            # Save MBP too
            if args.save_models:
                mb_dir = f"{args.save_dir}_r{int(prune_ratio*100):02d}_mbp"
                os.makedirs(mb_dir, exist_ok=True)
                print(f"Saving magnitude-pruned model to: {mb_dir}")
                model_mb.save_pretrained(mb_dir)

        # Optional Wanda comparison (weights × activations)
        acc_wanda = None
        time_wanda = None
        wanda_target_params = None
        if args.compare_wanda:
            print("\n=== Wanda (weights×activations) pruning comparison ===")
            model_wanda = AutoModel.from_pretrained(args.model_id)
            mlps_wanda = find_mlp_modules(model_wanda)
            target_wanda = [m for m in mlps_wanda if m.idx in target_indices]
            print("Computing Wanda scores (|W| × √activation_norm)…")
            wanda_scores = compute_wanda_scores(
                model=model_wanda,
                processor=processor,
                dataset_name=args.calib_ds,
                num_images=args.calib_samples,
                target_mlps=target_wanda,
                batch_size=args.batch_size,
                device=args.device,
            )
            print(f"Applying structured pruning (Wanda): ratio={prune_ratio:.2f} on {len(target_wanda)} blocks…")
            _ = apply_structured_mlp_pruning(
                model=model_wanda,
                processor=processor,
                target_mlps=target_wanda,
                scores=wanda_scores,
                prune_ratio=prune_ratio,
                device=args.device,
            )
            wanda_target_params = count_target_mlp_params(target_wanda)
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    eval_root = auto_download_imagenette()
                else:
                    eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                acc_wanda, time_wanda = knn_probe_local(model_wanda, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=args.eval_seed)
                speedup_wanda = (time_base / time_wanda) if (time_base and time_wanda) else 0
                print(f"Wanda k-NN@20: top-1 = {acc_wanda*100:.2f}%  |  Δ vs base = {(acc_wanda - (acc_base or acc_wanda)) * 100:.2f}%  |  Inference: {time_wanda:.2f} ms ({speedup_wanda:.2f}x speedup)")
            # Save Wanda too
            if args.save_models:
                wanda_dir = f"{args.save_dir}_r{int(prune_ratio*100):02d}_wanda"
                os.makedirs(wanda_dir, exist_ok=True)
                print(f"Saving Wanda-pruned model to: {wanda_dir}")
                model_wanda.save_pretrained(wanda_dir)

        # Optional fc2-based NeuronRank comparison (uses downstream weights)
        acc_fc2 = None
        time_fc2 = None
        fc2_target_params = None
        if has_fc2_scores:
            print("\n=== NeuronRank-fc2 (Var×fc2_weights) pruning comparison ===")
            model_fc2 = copy.deepcopy(model)
            mlps_fc2 = find_mlp_modules(model_fc2)
            target_fc2 = [m for m in mlps_fc2 if m.idx in target_indices]
            print(f"Applying structured pruning (NR-fc2): ratio={prune_ratio:.2f} on {len(target_fc2)} blocks…")
            _ = apply_structured_mlp_pruning(
                model=model_fc2,
                processor=processor,
                target_mlps=target_fc2,
                scores=scores_fc2,
                prune_ratio=prune_ratio,
                device=args.device,
            )
            fc2_target_params = count_target_mlp_params(target_fc2)
            if args.eval != "none":
                if args.calib_ds == "imagenette":
                    eval_root = auto_download_imagenette()
                else:
                    eval_root, _ = _resolve_imagefolder_root_and_split(args.calib_ds)
                acc_fc2, time_fc2 = knn_probe_local(model_fc2, processor, root=eval_root, train_n=args.eval_train, val_n=args.eval_val, device=args.device, eval_seed=args.eval_seed)
                speedup_fc2 = (time_base / time_fc2) if (time_base and time_fc2) else 0
                print(f"NR-fc2 k-NN@20: top-1 = {acc_fc2*100:.2f}%  |  Δ vs base = {(acc_fc2 - (acc_base or acc_fc2)) * 100:.2f}%  |  Inference: {time_fc2:.2f} ms ({speedup_fc2:.2f}x speedup)")

        # Save NRP model with ratio in directory name
        if args.save_models:
            save_dir = f"{args.save_dir}_r{int(prune_ratio*100):02d}"
            os.makedirs(save_dir, exist_ok=True)
            print(f"\nSaving NRP-pruned model to: {save_dir}")
            model_pruned.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
        
        # Store results for comparison table
        if args.eval != "none":
            results_list.append((
                prune_ratio,
                acc_nrp,
                acc_mb,
                acc_wanda,
                acc_fc2,
                nrp_target_params,
                mb_target_params,
                wanda_target_params,
                fc2_target_params,
                time_nrp,
                time_mb,
                time_wanda,
                time_fc2
            ))

    # Summary
    if args.eval != "none":
        print("\n=== Summary (k-NN@20 top-1) ===")
        if acc_base is not None: print(f"Base    : {acc_base*100:.2f}%")
        if acc_nrp is not None:  print(f"NR-fc1  : {acc_nrp*100:.2f}%  (Δ {((acc_nrp - acc_base) if acc_base is not None else 0)*100:.2f}%)")
        if has_fc2_scores and 'acc_fc2' in locals():
            print(f"NR-fc2  : {acc_fc2*100:.2f}%  (Δ {((acc_fc2 - acc_base) if acc_base is not None else 0)*100:.2f}%)")
        if args.compare_mb and 'acc_mb' in locals():
            print(f"MB      : {acc_mb*100:.2f}%  (Δ {((acc_mb - acc_base) if acc_base is not None else 0)*100:.2f}%)")
        if args.compare_wanda and 'acc_wanda' in locals():
            print(f"Wanda   : {acc_wanda*100:.2f}%  (Δ {((acc_wanda - acc_base) if acc_base is not None else 0)*100:.2f}%)")

    # === Final Comparison Table ===
    if len(results_list) > 0 and args.eval != "none":
        print("\n" + "="*120)
        print("FINAL COMPARISON: NeuronRank vs Baselines (Magnitude + Wanda)")
        print("="*120)
        
        # Table header
        print(f"\n{'Sparsity':<10} {'Method':<8} {'Params (M)':<12} {'Compression':<14} {'k-NN Acc':<12} {'Δ Acc':<12} {'Inference':<14} {'Speedup':<10}")
        print("-" * 140)
        
        # Baseline row
        if acc_base is not None:
            base_time_str = f"{time_base:.2f} ms" if time_base else "-"
            print(f"{'0%':<10} {'Base':<8} {base_target_params/1e6:>10.2f}   {'1.00x':>12}   {acc_base*100:>10.2f}%  {'-':>10}   {base_time_str:>12}   {'1.00x':>8}")
            print("-" * 140)
        
        # Results rows
        for ratio, nrp_acc, mb_acc, wanda_acc, fc2_acc, nrp_params, mb_params, wanda_params, fc2_params, nrp_time, mb_time, wanda_time, fc2_time in results_list:
            sparsity_pct = ratio * 100
            
            # NeuronRank (fc1) row
            if nrp_acc is not None and nrp_params is not None:
                compression_nr = base_target_params / nrp_params if nrp_params > 0 else 0
                delta_acc_nr = (nrp_acc - acc_base) * 100 if acc_base else 0
                speedup_nr = (time_base / nrp_time) if (time_base and nrp_time) else 0
                time_str_nr = f"{nrp_time:.2f} ms" if nrp_time else "-"
                speedup_str_nr = f"{speedup_nr:.2f}x" if speedup_nr else "-"
                print(f"{f'{sparsity_pct:.0f}%':<10} {'NR-fc1':<8} {nrp_params/1e6:>10.2f}   {f'{compression_nr:.2f}x':>12}   {nrp_acc*100:>10.2f}%  {delta_acc_nr:>+9.2f}%   {time_str_nr:>12}   {speedup_str_nr:>8}")
            
            # NeuronRank (fc2) row
            if has_fc2_scores and fc2_acc is not None and fc2_params is not None:
                compression_fc2 = base_target_params / fc2_params if fc2_params > 0 else 0
                delta_acc_fc2 = (fc2_acc - acc_base) * 100 if acc_base else 0
                speedup_fc2 = (time_base / fc2_time) if (time_base and fc2_time) else 0
                time_str_fc2 = f"{fc2_time:.2f} ms" if fc2_time else "-"
                speedup_str_fc2 = f"{speedup_fc2:.2f}x" if speedup_fc2 else "-"
                print(f"{'':<10} {'NR-fc2':<8} {fc2_params/1e6:>10.2f}   {f'{compression_fc2:.2f}x':>12}   {fc2_acc*100:>10.2f}%  {delta_acc_fc2:>+9.2f}%   {time_str_fc2:>12}   {speedup_str_fc2:>8}")
            
            # Magnitude-based row
            if args.compare_mb and mb_acc is not None and mb_params is not None:
                compression_mb = base_target_params / mb_params if mb_params > 0 else 0
                delta_acc_mb = (mb_acc - acc_base) * 100 if acc_base else 0
                speedup_mb = (time_base / mb_time) if (time_base and mb_time) else 0
                time_str_mb = f"{mb_time:.2f} ms" if mb_time else "-"
                speedup_str_mb = f"{speedup_mb:.2f}x" if speedup_mb else "-"
                print(f"{'':<10} {'MB':<8} {mb_params/1e6:>10.2f}   {f'{compression_mb:.2f}x':>12}   {mb_acc*100:>10.2f}%  {delta_acc_mb:>+9.2f}%   {time_str_mb:>12}   {speedup_str_mb:>8}")
            
            # Wanda row
            if args.compare_wanda and wanda_acc is not None and wanda_params is not None:
                compression_wanda = base_target_params / wanda_params if wanda_params > 0 else 0
                delta_acc_wanda = (wanda_acc - acc_base) * 100 if acc_base else 0
                speedup_wanda = (time_base / wanda_time) if (time_base and wanda_time) else 0
                time_str_wanda = f"{wanda_time:.2f} ms" if wanda_time else "-"
                speedup_str_wanda = f"{speedup_wanda:.2f}x" if speedup_wanda else "-"
                print(f"{'':<10} {'Wanda':<8} {wanda_params/1e6:>10.2f}   {f'{compression_wanda:.2f}x':>12}   {wanda_acc*100:>10.2f}%  {delta_acc_wanda:>+9.2f}%   {time_str_wanda:>12}   {speedup_str_wanda:>8}")
            
            print("-" * 140)
        
        # Key insights
        if (args.compare_mb or args.compare_wanda) and len(results_list) > 0:
            print("\nKEY INSIGHTS:")
            # Find best NR performance
            best_nr = max([(r[0], r[1]) for r in results_list if r[1] is not None], key=lambda x: x[1], default=(0, 0))
            print(f"  • Best NR performance: {best_nr[0]*100:.0f}% sparsity → {best_nr[1]*100:.2f}% accuracy")
            
            # Average advantage over MB
            if args.compare_mb:
                advantages = [(r[1] - r[2]) * 100 for r in results_list if r[1] is not None and r[2] is not None]
                if advantages:
                    avg_adv = sum(advantages) / len(advantages)
                    print(f"  • Average NR advantage over MB: {avg_adv:+.2f}%")
            
            # Average advantage over Wanda
            if args.compare_wanda:
                advantages_wanda = [(r[1] - r[3]) * 100 for r in results_list if r[1] is not None and r[3] is not None]
                if advantages_wanda:
                    avg_adv_wanda = sum(advantages_wanda) / len(advantages_wanda)
                    print(f"  • Average NR advantage over Wanda: {avg_adv_wanda:+.2f}%")
            
            # Most aggressive pruning
            if results_list:
                most_aggressive = max(results_list, key=lambda x: x[0])
                if most_aggressive[1] is not None and most_aggressive[4] is not None:
                    comp_rate = base_target_params / most_aggressive[4]
                    retention = (most_aggressive[1] / acc_base * 100) if acc_base else 0
                    # Extract inference times (now at index 7)
                    ma_time = most_aggressive[7] if len(most_aggressive) > 7 else None
                    ma_speedup = (time_base / ma_time) if (time_base and ma_time) else 0
                    print(f"  • Most aggressive: {most_aggressive[0]*100:.0f}% sparsity → {comp_rate:.2f}x compression, {retention:.1f}% accuracy retention, {ma_speedup:.2f}x speedup")
        
        print("="*140)
        
        # === Model Size Summary ===
        print("\n" + "="*80)
        print("MODEL SIZE COMPARISON")
        print("="*80)
        print(f"{'Configuration':<30} {'Total Params':<15} {'Model Size':<15} {'Reduction':<15}")
        print("-" * 80)
        print(f"{'Baseline (unpruned)':<30} {base_params/1e6:>12.2f}M   {base_size_gb:>12.3f} GB   {'-':>12}")
        
        # Calculate sizes for each pruned configuration
        if len(results_list) > 0:
            for ratio, _, _, _, nrp_params, mb_params, wanda_params, _, _, _ in results_list:
                # Rough estimate: assume params scale proportionally to target layer reduction
                target_reduction = (base_target_params - nrp_params) / base_target_params
                estimated_total_params = base_params - (base_target_params - nrp_params)
                estimated_size = base_size_gb * (estimated_total_params / base_params)
                size_reduction = (1 - estimated_size / base_size_gb) * 100
                
                config_name = f"NR @ {ratio*100:.0f}% sparsity"
                print(f"{config_name:<30} {estimated_total_params/1e6:>12.2f}M   {estimated_size:>12.3f} GB   {size_reduction:>11.1f}%")
                
                if mb_params:
                    mb_estimated_total = base_params - (base_target_params - mb_params)
                    mb_estimated_size = base_size_gb * (mb_estimated_total / base_params)
                    mb_size_reduction = (1 - mb_estimated_size / base_size_gb) * 100
                    config_name_mb = f"MB @ {ratio*100:.0f}% sparsity"
                    print(f"{config_name_mb:<30} {mb_estimated_total/1e6:>12.2f}M   {mb_estimated_size:>12.3f} GB   {mb_size_reduction:>11.1f}%")
                
                if wanda_params:
                    wanda_estimated_total = base_params - (base_target_params - wanda_params)
                    wanda_estimated_size = base_size_gb * (wanda_estimated_total / base_params)
                    wanda_size_reduction = (1 - wanda_estimated_size / base_size_gb) * 100
                    config_name_wanda = f"Wanda @ {ratio*100:.0f}% sparsity"
                    print(f"{config_name_wanda:<30} {wanda_estimated_total/1e6:>12.2f}M   {wanda_estimated_size:>12.3f} GB   {wanda_size_reduction:>11.1f}%")
        
        print("="*80)
        print("Note: Total model size includes all layers (pruned MLP + unpruned attention/embeddings)")
        print("="*80 + "\n")
        
        # === Matplotlib Graph ===
        if len(results_list) > 1 and HAS_MATPLOTLIB:
            print("\nGenerating comparison graph...")
            
            # Prepare data
            sparsities = [0] + [r[0] * 100 for r in results_list]
            nr_accs = [acc_base * 100 if acc_base else 0] + [r[1] * 100 if r[1] else 0 for r in results_list]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot NeuronRank
            plt.plot(sparsities, nr_accs, marker='o', linewidth=2.5, markersize=8, 
                    label='NeuronRank', color='#2E86AB', linestyle='-')
            
            # Plot Magnitude-Based if available
            if args.compare_mb:
                mb_accs = [acc_base * 100 if acc_base else 0] + [r[2] * 100 if r[2] else 0 for r in results_list if r[2] is not None]
                if len(mb_accs) == len(sparsities):
                    plt.plot(sparsities, mb_accs, marker='s', linewidth=2.5, markersize=8,
                            label='Magnitude-Based', color='#A23B72', linestyle='--')
            
            # Plot Wanda if available
            if args.compare_wanda:
                wanda_accs = [acc_base * 100 if acc_base else 0] + [r[3] * 100 if r[3] else 0 for r in results_list if r[3] is not None]
                if len(wanda_accs) == len(sparsities):
                    plt.plot(sparsities, wanda_accs, marker='^', linewidth=2.5, markersize=8,
                            label='Wanda', color='#F18F01', linestyle='-.')
            
            # Baseline horizontal line
            if acc_base is not None:
                plt.axhline(y=acc_base * 100, color='#06A77D', linestyle=':', linewidth=2, 
                           label='Baseline', alpha=0.7)
            
            # Formatting
            plt.xlabel('Sparsity (%)', fontsize=12, fontweight='bold')
            plt.ylabel('k-NN Accuracy (%)', fontsize=12, fontweight='bold')
            plt.title('Pruning Performance: NeuronRank vs Baselines', fontsize=14, fontweight='bold', pad=20)
            plt.legend(loc='best', fontsize=11, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Set reasonable axis limits
            plt.xlim(-5, 105)
            all_accs = nr_accs.copy()
            if args.compare_mb and 'mb_accs' in locals():
                all_accs.extend(mb_accs)
            if args.compare_wanda and 'wanda_accs' in locals():
                all_accs.extend(wanda_accs)
            acc_min = min(all_accs) - 2
            acc_max = max(all_accs) + 2
            plt.ylim(acc_min, acc_max)
            
            # Add annotations for key points
            # Best NR point
            best_idx = nr_accs.index(max(nr_accs[1:], default=nr_accs[0]))
            if best_idx > 0:
                plt.annotate(f'{nr_accs[best_idx]:.1f}%', 
                           xy=(sparsities[best_idx], nr_accs[best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           fontsize=9)
            
            plt.tight_layout()
            
            # Save graph
            if args.save_graph:
                graph_path = os.path.join(args.save_dir, "pruning_comparison.png")
                os.makedirs(os.path.dirname(graph_path) if os.path.dirname(graph_path) else '.', exist_ok=True)
                plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved comparison graph: {graph_path}")
                
                # Also save as PDF for papers
                pdf_path = graph_path.replace('.png', '.pdf')
                plt.savefig(pdf_path, bbox_inches='tight')
                print(f"✓ Saved comparison graph (PDF): {pdf_path}")
            
            plt.close()

    print("\nDone.")


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