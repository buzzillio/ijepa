#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for IJepa NeuronRank pruning.

Searches over NeuronRank hyperparameters to maximize post-pruning k-NN accuracy.
"""

import re
import subprocess
import sys
import shlex
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Regex to extract NRP accuracy from script output
NRP_REGEX = re.compile(r"NRP\s*:\s*([0-9]+(?:\.[0-9]+)?)%")

def run_pruning_script(args_list: list) -> float:
    """Run ijepa_prune_neuronrank.py and extract NRP k-NN accuracy.
    
    Returns: NRP top-1 accuracy as a percentage (0-100).
    """
    import os
    # Use script in same directory, and system Python (assumes running in venv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "ijepa_prune_neuronrank.py")
    
    # Use current Python interpreter (works if you're already in a venv)
    cmd = [sys.executable, script_path] + args_list
    
    # Fix MKL threading issue
    env = os.environ.copy()
    env['MKL_THREADING_LAYER'] = 'GNU'
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per trial
            env=env,
        )
        output = proc.stdout + "\n" + proc.stderr
        
        # Extract and print device info from output
        import re
        device_match = re.search(r"Device:\s*(\w+)", output)
        if device_match:
            print(f"    → Using device: {device_match.group(1).upper()}")
        
        # Check for errors first
        if proc.returncode != 0:
            print(f"=== TRIAL FAILED: Script exited with code {proc.returncode} ===")
            print("STDOUT:")
            print(output[-3000:])  # Print last 3000 chars for debugging
            raise RuntimeError(f"Script failed with exit code {proc.returncode}")
        
        # Extract NRP accuracy
        match = NRP_REGEX.search(output)
        if not match:
            print("=== TRIAL FAILED: No NRP metric found ===")
            print("FULL OUTPUT:")
            print(output[-3000:])  # Print last 3000 chars for debugging
            raise RuntimeError("NRP metric not found in output")
        
        accuracy = float(match.group(1))
        return accuracy
        
    except subprocess.TimeoutExpired:
        print("=== TRIAL TIMED OUT ===")
        raise RuntimeError("Trial exceeded timeout")
    except RuntimeError:
        # Re-raise RuntimeError (from our checks above)
        raise
    except Exception as e:
        print(f"=== TRIAL ERROR: {e} ===")
        raise


def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function: maximize NRP k-NN accuracy."""
    
    # Hyperparameter search space (using integer ticks for discrete steps)
    
    # --nr-activation-threshold: {0.01, 0.11, 0.21, ..., 0.81, 0.91} (9 values)
    thr_tick = trial.suggest_int("thr_tick", 0, 9)
    nr_activation_threshold = round(0.01 + 0.10 * thr_tick, 2)
    
    # --nr-tf-power: {0.0, 0.5, 1.0, 1.5, 2.0} (5 values)
    tf_tick = trial.suggest_int("tf_tick", 0, 4)
    nr_tf_power = 0.5 * tf_tick
    
    # --nr-idf-power: {0.0, 0.5, 1.0, 1.5, 2.0} (5 values)
    idf_tick = trial.suggest_int("idf_tick", 0, 4)
    nr_idf_power = 0.5 * idf_tick
    
    # --nr-weight-power: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0} (7 values)
    weight_tick = trial.suggest_int("weight_tick", 0, 6)
    nr_weight_power = 0.5 * weight_tick
    
    # --nr-idf-smooth: {0, 5, 10, 15, 20} (5 values)
    nr_idf_smooth = trial.suggest_int("idf_smooth", 0, 20, step=5)
    
    # Build command arguments (adapt to your setup)
    args = [
        "--use-benchmark-nr",
        "--model-id", "facebook/ijepa_vith14_1k",
        "--layers", "last3",
        "--prune-ratio", "0.95",
        "--calib-ds", "imagenette",  # Auto-downloads to ~/Datasets if not present
        "--calib-samples", "1000",
        "--batch-size", "16",
        # NeuronRank hyperparameters
        "--nr-activation-threshold", str(nr_activation_threshold),
        "--nr-tf-power", str(nr_tf_power),
        "--nr-idf-power", str(nr_idf_power),
        "--nr-weight-power", str(nr_weight_power),
        "--nr-idf-smooth", str(nr_idf_smooth),
        # Evaluation
        "--eval", "knn",
        "--eval-train", "1000",
        "--eval-val", "1000",
        "--eval-seed", "123",  # Fixed seed for reproducible k-NN sampling
        # Don't save models during HPO to save disk space
        "--save-dir", f"/tmp/ijepa_trial_{trial.number}",
    ]
    
    # Run and get accuracy
    accuracy = run_pruning_script(args)
    
    # Store hyperparameters for later analysis
    trial.set_user_attr("nr_activation_threshold", nr_activation_threshold)
    trial.set_user_attr("nr_tf_power", nr_tf_power)
    trial.set_user_attr("nr_idf_power", nr_idf_power)
    trial.set_user_attr("nr_weight_power", nr_weight_power)
    trial.set_user_attr("nr_idf_smooth", nr_idf_smooth)
    trial.set_user_attr("nrp_accuracy", accuracy)
    
    print(f"\n{'='*80}")
    print(f"Trial {trial.number} completed:")
    print(f"  Eval seed: 123 (fixed for reproducibility)")
    print(f"  Threshold: {nr_activation_threshold}")
    print(f"  TF power: {nr_tf_power}, IDF power: {nr_idf_power}")
    print(f"  Weight power: {nr_weight_power}, IDF smooth: {nr_idf_smooth}")
    print(f"  → NRP Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")
    
    return accuracy


def main():
    # Print device info at startup
    import torch
    if torch.cuda.is_available():
        detected_device = "CUDA"
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        detected_device = "MPS"
        print(f"\n🚀 MPS (Apple Silicon GPU) detected")
    else:
        detected_device = "CPU"
        print(f"\n⚠️  No GPU detected, using CPU")
    print(f"Default device for trials: {detected_device}\n")
    
    # Configure Optuna
    sampler = TPESampler(
        seed=42,
        multivariate=True,
        group=True,
        n_startup_trials=10,  # Random search for first 10 trials
        n_ei_candidates=24,
    )
    
    pruner = MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=0,
    )
    
    study = optuna.create_study(
        study_name="ijepa_neuronrank_hpo",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    print("Starting Optuna hyperparameter optimization...")
    print(f"Search space:")
    print(f"  --nr-activation-threshold: [0.01, 0.91] step 0.10 (9 values)")
    print(f"  --nr-tf-power: [0.0, 2.0] step 0.5 (5 values)")
    print(f"  --nr-idf-power: [0.0, 2.0] step 0.5 (5 values)")
    print(f"  --nr-weight-power: [0.0, 3.0] step 0.5 (7 values)")
    print(f"  --nr-idf-smooth: [0, 20] step 5 (5 values)")
    print(f"  Total combinations: 9 × 5 × 5 × 7 × 5 = 7,875")
    print(f"\nRunning 40 trials with TPE sampler...\n")
    
    try:
        study.optimize(objective, n_trials=40, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best NRP accuracy: {study.best_value:.2f}%")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest configuration (user attributes):")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")
    
    # Print top 5 trials
    print(f"\n\nTop 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    for i, trial in enumerate(trials_sorted[:5], 1):
        if trial.value is None:
            continue
        print(f"\n{i}. Trial #{trial.number}: {trial.value:.2f}%")
        print(f"   Threshold={trial.user_attrs.get('nr_activation_threshold', 'N/A')}, "
              f"TF={trial.user_attrs.get('nr_tf_power', 'N/A')}, "
              f"IDF={trial.user_attrs.get('nr_idf_power', 'N/A')}, "
              f"Weight={trial.user_attrs.get('nr_weight_power', 'N/A')}, "
              f"Smooth={trial.user_attrs.get('nr_idf_smooth', 'N/A')}")
    
    # Save study
    import pickle
    with open("optuna_ijepa_nrp_study.pkl", "wb") as f:
        pickle.dump(study, f)
    print(f"\n✓ Study saved to: optuna_ijepa_nrp_study.pkl")
    
    # Generate command with best params
    best_attrs = study.best_trial.user_attrs
    print(f"\n\n{'='*80}")
    print("BEST COMMAND TO REPRODUCE:")
    print("="*80)
    print(f"""
python ijepa_prune_neuronrank.py \\
  --use-benchmark-nr \\
  --model-id facebook/ijepa_vith14_1k \\
  --layers last3 \\
  --prune-ratio 0.95 \\
  --calib-ds imagenette \\
  --calib-samples 1000 \\
  --batch-size 16 \\
  --nr-activation-threshold {best_attrs.get('nr_activation_threshold', 0.05)} \\
  --nr-tf-power {best_attrs.get('nr_tf_power', 1.0)} \\
  --nr-idf-power {best_attrs.get('nr_idf_power', 1.0)} \\
  --nr-weight-power {best_attrs.get('nr_weight_power', 1.0)} \\
  --nr-idf-smooth {best_attrs.get('nr_idf_smooth', 1.0)} \\
  --eval knn --eval-train 1000 --eval-val 1000 \\
  --eval-seed 123 \\
  --compare-mb \\
  --save-dir ./ijepa_best_nr
""")


if __name__ == "__main__":
    main()

