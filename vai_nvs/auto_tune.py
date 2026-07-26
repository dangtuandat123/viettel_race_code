"""Per-scene hyperparameter auto-tuning via Optuna (Bayesian Optimization).

Runs train_gs.py fold-0 (pseudo-validation) with different hyperparameter
combinations and picks the best one per scene.  Much more efficient than
grid search or RL — typically converges in 10-20 trials.

Usage:
  # Quick mode (20k steps, 10 trials, ~4h for 2 scenes):
  python -m vai_nvs.auto_tune --data /workspace/VAI_NVS_DATA_ROUND2 \
      --work /workspace/work --n-trials 10 --tune-steps 20000

  # After auto-tune, run the best config for full training:
  python -m vai_nvs.auto_tune --data /workspace/VAI_NVS_DATA_ROUND2 \
      --work /workspace/work --apply-best --full-steps 40000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="root dir containing scene subdirs")
    ap.add_argument("--work", required=True, help="workspace root")
    ap.add_argument("--scenes", default=None, help="comma-separated scene names (default: all)")
    ap.add_argument("--n-trials", type=int, default=10, help="Optuna trials per scene")
    ap.add_argument("--tune-steps", type=int, default=20000,
                    help="max_steps for tuning runs (shorter = faster)")
    ap.add_argument("--full-steps", type=int, default=40000,
                    help="max_steps for final production run")
    ap.add_argument("--apply-best", action="store_true",
                    help="skip tuning, read best params and run full training")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def discover_scenes(data_root: str) -> list[str]:
    """Find scene subdirectory names."""
    root = Path(data_root)
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / "train").exists())


def build_train_cmd(work: str, scene: str, fold: int, params: dict) -> list[str]:
    """Build the command line for train_gs.py."""
    cmd = [
        sys.executable, "-m", "vai_nvs.train_gs",
        "--work", work,
        "--scene", scene,
        "--fold", str(fold),
    ]
    for k, v in params.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def run_trial_training(work: str, scene: str, params: dict) -> float:
    """Run one fold-0 training and return the best score_lb."""
    run_name = f"optuna_trial_{int(time.time())}"
    trial_params = dict(params)
    trial_params["run_name"] = run_name
    trial_params["sanity_views"] = 0  # skip sanity to save time

    cmd = build_train_cmd(work, scene, fold=0, params=trial_params)

    print(f"\n{'='*60}")
    print(f"  Trial: {run_name}")
    print(f"  Params: {json.dumps({k: v for k, v in params.items()}, indent=2)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return -1.0

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.0f}s")

    # Parse best score from eval.jsonl
    eval_path = Path(work) / scene / "runs" / run_name / "eval.jsonl"
    if not eval_path.exists():
        print(f"  No eval.jsonl found at {eval_path}")
        return -1.0

    best_score = -1.0
    with open(eval_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if "score_lb" in row:
                best_score = max(best_score, row["score_lb"])

    print(f"  Best score_lb = {best_score:.5f}")
    return best_score


def define_search_space(trial) -> dict:
    """Define the hyperparameter search space for Optuna.

    Tuned for 12 GB VRAM (RTX 3060):
      - cap_max  ≤ 1.5M  (prevents OOM)
      - sh_degree ≤ 2     (saves ~1 GB per degree)
    """
    return {
        # High-impact parameters
        "strategy": "mcmc",
        "cap_max": trial.suggest_int("cap_max", 500_000, 1_500_000, step=250_000),
        "sh_degree": trial.suggest_int("sh_degree", 1, 2),
        "ssim_lambda": trial.suggest_float("ssim_lambda", 0.1, 0.5, step=0.05),
        "lpips_weight": trial.suggest_float("lpips_weight", 0.0, 0.2, step=0.025),
        "lpips_start_frac": trial.suggest_float("lpips_start_frac", 0.2, 0.5, step=0.05),
        # Medium-impact parameters
        "init_opacity": trial.suggest_float("init_opacity", 0.05, 0.3, step=0.05),
        "eval_every": 5000,
        "eval_supersample": 2.0,
        # Fixed
        "charb_eps": 1e-3,
        "seed": 42,
    }


def tune_scene(work: str, scene: str, n_trials: int, tune_steps: int):
    """Run Optuna optimization for one scene."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = Path(work) / scene / "optuna_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=f"tune_{scene}",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial):
        params = define_search_space(trial)
        params["max_steps"] = tune_steps
        return run_trial_training(work, scene, params)

    remaining = max(0, n_trials - len(study.trials))
    if remaining > 0:
        print(f"\n[{scene}] Starting Optuna: {remaining} trials "
              f"(tune_steps={tune_steps})")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"\n[{scene}] Already have {len(study.trials)} trials, skipping.")

    # Report
    print(f"\n{'='*60}")
    print(f"  [{scene}] Optuna Results ({len(study.trials)} trials)")
    print(f"  Best score: {study.best_value:.5f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")

    # Save best params
    best_path = Path(work) / scene / "best_params.json"
    with open(best_path, "w") as f:
        json.dump({
            "score": study.best_value,
            "params": study.best_params,
            "n_trials": len(study.trials),
            "tune_steps": tune_steps,
        }, f, indent=2)
    print(f"  Saved to: {best_path}")
    return study.best_params, study.best_value


def apply_best_and_train(work: str, scene: str, data: str, full_steps: int):
    """Load best params and run full training (fold -1) + render test."""
    best_path = Path(work) / scene / "best_params.json"
    if not best_path.exists():
        print(f"[{scene}] No best_params.json found — run tuning first!")
        return

    with open(best_path) as f:
        info = json.load(f)

    params = dict(info["params"])
    print(f"\n[{scene}] Applying best params (score={info['score']:.5f} "
          f"from {info['n_trials']} trials @ {info['tune_steps']} steps):")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Common overrides for production runs
    params["max_steps"] = full_steps
    params["strategy"] = "mcmc"
    params["eval_every"] = 3000
    params["eval_supersample"] = 2.0
    params["sanity_views"] = 4

    # Fold 0
    params["run_name"] = f"best_f0_s{full_steps // 1000}k"
    print(f"\n[{scene}] Training fold 0 ({full_steps} steps)...")
    subprocess.run(build_train_cmd(work, scene, 0, params), check=True)

    # Fold -1
    params["run_name"] = f"best_f-1_s{full_steps // 1000}k"
    print(f"\n[{scene}] Training fold -1 ({full_steps} steps)...")
    subprocess.run(build_train_cmd(work, scene, -1, params), check=True)

    # Render test
    print(f"\n[{scene}] Rendering test images...")
    subprocess.run([
        sys.executable, "-m", "vai_nvs.render_test",
        "--work", work, "--scene", scene,
        "--run", f"best_f-1_s{full_steps // 1000}k",
    ], check=True)
    print(f"[{scene}] Done!")


def main():
    args = parse_args()
    data, work = args.data, args.work

    scenes = ([s.strip() for s in args.scenes.split(",")]
              if args.scenes else discover_scenes(data))
    if not scenes:
        print("No scenes found!")
        return

    print(f"Scenes: {scenes}")
    print(f"Mode: {'apply-best' if args.apply_best else 'tune'}")

    # Ensure all scenes are prepared
    for scene in scenes:
        if not (Path(work) / scene / "meta.json").exists():
            print(f"\n[{scene}] Preparing...")
            subprocess.run([
                sys.executable, "-m", "vai_nvs.prepare",
                "--data", f"{data}/{scene}", "--work", work,
            ], check=True)

    if args.apply_best:
        for scene in scenes:
            apply_best_and_train(work, scene, data, args.full_steps)
    else:
        results = {}
        for scene in scenes:
            best_params, best_score = tune_scene(
                work, scene, args.n_trials, args.tune_steps)
            results[scene] = {"score": best_score, "params": best_params}

        print(f"\n{'='*60}")
        print("  TUNING SUMMARY")
        print(f"{'='*60}")
        for scene, r in results.items():
            print(f"  {scene}: score={r['score']:.5f}")
            for k, v in r['params'].items():
                print(f"    {k}: {v}")
        print(f"\nTo apply best params and run full training:")
        print(f"  python -m vai_nvs.auto_tune --data {data} --work {work} "
              f"--apply-best --full-steps {args.full_steps}")


if __name__ == "__main__":
    main()
