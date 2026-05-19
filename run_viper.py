"""
run_viper.py

Entry point for training a VIPER decision-tree policy by distilling a saved
Tensorforce PPO oracle.

Usage
-----
    python run_viper.py --agent-dir "agents/ppo1 - 66 states throughput"

    # With tree-size constraints and criticality weights:
    python run_viper.py \\
        --agent-dir "agents/ppo1 - 66 states throughput" \\
        --n-iter 80 \\
        --max-depth 10 \\
        --max-leaves 64 \\
        --n-criticality-samples 20 \\
        --save-path "agents/viper_tree.joblib"

Run ``python run_viper.py --help`` for the full option list.
"""

import os

# Deterministic ops (mirrors run.py / test.py)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random
import argparse
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from decision_tree.viper import train_viper, TreePolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "VIPER: distil a trained Tensorforce PPO agent into an interpretable "
            "decision-tree policy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Oracle ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--agent-dir",
        type=str,
        default=os.path.join("agents", "ppo1 - 66 states throughput"),
        help="Path to the saved Tensorforce agent directory.",
    )

    # ── VIPER loop ───────────────────────────────────────────────────────────
    p.add_argument(
        "--n-iter",
        type=int,
        default=80,
        help="Number of VIPER iterations.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Timesteps per episode (must match the oracle's training horizon).",
    )
    p.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Episodes used to evaluate each candidate tree per iteration.",
    )

    # ── Tree hyperparameters ─────────────────────────────────────────────────
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth.  Omit for unlimited.",
    )
    p.add_argument(
        "--max-leaves",
        type=int,
        default=None,
        help="Maximum leaf nodes.  Omit for unlimited.",
    )
    p.add_argument(
        "--criterion",
        type=str,
        default="entropy",
        choices=["entropy", "gini"],
        help="Splitting criterion for the decision tree.",
    )
    p.add_argument(
        "--ccp-alpha",
        type=float,
        default=0.0001,
        help="Cost-complexity pruning alpha.",
    )

    # ── Criticality weights ──────────────────────────────────────────────────
    p.add_argument(
        "--n-criticality-samples",
        type=int,
        default=0,
        help=(
            "Number of noisy oracle queries per state for criticality estimation. "
            "0 = uniform weights (DAgger-style, fastest). "
            ">0 = perturb observation N times and measure action variance."
        ),
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help=(
            "Std-dev of Gaussian noise added to observations when "
            "--n-criticality-samples > 0."
        ),
    )

    # ── Output ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--save-path",
        type=str,
        default=os.path.join("agents", "viper_tree.joblib"),
        help="Joblib path to save the best tree policy.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration output.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    best_tree: TreePolicy = train_viper(
        agent_dir=args.agent_dir,
        n_iter=args.n_iter,
        max_depth=args.max_depth,
        max_leaves=args.max_leaves,
        n_eval_episodes=args.n_eval_episodes,
        timesteps_per_episode=args.timesteps,
        save_path=args.save_path,
        n_criticality_samples=args.n_criticality_samples,
        noise_std=args.noise_std,
        criterion=args.criterion,
        ccp_alpha=args.ccp_alpha,
        verbose=not args.quiet,
    )

    print("\nFinal best tree policy:")
    best_tree.print_info()


if __name__ == "__main__":
    main()
