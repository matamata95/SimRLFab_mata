import os

# Set TensorFlow to use deterministic operations for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import argparse
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from logger import export_statistics_logging
from tensorforce.environments import Environment
from decision_tree.viper import TreePolicy, _extract_obs_and_mask
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate a VIPER-extracted decision tree policy on the "
                "ProductionEnv simulation."
)
parser.add_argument(
    '--tree-path',
    type=str,
    default=os.path.join('agents', 'viper_tree.joblib'),
    help="Path to the saved TreePolicy joblib file "
         "(default: agents/viper_tree.joblib).",
)
parser.add_argument(
    '--episodes',
    type=int,
    default=10 ** 3,
    help="Number of evaluation episodes to run (default: 1000).",
)
parser.add_argument(
    '--timesteps',
    type=int,
    default=10 ** 2,
    help="Max timesteps per episode (default: 100).",
)
args = parser.parse_args()

TREE_PATH = args.tree_path
EPISODES = args.episodes
TIMESTEPS = args.timesteps

# ---------------------------------------------------------------------------
# Load tree policy
# ---------------------------------------------------------------------------

print(f"Loading tree policy from: {TREE_PATH}")
tree_policy = TreePolicy.load(TREE_PATH)
tree_policy.print_info()

# ---------------------------------------------------------------------------
# Build environment
# ---------------------------------------------------------------------------

environment_production = Environment.create(
    environment='production.envs.ProductionEnv',
    max_episode_timesteps=TIMESTEPS,
)

inner_env = environment_production.environment
num_actions = inner_env.actions()['num_values']

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

episode_rewards = []

print(f"\nRunning {EPISODES} evaluation episodes …")
for ep in range(EPISODES):
    state_dict = environment_production.reset()
    total_reward = 0.0

    for _ in range(TIMESTEPS):
        obs, mask = _extract_obs_and_mask(state_dict)
        if mask is None:
            mask = [bool(v > 0.5) for v in obs[:num_actions]]

        action = tree_policy.predict(obs, mask)
        state_dict, terminal, reward = environment_production.execute(actions=action)
        total_reward += reward

        if terminal:
            break

    episode_rewards.append(total_reward)

    if (ep + 1) % 100 == 0:
        recent = episode_rewards[-100:]
        print(
            f"  Episode {ep + 1:5d}/{EPISODES}  |  "
            f"last-100 mean reward: {np.mean(recent):+.4f} "
            f"± {np.std(recent):.4f}"
        )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 50)
print("Evaluation complete")
print(f"  Episodes          : {EPISODES}")
print(f"  Mean reward       : {np.mean(episode_rewards):+.4f}")
print(f"  Std  reward       : {np.std(episode_rewards):.4f}")
print(f"  Min / Max reward  : {np.min(episode_rewards):+.4f} / {np.max(episode_rewards):+.4f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# Export logs (mirrors test.py)
# ---------------------------------------------------------------------------

inner_env.statistics.update({'time_end': inner_env.env.now})
export_statistics_logging(
    statistics=inner_env.statistics,
    parameters=inner_env.parameters,
    resources=inner_env.resources,
)
