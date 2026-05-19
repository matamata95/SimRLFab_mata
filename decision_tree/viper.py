"""
decision_tree/viper.py

VIPER (Verifiable Reinforcement Learning via Policy Extraction) adapted for
SimRLFab_mata.

Reference
---------
Bastani et al. (2019) "Verifiable Reinforcement Learning via Policy Extraction"
https://arxiv.org/abs/1805.08328

Algorithm overview
------------------
VIPER is an iterative imitation-learning algorithm (a variant of DAgger):

  Iteration 0  : roll out the oracle (trained Tensorforce PPO) to collect
                 (state, oracle_action, criticality_weight) triples.
  Iteration i>0: roll out the *current* tree policy to visit its own state
                 distribution; for every state visited, query the oracle for
                 the correct action label and a criticality weight.
  After each iteration the accumulated dataset is used to fit a
  DecisionTreeClassifier (with sample_weight = criticality weights).
  The best-scoring tree across all iterations is returned/saved.

Key adaptations from the reference implementation
--------------------------------------------------
* Oracle framework  : Tensorforce PPO instead of Stable-Baselines3.
* Environment       : custom SimPy-based ProductionEnv instead of Gym envs.
* Criticality weight: two modes (see ``n_criticality_samples`` in train_viper):
    n_criticality_samples == 0  → uniform weights (DAgger-style, fast).
    n_criticality_samples  > 0  → noise-sensitivity proxy: the observation is
        perturbed N times with small Gaussian noise and the oracle is queried
        deterministically each time.  States near a decision boundary will
        flip the oracle's action under perturbation → higher weight.
        This avoids accessing Tensorforce's internal probability distributions
        while still capturing state criticality.
"""

import os
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from tensorforce.agents import Agent
from tensorforce.environments import Environment


# ---------------------------------------------------------------------------
# Tree policy wrapper
# ---------------------------------------------------------------------------

class TreePolicy:
    """
    Wraps a scikit-learn DecisionTreeClassifier for use in the VIPER loop.

    Provides:
    - ``predict(observation, action_mask)`` with automatic fallback to the
      next-best valid action when the tree's top prediction is masked out.
    - Joblib-based serialisation.
    """

    def __init__(self, clf: DecisionTreeClassifier):
        self.clf = clf

    def predict(self, observation: np.ndarray, action_mask: list) -> int:
        """
        Return the tree's action for *observation*, respecting *action_mask*.

        If the tree's highest-probability action is currently invalid, the
        method walks down the tree's leaf-node class probabilities and returns
        the first valid action.  As a last resort (all probabilities equal)
        it returns the first action flagged as valid in *action_mask*.

        Parameters
        ----------
        observation : np.ndarray
            Flat state-observation vector (shape ``(n_features,)``).
        action_mask : list of bool
            Boolean list of length ``n_actions``.  ``True`` = action is valid.

        Returns
        -------
        int
            A valid action index.
        """
        obs_2d = observation.reshape(1, -1)
        pred = int(self.clf.predict(obs_2d)[0])

        if pred < len(action_mask) and action_mask[pred]:
            return pred

        # Fall back to highest-probability valid action
        probs = self.clf.predict_proba(obs_2d)[0]
        for a in np.argsort(probs)[::-1]:
            a = int(a)
            if a < len(action_mask) and action_mask[a]:
                return a

        # Final fallback: first valid action
        for a, valid in enumerate(action_mask):
            if valid:
                return a

        raise ValueError("action_mask contains no valid action (all False).")

    def save(self, path: str) -> None:
        """Serialise the underlying classifier to *path* via joblib."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        joblib.dump(self.clf, path)
        print(f"  Tree policy saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TreePolicy":
        """Load a serialised TreePolicy from *path*."""
        return cls(joblib.load(path))

    def print_info(self) -> None:
        """Print depth and leaf count of the underlying tree."""
        print(f"  max_depth : {self.clf.get_depth()}")
        print(f"  n_leaves  : {self.clf.get_n_leaves()}")


# ---------------------------------------------------------------------------
# Oracle loading
# ---------------------------------------------------------------------------

def load_oracle(agent_dir: str, environment) -> Agent:
    """
    Load a saved Tensorforce agent to serve as the VIPER oracle.

    Parameters
    ----------
    agent_dir : str
        Directory of the saved agent (e.g. ``'agents/ppo1 - 66 states throughput'``).
    environment : TensorforceEnvironment
        The environment instance the agent was trained on (required for
        Tensorforce to reconstruct the agent spec).

    Returns
    -------
    tensorforce.agents.Agent
    """
    oracle = Agent.load(
        directory=agent_dir,
        format='tensorflow',
        environment=environment,
    )
    return oracle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_obs_and_mask(state_dict):
    """
    Pull the observation array and action mask out of a Tensorforce state dict.

    Tensorforce environments return state as either:
    - a dict ``{'observation': np.ndarray, 'action_mask': list}``  (typical for
      ProductionEnv which registers a single 'observation' component), or
    - a bare numpy array (single-component environments without masking).

    Returns
    -------
    obs : np.ndarray, dtype float32
    mask : list of bool or None
    """
    if isinstance(state_dict, dict):
        obs = np.array(state_dict.get('observation', list(state_dict.values())[0]),
                       dtype=np.float32)
        mask = state_dict.get('action_mask', None)
    else:
        obs = np.array(state_dict, dtype=np.float32)
        mask = None
    return obs, mask


# ---------------------------------------------------------------------------
# Criticality weight estimation
# ---------------------------------------------------------------------------

def estimate_criticality(
    oracle: Agent,
    state_dict,
    num_actions: int,
    n_samples: int,
    noise_std: float = 0.02,
) -> float:
    """
    Estimate VIPER's criticality weight ``l(s)`` for a single state.

    Mode A – uniform weights (n_samples == 0)
        Returns 1.0 immediately.  Equivalent to DAgger without state weighting.
        Fastest; use when wall-clock time matters more than sample efficiency.

    Mode B – noise-sensitivity proxy (n_samples > 0)
        Adds small Gaussian noise to the *non-mask* portion of the observation
        vector and queries the oracle deterministically *n_samples* times.
        The weight is the fraction of *unique* actions seen:

            weight = |{oracle(s + ε_i)}| / num_actions

        States near a decision boundary will change action under perturbation
        → fraction close to 1 → high weight.
        States far from any boundary return the same action every time
        → fraction close to 1/num_actions → low weight.

    Note: Only the feature portion of the observation (indices num_actions …
    end) is perturbed.  The first ``num_actions`` elements are the binary
    action-validity flags and must remain intact for the oracle's masking.

    Parameters
    ----------
    oracle : Agent
    state_dict : dict or np.ndarray
    num_actions : int
    n_samples : int
    noise_std : float

    Returns
    -------
    float  in (0, 1]
    """
    if n_samples == 0:
        return 1.0

    obs, mask = _extract_obs_and_mask(state_dict)
    feature_start = num_actions   # skip the binary action-mask prefix

    seen_actions = set()
    for _ in range(n_samples):
        noisy_obs = obs.copy()
        if feature_start < len(noisy_obs):
            noisy_obs[feature_start:] += np.random.normal(
                0.0, noise_std, size=len(noisy_obs) - feature_start
            ).astype(np.float32)

        if mask is not None:
            noisy_state = {'observation': noisy_obs, 'action_mask': mask}
        else:
            noisy_state = noisy_obs

        a = int(oracle.act(states=noisy_state, independent=True))
        seen_actions.add(a)

    return float(len(seen_actions)) / num_actions


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectory(
    env,
    oracle: Agent,
    current_policy,
    beta: float,
    timesteps_per_episode: int,
    num_actions: int,
    n_criticality_samples: int = 0,
    noise_std: float = 0.02,
) -> list:
    """
    Roll out one episode and collect VIPER training samples.

    At each step:
    1. The *driving* policy (oracle if beta=1, tree otherwise) selects an
       action that is executed in the environment.
    2. The oracle is always queried for the *label* (independent of who drives).
    3. A criticality weight is computed for the state.

    Parameters
    ----------
    env : TensorforceEnvironment
        Result of ``Environment.create()``.
    oracle : Agent
        Loaded Tensorforce oracle (used read-only via ``independent=True``).
    current_policy : TreePolicy or None
        The current tree policy.  ``None`` in iteration 0 (oracle drives).
    beta : float
        Probability that the oracle drives the environment step.
        VIPER sets this to 1.0 for iteration 0, 0.0 for all later iterations.
    timesteps_per_episode : int
        Maximum steps before the episode is treated as done.
    num_actions : int
        Total number of discrete actions.
    n_criticality_samples : int
        Passed to ``estimate_criticality()``.
    noise_std : float
        Noise level for criticality estimation (ignored when
        ``n_criticality_samples == 0``).

    Returns
    -------
    list of (obs_array, oracle_action, weight) tuples
    """
    dataset = []
    state_dict = env.reset()

    for _ in range(timesteps_per_episode):
        obs, mask = _extract_obs_and_mask(state_dict)

        # Derive mask from binary prefix if not present in state dict
        if mask is None:
            mask = [bool(v > 0.5) for v in obs[:num_actions]]

        # ── Oracle label (always deterministic from oracle) ─────────────────
        oracle_action = int(oracle.act(states=state_dict, independent=True))

        # ── Driving action ───────────────────────────────────────────────────
        use_oracle = (current_policy is None) or (np.random.random() < beta)
        if use_oracle:
            action = oracle_action
        else:
            action = current_policy.predict(obs, mask)

        # ── Criticality weight ───────────────────────────────────────────────
        weight = estimate_criticality(
            oracle, state_dict, num_actions,
            n_samples=n_criticality_samples,
            noise_std=noise_std,
        )

        dataset.append((obs, oracle_action, weight))

        # ── Step environment ─────────────────────────────────────────────────
        next_state_dict, terminal, _ = env.execute(actions=action)
        state_dict = next_state_dict

        if terminal:
            break

    return dataset


# ---------------------------------------------------------------------------
# Tree policy evaluation
# ---------------------------------------------------------------------------

def evaluate_tree_policy(
    env,
    tree_policy: TreePolicy,
    n_episodes: int,
    timesteps_per_episode: int,
    num_actions: int,
) -> tuple:
    """
    Evaluate *tree_policy* by rolling out *n_episodes* episodes.

    Parameters
    ----------
    env : TensorforceEnvironment
    tree_policy : TreePolicy
    n_episodes : int
    timesteps_per_episode : int
    num_actions : int
        Used to derive the action mask when it is not in the state dict.

    Returns
    -------
    (mean_reward, std_reward) : (float, float)
    """
    episode_rewards = []

    for _ in range(n_episodes):
        state_dict = env.reset()
        total_reward = 0.0

        for _ in range(timesteps_per_episode):
            obs, mask = _extract_obs_and_mask(state_dict)
            if mask is None:
                mask = [bool(v > 0.5) for v in obs[:num_actions]]

            action = tree_policy.predict(obs, mask)
            state_dict, terminal, reward = env.execute(actions=action)
            total_reward += reward

            if terminal:
                break

        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


# ---------------------------------------------------------------------------
# Main VIPER training loop
# ---------------------------------------------------------------------------

def train_viper(
    agent_dir: str,
    n_iter: int = 80,
    max_depth: int = None,
    max_leaves: int = None,
    n_eval_episodes: int = 5,
    timesteps_per_episode: int = 100,
    save_path: str = None,
    n_criticality_samples: int = 0,
    noise_std: float = 0.02,
    criterion: str = 'entropy',
    ccp_alpha: float = 0.0001,
    verbose: bool = True,
) -> TreePolicy:
    """
    Run the VIPER algorithm to distil a trained Tensorforce PPO oracle into a
    ``DecisionTreeClassifier`` policy.

    Parameters
    ----------
    agent_dir : str
        Path to the saved Tensorforce agent directory.
        Example: ``'agents/ppo1 - 66 states throughput'``.
    n_iter : int
        Number of VIPER iterations.  Each iteration collects one episode of
        experience and refits the tree on the accumulated dataset.
    max_depth : int or None
        Maximum depth of the extracted tree.  ``None`` = unlimited.
    max_leaves : int or None
        Maximum number of leaf nodes.  ``None`` = unlimited.
    n_eval_episodes : int
        Episodes used to evaluate each candidate tree after fitting.
        The tree with the highest mean reward across *all* iterations is kept.
    timesteps_per_episode : int
        Maximum timesteps per episode.  Must match the value used when
        the oracle was trained (default 100, see ``run.py``).
    save_path : str or None
        Joblib path to save the best tree policy.  ``None`` = do not save.
    n_criticality_samples : int
        Samples for the noise-sensitivity criticality estimator.
        ``0`` = uniform weights (DAgger-style, faster).
        ``>0`` = perturb the observation *n* times and measure how often the
        oracle changes its action.  Higher values give a more accurate
        estimate but increase wall-clock time linearly.
    noise_std : float
        Standard deviation of the Gaussian noise added to the observation for
        criticality estimation (only used when ``n_criticality_samples > 0``).
    criterion : str
        Splitting criterion for the DecisionTreeClassifier.
        ``'entropy'`` is recommended (see VIPER LEARNINGS).
    ccp_alpha : float
        Minimal cost-complexity pruning parameter.
        ``0.0001`` works well in practice (see VIPER LEARNINGS).
    verbose : bool
        Print per-iteration progress via tqdm.

    Returns
    -------
    TreePolicy
        The best-performing tree policy found across all iterations.

    Notes
    -----
    * Only ``TRANSP_AGENT_ACTION_MAPPING = 'direct'`` is supported.
    * The SimPy simulation inside ``ProductionEnv`` is *continuous* – it does
      not restart on ``env.reset()``.  Each call to ``reset()`` increments
      the episode counter and continues the ongoing simulation.  This mirrors
      the behaviour of ``run.py`` / ``test.py``.
    * The oracle is used in ``independent=True`` mode throughout, meaning no
      experience is buffered and no gradient updates are performed.
    """
    # ------------------------------------------------------------------
    # 1. Build environment and derive action-space size
    # ------------------------------------------------------------------
    env = Environment.create(
        environment='production.envs.ProductionEnv',
        max_episode_timesteps=timesteps_per_episode,
    )

    inner_env = env.environment
    actions_spec = inner_env.actions()

    if 'shape' in actions_spec:
        raise NotImplementedError(
            "VIPER currently supports 'direct' action mapping only. "
            "Set TRANSP_AGENT_ACTION_MAPPING='direct' in initialize_env.py."
        )

    num_actions = actions_spec['num_values']

    # ------------------------------------------------------------------
    # 2. Load oracle
    # ------------------------------------------------------------------
    oracle = load_oracle(agent_dir, env)

    if verbose:
        print("=" * 60)
        print("VIPER  –  Policy Extraction from PPO Oracle")
        print("=" * 60)
        print(f"  Oracle dir            : {agent_dir}")
        print(f"  Iterations            : {n_iter}")
        print(f"  Timesteps/episode     : {timesteps_per_episode}")
        print(f"  Eval episodes/iter    : {n_eval_episodes}")
        print(f"  max_depth             : {max_depth}")
        print(f"  max_leaves            : {max_leaves}")
        print(f"  Criterion             : {criterion}  (ccp_alpha={ccp_alpha})")
        weight_mode = (
            f"noise-sensitivity (n={n_criticality_samples}, std={noise_std})"
            if n_criticality_samples else "uniform"
        )
        print(f"  Criticality weights   : {weight_mode}")
        print(f"  Save path             : {save_path}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 3. VIPER main loop
    # ------------------------------------------------------------------
    dataset = []            # accumulated (obs, oracle_action, weight) triples
    current_policy = None   # None in iteration 0 → oracle drives the env
    policies = []
    rewards = []

    for i in tqdm(range(n_iter), desc="VIPER", disable=not verbose):
        # beta=1.0 in iteration 0 (oracle drives); 0.0 afterwards (tree drives)
        beta = 1.0 if i == 0 else 0.0

        # ── Collect one episode ──────────────────────────────────────────────
        new_samples = collect_trajectory(
            env=env,
            oracle=oracle,
            current_policy=current_policy,
            beta=beta,
            timesteps_per_episode=timesteps_per_episode,
            num_actions=num_actions,
            n_criticality_samples=n_criticality_samples,
            noise_std=noise_std,
        )
        dataset.extend(new_samples)

        # ── Build training arrays ────────────────────────────────────────────
        X = np.array([s[0] for s in dataset], dtype=np.float32)
        y = np.array([s[1] for s in dataset], dtype=np.int32)
        w = np.array([s[2] for s in dataset], dtype=np.float64)

        # ── Fit decision tree ────────────────────────────────────────────────
        clf = DecisionTreeClassifier(
            criterion=criterion,
            ccp_alpha=ccp_alpha,
            max_depth=max_depth,
            max_leaf_nodes=max_leaves,
            random_state=42,
        )
        clf.fit(X, y, sample_weight=w)
        current_policy = TreePolicy(clf)
        policies.append(current_policy)

        # ── Evaluate tree policy ─────────────────────────────────────────────
        mean_r, std_r = evaluate_tree_policy(
            env=env,
            tree_policy=current_policy,
            n_episodes=n_eval_episodes,
            timesteps_per_episode=timesteps_per_episode,
            num_actions=num_actions,
        )
        rewards.append(mean_r)

        if verbose:
            tqdm.write(
                f"[{i + 1:3d}/{n_iter}]  "
                f"dataset={len(dataset):6d}  "
                f"depth={clf.get_depth():3d}  "
                f"leaves={clf.get_n_leaves():4d}  "
                f"reward={mean_r:+.4f} ± {std_r:.4f}"
            )

    # ------------------------------------------------------------------
    # 4. Select and optionally save the best policy
    # ------------------------------------------------------------------
    best_idx = int(np.argmax(rewards))
    best_policy = policies[best_idx]

    if verbose:
        print("\n" + "=" * 60)
        print(f"VIPER complete.  Best iteration : {best_idx + 1} / {n_iter}")
        print(f"Best mean reward               : {rewards[best_idx]:.4f}")
        best_policy.print_info()
        print("=" * 60)

    if save_path is not None:
        best_policy.save(save_path)

    oracle.close()
    env.close()

    return best_policy
