import os
import builtins
import io
import unittest.mock as mock


AGENT_FOLDER = 'ppo1'
AGENT_PATH = os.path.join('agents', AGENT_FOLDER)
STATES_PATH = r"C:\Users\grulovicma\Matija Grulovic\GitHub\SimRLFab_mata\log\expl_log\agent_reward_log.csv"
TIME_STEPS = 10 ** 2
EPISODES = 10 ** 1
TOP_K = 10


# ?Log-suppression because Jesus Christ.

_real_makedirs = os.makedirs
_real_open = builtins.open

# Open os.devnull once to get a real fd that supports fileno() and fsync()
_null_fd = _real_open(os.devnull, 'w')

class _NullFile(io.StringIO):
    """StringIO that also supports fileno() by pointing to os.devnull."""
    def fileno(self):
        return _null_fd.fileno()

def _suppress_log_makedirs(path, *args, **kwargs):
    if os.path.normpath(str(path)).startswith('log' + os.sep):
        return
    return _real_makedirs(path, *args, **kwargs)

def _suppress_log_open(path, *args, **kwargs):
    if os.path.normpath(str(path)).startswith('log' + os.sep):
        return _NullFile()
    return _real_open(path, *args, **kwargs)

mock.patch('os.makedirs', _suppress_log_makedirs).start()
mock.patch('builtins.open', _suppress_log_open).start()
mock.patch('os.fsync', lambda fd: None).start()

import simpy
import numpy as np
import pandas as pd
import shap
import copy
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from production.envs.initialize_env import define_production_parameters
from production.envs.production_env import ProductionEnv

env = simpy.Environment()
tf_env = Environment.create(
    environment='production.envs.ProductionEnv',
    max_episode_timesteps=TIME_STEPS,
)
agent = Agent.load(directory=AGENT_PATH, format='tensorflow', environment=tf_env)
parameters = define_production_parameters(env, 0)


def build_feature_names(parameters):
    """
    Returns a list of human-readable names for every element of the state vector,
    built from parameters dictionary.

    State vector depends on the order of features defined in Transport.calculate_state().
    Current order is: 
        1. valid-action mask                - depends on the amount of machines/sources/sinks and their mapping, defined in parameters
        2. bin_buffer_fill                  - num_machines + num_sources
        3. bin_location                     - num_machines + num_sources + num_sinks 
        4. bin_machine_failure              - num_machines
        5. int_buffer_fill                  - num_machines + num_sources
        6. rel_buffer_fill                  - num_machines + num_sources
        7. rel_buffer_fill_in_out           - num_machines*2 + num_sources
        8. order_waiting_time               - num_machines + num_sources
        9. order_waiting_time_normalized    - num_machines + num_sources
        10. distance_to_action              - num_machines + num_sources
        11. remaining_process_time          - num_machines
        12. total_process_time              - num_machines
    """
    num_m  = parameters['NUM_MACHINES']
    num_so = parameters['NUM_SOURCES']
    num_si = parameters['NUM_SINKS']


    # Build action labels first - state space always contains valid-action mask as first block
    action_labels = build_action_labels_from_parameters(parameters)
    names = []

    # Block 1: valid-action mask — one entry per mapping action
    for i, label in enumerate(action_labels):
        names.append(f"valid_action[{i:02d}]  ({label})")

    # Block 2: bin_buffer_fill - one entry per machine/source
    if 'bin_buffer_fill' in parameters['TRANSP_AGENT_STATE']:
        for i in range(num_m):
            names.append(f"order_at_machine_{i}")
        for i in range(num_so):
            names.append(f"order_at_source_{i}")

    # Block 3: bin_location - one-hot encoding of current location (machine/source/sink id) 
    if 'bin_location' in parameters['TRANSP_AGENT_STATE']:
        for i in range(num_m):
            names.append(f"at_machine_{i}")
        for i in range(num_so):
            names.append(f"at_source_{i}")
        for i in range(num_si):
            names.append(f"at_sink_{i}")

    # Block 4: bin_machine_failure - one entry per machine, 1.0 if broken, 0.0 if working
    if 'bin_machine_failure' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_broken")

    # Block 5: int_buffer_fill - numerical value of how many orders are at the resource (machine/source)
    if 'int_buffer_fill' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_has_orders")
        for s in range(num_so):
            names.append(f"source_{s}_has_orders")

    # Block 6: rel_buffer_fill - float value of how full the resource buffers are (machine/source), 1.0 means full, 0.5 means half full, etc. 
    if 'rel_buffer_fill' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_is_filled")
        for s in range(num_so):
            names.append(f"source_{s}_is_filled")

    # Block 7: rel_buffer_fill_in_out — NUM_MACHINES*2 + NUM_SOURCES = 19 entries
    if 'rel_buffer_fill_in_out' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_buffer_in_free")
            names.append(f"machine_{m}_buffer_out_free")
        for s in range(num_so):
            src_id = num_m + s           # matches res.id = NUM_MACHINES + src_list_idx
            names.append(f"source_{s}_(id={src_id})_buffer_out_free")

    # Block 8: order_waiting_time - numerical value of how long the oldest waiting order has been waiting at the resource (machine/source)
    if 'order_waiting_time' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_order_waiting_time")
        for s in range(num_so):
            names.append(f"source_{s}_order_waiting_time")

    # Block 9: order_waiting_time_normalized
    if 'order_waiting_time_normalized' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_order_waiting_time_normalized")
        for s in range(num_so):
            names.append(f"source_{s}_order_waiting_time_normalized")

    # Block 10: distance_to_action - numerical value of how far the agent is from the action destination, defined by MAX_TRANSPORT_TIME in parameters
    if 'distance_to_action' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"distance_from_machine_{m}_to_action")
        for s in range(num_so):
            names.append(f"distance_from_source_{s}_to_action")

    # Block 11: remaining_process_time - numerical value of how much processing time is left for the current order at the machine (0 if no order)
    if 'remaining_process_time' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_remaining_process_time")

    # Block 12: total_process_time - numerical value of total processing time for the current order at the machine (0 if no order has been processed)
    if 'total_process_time' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_total_process_time")

    return names


def build_action_labels_from_parameters(parameters):
    """
    Builds action label strings from parameters.

    Mirrors the mapping loop in Transport.__init__:
      First block:  source -> machine  (iterating RESP_AREA_SOURCE)
      Second block: machine -> sink    (iterating RESP_AREA_SINK)

    IDs match resource creation in initialize_env.py:
      machine id = machine list index (0, 1, ..., NUM_MACHINES-1)
      source  id = NUM_MACHINES + source list index
      sink    id = NUM_MACHINES + NUM_SOURCES + sink list index
    """
    num_m  = parameters['NUM_MACHINES']
    num_so = parameters['NUM_SOURCES']
    labels = []

    # source -> machine
    for src_idx, mach_ids in enumerate(parameters['RESP_AREA_SOURCE']):
        src_id = num_m + src_idx
        for mid in mach_ids:
            labels.append(f"source_{src_id}_to_machine_{mid}")

    # machine -> sink
    for sink_idx, mach_ids in enumerate(parameters['RESP_AREA_SINK']):
        sink_id = num_m + num_so + sink_idx
        for mid in mach_ids:
            labels.append(f"machine_{mid}_to_sink_{sink_id}")

    return labels


def grad_wrt_state(state, W0, b0, W1, b1, Wo, bo, action_idx):
    """
    Analytical gradient of log P(action_idx) with respect to the input state vector.
    Returns:
        dstate  — shape (feature_dim,)   signed sensitivity of log P(a) per state feature
        probs   — shape (action_dim,)   action probability distribution
    
    dstate - is signed sensitivity. Positive values mean that increasing
        the feature will increase the log-probability of the chosen action,
        while negative values mean that increasing the feature will decrease
        the log-probability of the chosen action.
    """
    # Activation function per tensorforce PPO policy network
    ACTIVATION = np.tanh
    h0 = ACTIVATION(state @ W0 + b0)
    h1 = ACTIVATION(h0 @ W1 + b1)
    logits = h1 @ Wo + bo
    logits -= logits.max()
    exp = np.exp(logits); probs = exp / exp.sum()

    dlogits = copy.deepcopy(-probs); dlogits[action_idx] += 1.0
    dh1 = dlogits @ Wo.T  *  (1.0 - h1**2)       # propagating back through linear + tanh, derivative of tanh is (1 - tanh^2)
    dh0 = dh1 @ W1.T  *  (1.0 - h0**2)           # propagating back through dense1 + tanh
    dstate = dh0 @ W0.T                          # propagating back through dense0

    return dstate, probs


def forward(state, W0, b0, W1, b1, Wo, bo):
    """
    Full policy forward pass in numpy. Returns (logits, probs, h0, h1).
    """
    ACTIVATION = np.tanh
    h0 = ACTIVATION(state @ W0 + b0)          # (64,)
    h1 = ACTIVATION(h0 @ W1 + b1)             # (32,)
    logits = h1 @ Wo + bo                     # (action_dim,)  — linear head
    logits -= logits.max()                    # numerical stability
    exp = np.exp(logits)
    probs = exp / exp.sum()                   # softmax
    return logits, probs, h0, h1


# ! implement a heuristic to select material with the longest waiting time at machine/source
# ! to compare to the agents decisions.

# ! use copilot to visualize logs

# ! try codex from openAI

def fetch_weights(agent):
    """
    Fetches all policy network weight matrices from the agent
    and returns them as numpy arrays for tensorforce agents.
    """
    W0 = np.array(agent.get_variable('policy/policy-network/dense0/weights'))   # (state_dim, hidden_dim1)
    b0 = np.array(agent.get_variable('policy/policy-network/dense0/bias'))      # (hidden_dim1,)
    W1 = np.array(agent.get_variable('policy/policy-network/dense1/weights'))   # (hidden_dim1, hidden_dim2)
    b1 = np.array(agent.get_variable('policy/policy-network/dense1/bias'))      # (hidden_dim2,)
    Wo = np.array(agent.get_variable('policy/action-distribution/deviations/deviations-linear/weights'))  # (hidden_dim2, action_dim) | output layer weights
    bo = np.array(agent.get_variable('policy/action-distribution/deviations/deviations-linear/bias'))     # (action_dim,)             | output layer bias
    return W0, b0, W1, b1, Wo, bo


def integrated_gradients(state, W0, b0, W1, b1, Wo, bo, action_idx, steps=50):
    """
    Integrated Gradients attribution from a zero-baseline (empty factory).
    IG_i ≈ state_i * mean( dlogP/dstate_i  along the interpolation path )
    Sums to:  logP(action_idx | state) - logP(action_idx | zeros)

    Returns:
        ig      — shape (feature_dim,)   attribution score per feature (signed, additive)
        probs   — shape (action_dim,)   actual action probabilities at the given state
    """
    # ?What is a good baseline for a factory state? Averaging all states during training is a good starting point.
    baseline = calculate_base_state(STATES_PATH)
    alphas = np.linspace(0.0, 1.0, steps)
    grad_acc = np.zeros_like(state)

    # Cumulating gradients along a straight path from baseline to the actual state
    for a in alphas:
        interp = baseline + a * (state - baseline)
        g, _ = grad_wrt_state(interp, W0, b0, W1, b1, Wo, bo, action_idx)
        grad_acc += g
    avg_grad = grad_acc / steps
    ig = (state - baseline) * avg_grad   # element-wise: x_i * mean_grad_i
    _, probs = grad_wrt_state(state, W0, b0, W1, b1, Wo, bo, action_idx)
    return ig, probs


def explain_action(state, action_idx, action_labels, feature_names,
                   W0, b0, W1, b1, Wo, bo, top_k=5, method='integrated_gradients'):
    """
    Returns a human-readable string explaining why the agent chose action_idx.

    method: 'gradient'  → plain input-level saliency
            'integrated_gradients' → IG attribution (recommended after training)
    """
    if method == 'integrated_gradients':
        attr, probs = integrated_gradients(state, W0, b0, W1, b1, Wo, bo, action_idx)
        method_label = "Integrated Gradients"
    else:
        attr, probs = grad_wrt_state(state, W0, b0, W1, b1, Wo, bo, action_idx)
        method_label = "Saliency"

    top_pos = np.argsort(-attr)[:top_k]
    top_neg = np.argsort( attr)[:top_k]

    lines = [
        f"Action chosen: [{action_idx}] {action_labels[action_idx]}",
        f"  P(chosen) = {probs[action_idx]:.3f}",
        f"  Top-3 alternatives: "
        + ", ".join(f"[{i}] {action_labels[i]} ({probs[i]:.3f})" for i in np.argsort(-probs)[1:4]),
        f"\n  [{method_label}] Top {top_k} features that INCREASED P(action):"
    ]
    for i in top_pos:
        lines.append(f"    +{attr[i]:+.4f}  {feature_names[i]}  (current value: {state[i]:.3f})")
    lines.append(f"  Top {top_k} features that DECREASED P(action):")
    for i in top_neg:
        lines.append(f"    {attr[i]:+.4f}  {feature_names[i]}  (current value: {state[i]:.3f})")

    return "\n".join(lines)

# ! implement sverl-p with "https://github.com/djeb20/SVERL_icml_2023" as a guideline
def sverl_p(state, background_states, critic_forward_fn, feature_names, top_k=10):
    """
    Computes SVERL-P Shapley values for a single state using KernelSHAP
    with the PPO critic V(s) as the performance characteristic function.
    """
    explainer = shap.KernelExplainer(critic_forward_fn, background_states)
    phi = explainer.shap_values(state.reshape(1, -1), nsamples=512)[0]
    ranking = np.argsort(-np.abs(phi))[:top_k]
    print(f"SVERL-P Explanation for current state value:")
    for i in ranking:
        print(f"  {phi[i]:+.4f}  {feature_names[i]}")
    return phi


def fetch_critic_weights(agent):
    Wc0 = np.array(agent.get_variable('baseline/baseline-network/state-dense0/weights'))  # (feature_dim, 64)
    bc0 = np.array(agent.get_variable('baseline/baseline-network/state-dense0/bias'))      # (64,)
    Wc1 = np.array(agent.get_variable('baseline/baseline-network/state-dense1/weights'))  # (64, 32)
    bc1 = np.array(agent.get_variable('baseline/baseline-network/state-dense1/bias'))      # (32,)
    Wv  = np.array(agent.get_variable('baseline/action-distribution/deviations/deviations-linear/weights'))       # (32, 1)
    bv  = np.array(agent.get_variable('baseline/action-distribution/deviations/deviations-linear/bias'))           # (1,)
    return Wc0, bc0, Wc1, bc1, Wv, bv


def critic_forward(state, Wc0, bc0, Wc1, bc1, Wv, bv):
    h0 = np.tanh(state @ Wc0 + bc0)   # (64,)
    h1 = np.tanh(h0   @ Wc1 + bc1)    # (32,)
    q  = h1 @ Wv + bv                  # (num_actions,) or (1,)
    if q.size == 1:
        return q.item()
    _, probs, _, _ = forward(state, *weights)   # weights is module-level
    return float(probs @ q)


critic_weights = fetch_critic_weights(agent)
def critic_value(masked_states):
    return np.array([critic_forward(s, *critic_weights) for s in masked_states])


def collect_background_states(agent, tf_env, n_episodes=10):
    collected = []
    for _ in range(n_episodes):
        state = tf_env.reset()
        terminal = False
        while not terminal:
            collected.append(np.array(state, dtype=np.float32))
            action = agent.act(states=state, independent=True)
            state, terminal, _ = tf_env.execute(actions=action)
    return np.array(collected)   # shape (N, feature_dim)


def calculate_base_state(df_path):
    df = pd.read_csv(df_path)

    states = df['state'][1::].to_list()
    states = [state[1:-1].split(",") for state in states]
    states = np.array(states, dtype=float)
    avg_state = np.zeros_like(states[0], dtype=float)

    for state in states:
        avg_state += state
    avg_state /= len(states)
    
    return avg_state



background_states = shap.kmeans(collect_background_states(agent, tf_env, n_episodes=EPISODES), k=50)

feature_names = build_feature_names(parameters)
action_labels = build_action_labels_from_parameters(parameters)

# contains state vectors of size (47,) for the following features:
# 1. valid action mask (20 entries)
# 2. bin_machine_failure (8 entries)
# 3. rel_buffer_fill_in_out (19 entries)
states_47 = [
    [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # valid actions first 20 entries
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # machine broken flags
    1.0, 1.0, 0.16666666666666663, 1.0, 0.33333333333333337, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0,  # machine rel buffers
    0.0, 0.0, 0.0  # source rel buffers
    ],
    [
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.33333333333333337, 1.0, 0.16666666666666663, 1.0, 0.5, 0.6666666666666667, 0.16666666666666663, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.33333333333333337, 1.0,
    0.0, 0.0, 0.0
    ]
]

# contains state vectors of size (80,) for the following features:
# 1. valid action mask (20 entries)
# 2. bin_location (14 entries)
# 3. bin_machine_failure (8 entries)
# 4. rel_buffer_fill_in_out (19 entries)
# 5. distance_to_action (11 entries)
# 6. total_process_time (8 entries)

states_80 = [
    [
    0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,  # valid action mask first 20 entries
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # agent location - machine/source/sink one-hot
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # machine failure flags
    1.0, 1.0, 1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.8333333333333334, 1.0, 0.16666666666666663, 0.0, 0.0, 0.0,  # rel buffers in/out
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # distance to action destination
    0.03564612344572155, 0.06746045665328886, 0.3626716318371299, 0.09322553295184832, 0.47325591487986696, 2.9510828526149706, 0.26557630611757477, 0.3229432733691529  # total process time for current orders at machines
    ]
]

current_state = states_80[0]
weights = fetch_weights(agent)
logits, probs, h0, h1 = forward(current_state, *weights)

best_action = int(np.argmax(probs))

action = explain_action(current_state, best_action, action_labels, feature_names, *weights, top_k=TOP_K, method='integrated_gradients')
print(action)

phi = sverl_p(
    state = np.array(current_state),
    background_states = background_states,
    critic_forward_fn = critic_value,
    feature_names = feature_names,
    top_k = TOP_K
)
