import os
import numpy as np
from tensorforce.agents import Agent
from production.envs.initialize_env import define_production_parameters
from production.envs.production_env import ProductionEnv
from tensorforce.environments import Environment
import simpy


AGENT_FOLDER = 'ppo1'
AGENT_PATH = os.path.join('agents', AGENT_FOLDER)

TIME_STEPS = 10 ** 2  # Must match run.py


def build_feature_names(parameters):
    """
    Returns a list of human-readable names for every element of the state vector,
    built purely from parameters (no live resources needed).

    Current state vector layout (47 elements for default config):
      [0  - 19]  valid-action mask          (len(mapping) = 20 entries)
      [20 - 27]  bin_machine_failure        (NUM_MACHINES = 8 entries)
      [28 - 43]  machine buffer_in/out free (NUM_MACHINES * 2 = 16 entries)
      [44 - 46]  source buffer_out free     (NUM_SOURCES = 3 entries)

    State vector depends on the order of features defined in Transport.calculate_state().
    Current order is: 
                    1. valid-action mask
                    2. bin_buffer_fill          - num_machines + num_sources
                    3. bin_location             - num_machines + num_sources + num_sinks 
                    4. bin_machine_failure      - num_machines
                    5. int_buffer_fill          - num_machines + num_sources
                    6. rel_buffer_fill          - num_machines + num_sources
                    7. rel_buffer_fill_in_out   - num_machines*2 + num_sources
                    8. order_waiting_time       - num_machines + num_sources
                    9. order_waiting_time_normalized    - num_machines + num_sources
                    10. distance_to_action      - num_machines + num_sources
                    11. remaining_process_time  - num_machines
                    12. total_process_time      - num_machines
    """
    num_m  = parameters['NUM_MACHINES']
    num_so = parameters['NUM_SOURCES']
    num_si = parameters['NUM_SINKS']

    # TODO: add remaining blocks for when vector space is expanded with more features (e.g. order waiting times, process times, etc.)

    # Build action labels first - state space always contains valid-action mask as first block
    action_labels = build_action_labels_from_parameters(parameters)
    names = []

    # Block 1: valid-action mask — one entry per mapping action
    for i, label in enumerate(action_labels):
        names.append(f"valid_action[{i:02d}]  ({label})")

    # TODO: BLOCK 2
    if 'bin_buffer_fill' in parameters['TRANSP_AGENT_STATE']:
        for i in range(num_m):
            names.append(f"order_at_machine_{i}")
        for i in range(num_so):
            names.append(f"order_at_source_{i}")

    # TODO: BLOCK 3
    # location of transport agent - one-hot encoding of current location (machine/source/sink id) 
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

    # TODO: BLOCK 5
    # Block 5: int_buffer_fill - numerical value of how many orders are at the resource (machine/source)
    if 'int_buffer_fill' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_has_orders")
        for s in range(num_so):
            names.append(f"source_{s}_has_orders")

    # TODO: BLOCK 6
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

    # TODO: BLOCK 8
    # Block 8: order_waiting_time - numerical value of how long the oldest waiting order has been waiting at the resource (machine/source)
    if 'order_waiting_time' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_order_waiting_time")
        for s in range(num_so):
            names.append(f"source_{s}_order_waiting_time")

    # TODO: BLOCK 9
    # Block 9: order_waiting_time_normalized
    if 'order_waiting_time_normalized' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_order_waiting_time_normalized")
        for s in range(num_so):
            names.append(f"source_{s}_order_waiting_time_normalized")

    # TODO: BLOCK 10
    # Block 10: distance_to_action - numerical value of how far the agent is from the action destination, defined by MAX_TRANSPORT_TIME in parameters
    if 'distance_to_action' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"distance_from_machine_{m}_to_action")
        for s in range(num_so):
            names.append(f"distance_from_source_{s}_to_action")

    # TODO: BLOCK 11
    # Block 11: remaining_process_time - numerical value of how much processing time is left for the current order at the machine (0 if no order)
    if 'remaining_process_time' in parameters['TRANSP_AGENT_STATE']:
        for m in range(num_m):
            names.append(f"machine_{m}_remaining_process_time")

    # TODO: BLOCK 12
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
      machine id = machine list index (0-based, 0 .. NUM_MACHINES-1)
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


def build_action_labels(mapping):
    """
    Builds action label strings from a live transport.mapping list.
    Handles regular [origin, destination] pairs, empty actions [-1, dest],
    and the waiting action [-1, -1].
    """
    labels = []
    for entry in mapping:
        if entry == [-1, -1] or entry == -1:
            labels.append("wait")
        elif isinstance(entry, list) and entry[0] == -1:
            dest = entry[1]
            labels.append(f"goto_{dest.type[0]}{dest.id}")
        elif isinstance(entry, list):
            o, d = entry[0], entry[1]
            labels.append(f"{o.type[0]}{o.id}->{d.type[0]}{d.id}")
        else:
            labels.append(str(entry))
    return labels


def grad_wrt_state(state, W0, b0, W1, b1, Wo, bo, action_idx):
    """
    Analytical gradient of log P(action_idx) w.r.t. the input state vector.
    Returns:
        dstate  — shape (47,)   signed sensitivity of log P(a) per state feature
        probs   — shape (20,)   action probability distribution
    """
    ACTIVATION = np.tanh
    h0 = ACTIVATION(state @ W0 + b0)
    h1 = ACTIVATION(h0 @ W1 + b1)
    logits = h1 @ Wo + bo
    logits -= logits.max()
    exp = np.exp(logits); probs = exp / exp.sum()

    dlogits = -probs.copy(); dlogits[action_idx] += 1.0
    dh1 = dlogits @ Wo.T  *  (1.0 - h1**2)       # through linear + tanh
    dh0 = dh1    @ W1.T  *  (1.0 - h0**2)        # through dense1 + tanh
    dstate = dh0 @ W0.T                          # through dense0

    return dstate, probs


def forward(state, W0, b0, W1, b1, Wo, bo):
    """Full policy forward pass in numpy. Returns (logits, probs, h0, h1)."""
    ACTIVATION = np.tanh
    h0 = ACTIVATION(state @ W0 + b0)          # (64,)
    h1 = ACTIVATION(h0 @ W1 + b1)             # (32,)
    logits = h1 @ Wo + bo                     # (20,)  — linear head
    logits -= logits.max()                    # numerical stability
    exp = np.exp(logits)
    probs = exp / exp.sum()                   # softmax
    return logits, probs, h0, h1


def fetch_weights(agent):
    """Fetches all policy network weight matrices from the agent and returns them as numpy arrays."""
    W0 = np.array(agent.get_variable('policy/policy-network/dense0/weights'))   # (47, 64) | state_dim x hidden_dim1
    b0 = np.array(agent.get_variable('policy/policy-network/dense0/bias'))       # (64,)
    W1 = np.array(agent.get_variable('policy/policy-network/dense1/weights'))   # (64, 32)
    b1 = np.array(agent.get_variable('policy/policy-network/dense1/bias'))       # (32,)
    Wo = np.array(agent.get_variable('policy/action-distribution/deviations/deviations-linear/weights'))  # (32, 20) | hidden_dim2 x action_dim
    bo = np.array(agent.get_variable('policy/action-distribution/deviations/deviations-linear/bias'))     # (20,)    | action_dim
    return W0, b0, W1, b1, Wo, bo


def integrated_gradients(state, W0, b0, W1, b1, Wo, bo, action_idx, steps=50):
    """
    Integrated Gradients attribution from a zero-baseline (empty factory).
    IG_i ≈ state_i * mean( dlogP/dstate_i  along the interpolation path )
    Sums to:  logP(action_idx | state) - logP(action_idx | zeros)

    Returns:
        ig      — shape (47,)   attribution score per feature (signed, additive)
        probs   — shape (20,)   actual action probabilities at the given state
    """
    baseline = np.zeros_like(state)
    alphas = np.linspace(0.0, 1.0, steps)
    grad_acc = np.zeros_like(state)
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

env = simpy.Environment()  # Needed to create a dummy env for fetching parameters
tf_env = Environment.create(
    environment='production.envs.ProductionEnv',
    max_episode_timesteps=TIME_STEPS,
)
parameters = define_production_parameters(env, 0)

agent = Agent.load(
    directory=AGENT_PATH,
    format='tensorflow',
    environment=tf_env
)

feature_names = build_feature_names(parameters)
action_labels = build_action_labels_from_parameters(parameters)

state_0 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # valid actions first 20 entries
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # machine broken flags
    1.0, 1.0, 0.16666666666666663, 1.0, 0.33333333333333337, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0, 0.16666666666666663, 1.0,  # machine rel buffers
    0.0, 0.0, 0.0  # source rel buffers
]

state_1 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.33333333333333337, 1.0, 0.16666666666666663, 1.0, 0.5, 0.6666666666666667, 0.16666666666666663, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.33333333333333337, 1.0,
    0.0, 0.0, 0.0
]

current_state = state_1
weights = fetch_weights(agent)
logits, probs, h0, h1 = forward(current_state, *weights)

best_action = int(np.argmax(probs))

action = explain_action(current_state, best_action, action_labels, feature_names, *weights, top_k=10, method='integrated_gradients')
print(action)

