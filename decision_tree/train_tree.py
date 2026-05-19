import os
import numpy as np
import pandas as pd
from pathlib import Path
import builtins
import io
import unittest.mock as mock

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

from production.envs.initialize_env import define_production_parameters
from simpy import Environment

# ? Start Log suppression.

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

# ? End log suppression.

X_PATH = "data/data_66_states_throughput/X/"
Y_PATH = "data/data_66_states_throughput/y/"

def build_feature_names(parameters):
    """
    Returns a list of human-readable names for every element of the state vector,
    built from parameters dictionary.

    State vector depends on the order of features defined in Transport.calculate_state().
    Current order is: 
        1. valid-action mask                - depends on the amount of machines/sources/sinks and their mapping, defined in parameters and if waiting action is enabled. Waiting action is LAST.
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
        names.append(f"act_{i:02d}: {label}")

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
    
    if parameters['TRANSP_AGENT_WAITING_ACTION']:
        labels.append("Waiting action")

    return labels

env = Environment()
parameters = define_production_parameters(env, 0)
feature_names = build_feature_names(parameters)
actions = build_action_labels_from_parameters(parameters)
NUM_ACTIONS = len(actions)

# ------------------------- Load dataset -------------------------
X, y = [], []
X = pd.read_csv(os.path.join(X_PATH, "X.txt"), sep=",", header=None, index_col=0)
y = pd.read_csv(os.path.join(Y_PATH, "y.txt"), sep=" ", header=None, index_col=0)

X_actions = X.iloc[:, :NUM_ACTIONS-1]  # actions without waiting action
X_features = X.iloc[:, NUM_ACTIONS-1:]

if not Path(X_PATH, "X.csv").exists() and not Path(Y_PATH, "y.csv").exists():
    print("CSV files do not exist, creating dataset...")
    if parameters['TRANSP_AGENT_WAITING_ACTION']:
        print("Waiting action is enabled, inserting 1.0 at the end of the valid action mask to reflect that waiting action is always valid.")
        X_actions.insert(loc=NUM_ACTIONS-1, column=NUM_ACTIONS, value=1.0)
        print("Finished inserting waiting action column.")

    X = pd.concat([X_actions, X_features], axis=1)

    #  Last value of state vector contains a trailing ']' that needs to be removed.
    print("Removing trailing '] from last column of X")
    for i in range(X.shape[0]):
        val = X.iloc[i, -1]
        if isinstance(val, str):
            X.iloc[i, -1] = val.replace("]", "")
            X.iloc[i, -1] = np.float(X.iloc[i, -1])
    print("Finished removing trailing ']' from last column of X.")

    X.rename(columns=lambda x: feature_names[x], inplace=True)
    y.rename(columns={1: "action"}, inplace=True)

    print("Saving dataset to CSV files...")
    X.to_csv(os.path.join(X_PATH, "X.csv"), index=False)
    y.to_csv(os.path.join(Y_PATH, "y.csv"), index=False)
    print("Finished saving dataset to CSV files.")
else:
    print("CSV files already exist, loading dataset from CSV files...")
    X = pd.read_csv(os.path.join(X_PATH, "X.csv"))
    y = pd.read_csv(os.path.join(Y_PATH, "y.csv"))
    print("Finished loading dataset from CSV files.")

#------------------------- Train decision tree -------------------------
print("Training decision tree...")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

depth = None
random_state = None
max_features = None
clf = DecisionTreeClassifier(max_depth=depth, max_features=max_features, random_state=random_state)
clf.fit(X_train, y_train)

print("Finished training decision tree.")

# print("Plotting decision tree...")
# fig, ax = plt.subplots(figsize=(20, 10))
# plot_tree(clf, filled=True, feature_names=feature_names, class_names=[str(a) for a in actions], ax=ax)
# fig.savefig("tree_visualization.svg", format="svg", bbox_inches="tight")
# print("Finished plotting decision.")
# plt.show()

plt.barh(list(X.columns), clf.feature_importances_)
plt.show()

#-------------------------- Test decision tree -------------------------
print("Testing decision tree...")
y_test_predicted = clf.predict(X_test)
print(metrics.classification_report(y_test, y_test_predicted))
print(metrics.confusion_matrix(y_test, y_test_predicted))
print("Finished testing decision tree.")
