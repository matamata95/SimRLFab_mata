import os
import builtins
import io
import unittest.mock as mock

from tqdm import tqdm
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from sklearn.tree import DecisionTreeClassifier

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

AGENT_PATH = "agents/ppo1 - 66 states throughput"
DATA_PATH = "data/data_66_states_throughput/"
X_PATH = os.path.join(DATA_PATH, "X/"); os.makedirs(X_PATH, exist_ok=True)
Y_PATH = os.path.join(DATA_PATH, "y/"); os.makedirs(Y_PATH, exist_ok=True)

TIMESTEPS = 10 ** 2
EPISODES_COLLECT = 10 ** 3

tf_env = Environment.create(environment='production.envs.ProductionEnv', max_episode_timesteps=TIMESTEPS)
agent = Agent.load(directory=AGENT_PATH, format='tensorflow', environment=tf_env)

X, y = [], []
for ep in tqdm(range(EPISODES_COLLECT)):
    state = tf_env.reset()
    terminal = False
    while not terminal:
        action = agent.act(states=state)
        X.append(state['observation'])  # state is a dictionary that contains 'observation' and 'action_mask'
        y.append(action)  # action is an integer representing the index of the 'action_mask' taken by the agent
        next_state, terminal, reward = tf_env.execute(actions=action)
        agent.observe(reward=reward, terminal=terminal)
        state = next_state

with open(os.path.join(X_PATH, "X.txt"), "w") as f:
    for i, state in enumerate(X):
        f.write("%s %s\n" % (i, state))

with open(os.path.join(Y_PATH, "y.txt"), "w") as f:
    for i, action in enumerate(y):
        f.write("%s %s\n" % (i, action))

# clf = DecisionTreeClassifier(max_depth=12)
# clf.fit(X, y)
# joblib.dump(clf, 'tree_policy.joblib')