import os

# Set TensorFlow to use deterministic operations for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random
import numpy as np

SEED = 10
random.seed(SEED)
np.random.seed(SEED)

from logger import export_statistics_logging
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from tensorforce.agents import Agent
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

AGENT_SAVE_PATH = os.path.join('agents', 'ppo1')

# ! set seed
tf.random.set_seed(SEED)
os.makedirs(AGENT_SAVE_PATH, exist_ok=True)

TIMESTEPS = 10 ** 2  # Set time steps per episode
EPISODES = 10 ** 3  # Set number of episodes

# Define environment
environment_production = Environment.create(
    environment='production.envs.ProductionEnv',
    max_episode_timesteps=TIMESTEPS,
)

# Tensorforce runner
agent = Agent.create(
    agent='config/ppo2.json',
    environment=environment_production,
    saver={
        'directory': os.path.join(AGENT_SAVE_PATH, 'model-checkpoint'),
        'frequency': 10,
        'max-checkpoints': 5,
        'load': False
    }
)
runner = Runner(agent=agent,
                environment=environment_production)
environment_production.agents = runner.agent

# Run training
runner.run(num_episodes=EPISODES)

environment_production.environment.statistics.update({'time_end': environment_production.environment.env.now})
export_statistics_logging(statistics=environment_production.environment.statistics,
                          parameters=environment_production.environment.parameters,
                          resources=environment_production.environment.resources)
# # Save agent
agent.save(directory=AGENT_SAVE_PATH, format='tensorflow')
