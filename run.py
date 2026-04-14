from logger import export_statistics_logging
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from tensorforce.agents import Agent
import tensorflow as tf
from datetime import datetime
import os

# date_time = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
AGENT_SAVE_PATH = os.path.join('agents', 'ppo1')


# tf.set_random_seed(10)
os.makedirs(AGENT_SAVE_PATH, exist_ok=True)

timesteps = 10 ** 3  # Set time steps per episode
episodes = 10 ** 3  # Set number of episodes

# Define environment
environment_production = Environment.create(
    environment='production.envs.ProductionEnv',
    max_episode_timesteps=timesteps,
)

# Tensorforce runner
agent = Agent.create(
    agent='config/ppo2.json',
    environment=environment_production,
    saver={
        'directory': os.path.join(AGENT_SAVE_PATH, 'model-checkpoint'),
        'frequency': 10,
        'max-checkpoints': 5
    }
)
runner = Runner(agent=agent,
                environment=environment_production)
environment_production.agents = runner.agent

# Run training
runner.run(num_episodes=episodes)

environment_production.environment.statistics.update({'time_end': environment_production.environment.env.now})
export_statistics_logging(statistics=environment_production.environment.statistics,
                          parameters=environment_production.environment.parameters,
                          resources=environment_production.environment.resources)
# # Save agent
agent.save(directory=AGENT_SAVE_PATH, format='tensorflow')
