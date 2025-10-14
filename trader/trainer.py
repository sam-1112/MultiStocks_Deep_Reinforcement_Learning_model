from algos.a2c import A2CAgent
from trader.envs.trading_env import TradingEnv

class Trainer:
    def __init__(self, env: TradingEnv, agent: A2CAgent, max_episodes=1000):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.agent.set_environment(env)

    def train(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state, reward, done = self.agent.actor_critic(state, done)
                total_reward += reward

            print(f"Episode {episode+1}/{self.max_episodes}, Total Reward: {total_reward}")