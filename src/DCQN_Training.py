import numpy as np
from collections import deque
import torch

class DCQN_Train:
    def __init__(self, env, agent, number_episodes=2000, max_timesteps_per_episode=10000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.agent = agent
        self.number_episodes = number_episodes
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start
        self.scores_on_100_episodes = deque(maxlen=100)
        self.all_scores = []

    def train(self):
        for episode in range(1, self.number_episodes + 1):
            state, _ = self.env.reset()
            score = 0
            for t in range(self.max_timesteps_per_episode):
                action = self.agent.act(state, self.epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            self.scores_on_100_episodes.append(score)
            self.all_scores.append(score)
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_on_100_episodes)), end="")
            if episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_on_100_episodes)))
            if np.mean(self.scores_on_100_episodes) >= 500.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(self.scores_on_100_episodes)))
                self.save_model('solved_model.pth')
                break
        return self.all_scores

    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.agent.local_qnetwork.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")