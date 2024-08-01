import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from src.DCQN_Network import Network
from src.DCQN_Dueling_Network import DuelingDQN
from src.Preprocess_Frame_Inputs import preprocess_frame

class Agent():
    def __init__(self, action_size, learning_rate=5e-4, minibatch_size=64, discount_factor=0.99, use_dueling=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.use_dueling = use_dueling
        
        if self.use_dueling:
            self.local_qnetwork = DuelingDQN(action_size).to(self.device)
            self.target_qnetwork = DuelingDQN(action_size).to(self.device)
        else:
            self.local_qnetwork = Network(action_size).to(self.device)
            self.target_qnetwork = Network(action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.minibatch_size:
            experiences = random.sample(self.memory, k=self.minibatch_size)
            self.learn(experiences)

    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()