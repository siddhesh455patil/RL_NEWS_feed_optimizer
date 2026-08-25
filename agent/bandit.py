import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        
        self.q_values = np.zeros(n_actions)   # estimated rewards
        self.action_counts = np.zeros(n_actions)

    def select_action(self):
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Exploitation
        return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        
        # Incremental mean update
        self.q_values[action] += (
            reward - self.q_values[action]
        ) / self.action_counts[action]