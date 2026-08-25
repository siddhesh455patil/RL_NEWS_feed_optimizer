import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):

        self.n_states = n_states
        self.n_actions = n_actions

        self.q_table = {}

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            
        if np.random.rand() < self.epsilon:
             return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])


    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
            
        best_next = np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )


    def decay_epsilon(self):
        # Gradually reduce exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)