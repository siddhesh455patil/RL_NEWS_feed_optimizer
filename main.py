import numpy as np

from utils.data_loader import load_data
from utils.preprocess import *
from utils.plot import plot_rewards

from agent.bandit import EpsilonGreedyBandit
from agent.Q_Learning import QLearningAgent


# ---------------- LOAD DATA ----------------
news, behaviors = load_data("data/train")

news_map = build_news_category_map(news)

rl_data = create_rl_dataset(behaviors, news_map)

# Reduce dataset for faster learning
rl_data = rl_data[:50000]

print("Total samples:", len(rl_data))
print("Sample:", rl_data[0])


# =========================================================
#  BANDIT MODEL
# =========================================================

print("\n--- Training Bandit ---")

n_actions = len(CATEGORY_TO_INDEX)
bandit = EpsilonGreedyBandit(n_actions=n_actions, epsilon=0.1)

bandit_rewards = []

for i, sample in enumerate(rl_data):

    action = bandit.select_action()

    # Correct reward logic
    if action == sample["action"]:
        reward = 2
    else:
        reward = -1

    bandit.update(action, reward)
    bandit_rewards.append(reward)

print("Bandit Avg Reward:", np.mean(bandit_rewards))


# =========================================================
# Q-LEARNING MODEL
# =========================================================

print("\n--- Training Q-Learning ---")

n_states = len(CATEGORY_TO_INDEX)
agent = QLearningAgent(n_states, n_actions)

# Better exploration
agent.epsilon = 1.0
agent.epsilon_min = 0.01
agent.epsilon_decay = 0.995

q_rewards = []
correct = 0

for i in range(len(rl_data) - 1):

    current = rl_data[i]
    next_sample = rl_data[i + 1]

    state = np.argmax(current["state"])
    next_state = np.argmax(next_sample["state"])

    action = agent.choose_action(state)

    # Correct reward logic (CRITICAL FIX)
    if action == current["action"]:
        reward = 3
        correct += 1
    else:
        reward = -1

    agent.update(state, action, reward, next_state)

    q_rewards.append(reward)

    # Epsilon decay
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    if i % 10000 == 0:
        print(f"Step {i}, Avg Reward: {np.mean(q_rewards):.4f}")


# ---------------- METRICS ----------------
accuracy = correct / len(rl_data)
ctr = sum([1 for i in range(len(rl_data)) if rl_data[i]["action"] == np.argmax(rl_data[i]["state"])]) / len(rl_data)

print("\n--- Final Results ---")
print("Q-Learning Accuracy:", round(accuracy, 4))
print("Average Reward:", round(np.mean(q_rewards), 4))
print("CTR (approx):", round(ctr, 4))


# ---------------- PLOT ----------------
plot_rewards(q_rewards)