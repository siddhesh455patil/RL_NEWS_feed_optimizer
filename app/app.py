import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.preprocess import *
from agent.Q_Learning import QLearningAgent   


# ---------------- CONFIG ----------------
st.set_page_config(page_title="RL News Recommender", layout="wide")

st.title("Reinforcement Learning News Recommendation System")


# ---------------- SIDEBAR ----------------
st.sidebar.header(" Controls")

steps = st.sidebar.slider("Training Steps", 1000, 50000, 10000)
epsilon = st.sidebar.slider("Exploration (Epsilon)", 0.01, 1.0, 1.0)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_all():
    news, behaviors = load_data("data/train")
    news_map = build_news_category_map(news)
    rl_data = create_rl_dataset(behaviors, news_map)
    return rl_data

rl_data = load_all()
rl_data = rl_data[:50000]


# ---------------- INIT AGENT (PERSISTENT) ----------------
if "agent" not in st.session_state:
    n_states = len(CATEGORY_TO_INDEX)
    n_actions = len(CATEGORY_TO_INDEX)

    st.session_state.agent = QLearningAgent(n_states, n_actions)
    st.session_state.agent.epsilon = epsilon

agent = st.session_state.agent


# ---------------- TRAIN FUNCTION ----------------
def train_agent(steps):
    rewards = []
    correct = 0

    for i in range(min(steps, len(rl_data)-1)):

        current = rl_data[i]
        next_sample = rl_data[i+1]

        state = np.argmax(current["state"])
        next_state = np.argmax(next_sample["state"])

        action = agent.choose_action(state)

        # FIXED reward logic (CRITICAL)
        if action == current["action"]:
            reward = 3
            correct += 1
        else:
            reward = -1

        agent.update(state, action, reward, next_state)

        rewards.append(reward)

        # Epsilon decay
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    accuracy = correct / steps
    return rewards, accuracy


# ---------------- TRAIN BUTTON ----------------
if st.button("Train Model"):

    rewards, accuracy = train_agent(steps)

    st.success("Training Completed!")

    # ---------------- METRICS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")

    with col2:
        st.metric("Avg Reward", f"{np.mean(rewards):.2f}")

    # ---------------- PLOT ----------------
    st.subheader("Learning Curve")

    moving_avg = np.convolve(rewards, np.ones(200)/200, mode='valid')

    fig, ax = plt.subplots()
    ax.plot(moving_avg)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Smoothed Learning Curve")

    st.pyplot(fig)


# ---------------- RECOMMENDATION ----------------
st.subheader("Get Recommendation")

user_choice = st.selectbox(
    "Select User Interest",
    list(CATEGORY_TO_INDEX.keys())
)

if st.button("Recommend"):

    state_index = CATEGORY_TO_INDEX[user_choice]

    # Use trained Q-table
    state = tuple([0]*len(CATEGORY_TO_INDEX))
    state[CATEGORY_TO_INDEX[user_choice]] = 1
    state = tuple(state)
    
    if state in agent.q_table:
        q_values = agent.q_table[state]
        best_action = np.argmax(q_values)
    else:
        best_action = np.random.randint(len(CATEGORY_TO_INDEX))

    best_action = np.argmax(q_values)

    recommended = list(CATEGORY_TO_INDEX.keys())[best_action]

    st.success(f" Recommended Category: {recommended}")


# ---------------- DEBUG (OPTIONAL) ----------------
with st.expander("Debug Info"):
    st.write("Epsilon:", agent.epsilon)
    sample_items = list(agent.q_table.items())[:3]
        
    for state, values in sample_items:
        st.write("State:", state)
        st.write("Q-values:", values)