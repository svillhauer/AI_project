# train_rl_agent.py
# This script trains and saves the RL agent. 

# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from collections import deque
import random
import datetime

from datasimulator import DataSimulator
from loopmodel import LoopModel
from graphoptimizer import GraphOptimizer

# To avoid crashes, disable TensorFlow GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# ============================
# RL Agent Definition
# ============================
class RLLoopAgent(nn.Module):
    def __init__(self, obs_size, action_size):
        super(RLLoopAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Sigmoid()  # Output = loop acceptance probability
        )

    def forward(self, x):
        return self.net(x)


# ============================
# Experience Replay
# ============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# ============================
# Training Hyperparameters
# ============================
EPISODES = 1 # 3
MAX_STEPS = 50 # 300
BATCH_SIZE = 34 #2 #64
GAMMA = 0.99
LR = 1e-4
REPLAY_CAPACITY = 50 #500 #10000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
EVAL_INTERVAL = 10

# ============================
# Setup
# ============================
obs_size = 10  # Feature vector size per loop candidate
action_size = 1

# agent = RLLoopAgent(obs_size, action_size).cuda()
agent = RLLoopAgent(obs_size, action_size)  # Use CPU for testing

optimizer = optim.Adam(agent.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
criterion = nn.BCELoss()

save_dir = f"runs/rl_loops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=save_dir)

# Load LoopModel and Data
loop_model = LoopModel()
loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")
# loop_model.theModel.eval() # Can't use this, since our model uses keras

# ============================
# Feature Extraction Function
# ============================
def extract_features(i_ref, i_cur):
    diff = np.abs(i_ref - i_cur).mean()
    feat = np.array([diff] * obs_size)
    return feat

# ============================
# Training Loop
# ============================
steps_done = 0
epsilon = EPS_START

for episode in range(EPISODES):
    ds = DataSimulator("/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
    optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)
    
    preID, preImage = ds.get_image()
    allID = [preID]
    allImages = [preImage]
    total_reward = 0

    while ds.update():
        curID, curImage = ds.get_image()
        optimizer_nn.add_odometry(curID, np.zeros((3,1)), np.eye(3))

        candidateIDs = allID[:-1]
        candidateImages = allImages[:-1]

        for idx, candID in enumerate(candidateIDs):
            i_ref = candidateImages[idx]
            features = extract_features(i_ref, curImage)
            state = torch.tensor(features, dtype=torch.float32) # .cuda()

            with torch.no_grad():
                # loop_pred = loop_model.predict_one(i_ref, curImage)
                loop_pred = loop_model.predict([i_ref[np.newaxis], curImage[np.newaxis]])[0][0] # Because model uses sigmoid for binary classification 
                q_value = agent(state)
                action = 1 if random.random() > epsilon else int(random.random() < 0.5)

            reward = 1.0 if (loop_pred > 0.5 and action == 1) else -0.1
            next_state = state  # Or update with new image pairs if needed
            done = False

            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
            total_reward += reward

        allID.append(curID)
        allImages.append(curImage)

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states, dtype=torch.float32) #.cuda()
            actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1) # .cuda()
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) # .cuda()

            q_vals = agent(states)
            loss = criterion(q_vals, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), steps_done)

        steps_done += 1

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    writer.add_scalar("Reward/episode", total_reward, episode)

    if episode % EVAL_INTERVAL == 0:
        torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

print("Training complete.")
writer.close()
