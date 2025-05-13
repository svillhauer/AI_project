import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from collections import deque
import random
import datetime
import cv2
import time
import pandas as pd

from datasimulator import DataSimulator
from loopmodel import LoopModel
from graphoptimizer import GraphOptimizer
from util import compute_absolute_trajectory_error

# Disable GPU for TensorFlow (inside LoopModel)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ===== Agent definition =====
class RLLoopAgent(nn.Module):
    def __init__(self, obs_size):
        super(RLLoopAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Q-values for [reject, accept]
        )

    def forward(self, x):
        return self.net(x)

# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        priority = abs(reward) + (1.0 if reward > 0 else 0)
        self.priorities.append(priority)

    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

# ===== Parameters =====
EPISODES = 10
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
REPLAY_CAPACITY = 5000
EPS_START = 1.0
EPS_END = 0.2
EPS_DECAY = 0.998
TARGET_UPDATE = 100
EVAL_INTERVAL = 2
OBS_SIZE = 8
DS_NOISES = [[0.625, np.pi / (180 * 4)], [2.5, np.pi / 180], [5, 2 * np.pi / 180]]

# ===== Initialize =====
agent = RLLoopAgent(OBS_SIZE)
target_net = RLLoopAgent(OBS_SIZE)
target_net.load_state_dict(agent.state_dict())
target_net.eval()

optimizer = optim.Adam(agent.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
criterion = nn.MSELoss()

save_dir = f"runs/rl_loops_improved_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=save_dir)

loop_model = LoopModel()
loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

# ===== Feature Extraction =====
def extract_features(i_ref, i_cur):
    diff = np.abs(i_ref - i_cur).mean()
    hist_ref = cv2.calcHist([i_ref], [0], None, [16], [0, 256])
    hist_cur = cv2.calcHist([i_cur], [0], None, [16], [0, 256])
    hist_diff = cv2.compareHist(hist_ref, hist_cur, cv2.HISTCMP_BHATTACHARYYA)
    edges_ref = cv2.Canny(i_ref, 100, 200).mean()
    edges_cur = cv2.Canny(i_cur, 100, 200).mean()
    edge_diff = np.abs(edges_ref - edges_cur)
    loop_pred = loop_model.predict([i_ref[np.newaxis], i_cur[np.newaxis]])[0][0]
    return np.array([
        diff,
        hist_diff,
        edge_diff,
        loop_pred,
        diff * loop_pred,
        hist_diff * loop_pred,
        edge_diff * loop_pred,
        (diff + hist_diff + edge_diff) / 3
    ])

# ===== Training Loop =====
steps_done = 0
epsilon = EPS_START
correct_decisions = 0
episode_times = []

for episode in range(EPISODES):
    start_time = time.time()

    motionSigma, angleSigma = random.choice(DS_NOISES)
    ds = DataSimulator(
        "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM",
        loadImages=True,
        motionSigma=motionSigma,
        angleSigma=angleSigma
    )
    optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)

    preID, preImage = ds.get_image()
    allID = [preID]
    allImages = [preImage]
    total_reward = 0

    while ds.update():
        curID, curImage = ds.get_image()
        optimizer_nn.add_odometry(curID, np.zeros((3, 1)), np.eye(3))
        candidateIDs = allID[:-1]
        candidateImages = allImages[:-1]

        for idx, candID in enumerate(candidateIDs):
            i_ref = candidateImages[idx]
            features = extract_features(i_ref, curImage)
            state = torch.tensor(features, dtype=torch.float32)

            with torch.no_grad():
                q_values = agent(state)
                if random.random() > epsilon:
                    action = torch.argmax(q_values).item()
                else:
                    action = random.choice([0, 1])

            _, theMotion, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)

            if action == 1 and doMatch:
                reward = 1.0
            elif action == 1 and not doMatch:
                reward = -1.0
            elif action == 0 and doMatch:
                reward = -0.5
            else:
                reward = 0.2

            next_state = state
            done = False
            replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)
            total_reward += reward

            writer.add_scalar("Agent/q_value_accept", q_values[1].item(), steps_done)
            writer.add_scalar("Agent/q_value_reject", q_values[0].item(), steps_done)
            writer.add_scalar("Agent/action", action, steps_done)
            writer.add_scalar("Agent/reward", reward, steps_done)
            writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)

            if (action == 1 and doMatch) or (action == 0 and not doMatch):
                correct_decisions += 1
            writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)
            writer.add_scalar("Agent/epsilon", epsilon, steps_done)

            steps_done += 1

            if len(replay_buffer) >= BATCH_SIZE and steps_done % 4 == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    targets = rewards + (GAMMA * next_q_values * (1 - dones))

                current_q = agent(states).gather(1, actions)
                loss = criterion(current_q, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), steps_done)

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(agent.state_dict())

        allID.append(curID)
        allImages.append(curImage)

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    writer.add_scalar("Reward/episode", total_reward, episode)

    duration = time.time() - start_time
    episode_times.append(duration)
    writer.add_scalar("Time/episode_duration_sec", duration, episode)

    if episode % EVAL_INTERVAL == 0:
        torch.save({
            'episode': episode,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'steps': steps_done
        }, os.path.join(save_dir, f"agent_ep{episode}.pt"))

# ===== Finalize =====
writer.close()

# Save timing CSV
timing_df = pd.DataFrame({
    "episode": list(range(EPISODES)),
    "duration_sec": episode_times
})
timing_df.to_csv(os.path.join(save_dir, "episode_times.csv"), index=False)
print(f"Saved episode timing info to '{os.path.join(save_dir, 'episode_times.csv')}'")
print("Training complete.")




# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import os
# from collections import deque
# import random
# import datetime
# import cv2
# import time
# from datasimulator import DataSimulator
# from loopmodel import LoopModel
# from graphoptimizer import GraphOptimizer
# from util import compute_absolute_trajectory_error

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# class RLLoopAgent(nn.Module):
#     def __init__(self, obs_size):
#         super(RLLoopAgent, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)  # Q-values for [reject, accept]
#         )

#     def forward(self, x):
#         return self.net(x)

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#         self.priorities = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#         # Prioritize positive rewards and loop closures
#         priority = abs(reward) + (1.0 if reward > 0 else 0)
#         self.priorities.append(priority)

#     def sample(self, batch_size):
#         # Prioritized sampling
#         probs = np.array(self.priorities) / sum(self.priorities)
#         indices = np.random.choice(len(self.buffer), batch_size, p=probs)
#         batch = [self.buffer[i] for i in indices]
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return np.array(states), actions, rewards, np.array(next_states), dones

#     def __len__(self):
#         return len(self.buffer)

# # Hyperparameters
# EPISODES = 10  # Increased for better exploration
# BATCH_SIZE = 32
# GAMMA = 0.99
# LR = 1e-4
# REPLAY_CAPACITY = 5000  # Larger buffer
# EPS_START = 1.0
# EPS_END = 0.2  # Higher minimum exploration
# EPS_DECAY = 0.998  # Slower decay
# TARGET_UPDATE = 100  # Steps between target network updates
# EVAL_INTERVAL = 2
# OBS_SIZE = 8  # Reduced to match our actual features

# # Initialize networks
# agent = RLLoopAgent(OBS_SIZE)
# target_net = RLLoopAgent(OBS_SIZE)
# target_net.load_state_dict(agent.state_dict())
# target_net.eval()

# optimizer = optim.Adam(agent.parameters(), lr=LR)
# replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
# criterion = nn.MSELoss()

# # Setup logging
# save_dir = f"runs/rl_loops_improved_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# os.makedirs(save_dir, exist_ok=True)
# writer = SummaryWriter(log_dir=save_dir)

# # Load loop detection model
# loop_model = LoopModel()
# loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

# def extract_features(i_ref, i_cur):
#     """Extracts diverse visual features for loop closure detection"""
#     # Basic image difference
#     diff = np.abs(i_ref - i_cur).mean()
    
#     # Histogram comparison
#     hist_ref = cv2.calcHist([i_ref], [0], None, [16], [0,256])
#     hist_cur = cv2.calcHist([i_cur], [0], None, [16], [0,256])
#     hist_diff = cv2.compareHist(hist_ref, hist_cur, cv2.HISTCMP_BHATTACHARYYA)
    
#     # Edge features (simplified)
#     edges_ref = cv2.Canny(i_ref, 100, 200).mean()
#     edges_cur = cv2.Canny(i_cur, 100, 200).mean()
#     edge_diff = np.abs(edges_ref - edges_cur)
    
#     # Loop model prediction
#     loop_pred = loop_model.predict([i_ref[np.newaxis], i_cur[np.newaxis]])[0][0]
    
#     # Combined features
#     features = np.array([
#         diff,
#         hist_diff,
#         edge_diff,
#         loop_pred,
#         diff * loop_pred,  # Interaction term
#         hist_diff * loop_pred,
#         edge_diff * loop_pred,
#         (diff + hist_diff + edge_diff) / 3  # Combined similarity
#     ])
    
#     return features

# # Training loop
# steps_done = 0
# epsilon = EPS_START
# correct_decisions = 0
# DS_NOISES = [[0.625, np.pi / (180 * 4)], [2.5, np.pi / 180], [5, 2 * np.pi / 180]]


# episode_times = []  # to store duration of each episode


# for episode in range(EPISODES):
#     motionSigma, angleSigma = random.choice(DS_NOISES)
#     ds = DataSimulator(
#         "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM",
#         loadImages=True,
#         motionSigma=motionSigma,
#         angleSigma=angleSigma
#     )
    
#     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)
#     preID, preImage = ds.get_image()
#     allID = [preID]
#     allImages = [preImage]
#     total_reward = 0

#     while ds.update():
#         curID, curImage = ds.get_image()
#         optimizer_nn.add_odometry(curID, np.zeros((3, 1)), np.eye(3))

#         candidateIDs = allID[:-1]
#         candidateImages = allImages[:-1]

#         for idx, candID in enumerate(candidateIDs):
#             i_ref = candidateImages[idx]
#             features = extract_features(i_ref, curImage)
#             state = torch.tensor(features, dtype=torch.float32)

#             # Epsilon-greedy action selection
#             with torch.no_grad():
#                 q_values = agent(state)
#                 if random.random() > epsilon:
#                     action = torch.argmax(q_values).item()
#                 else:
#                     action = random.choice([0, 1])

#             # Get ground truth from simulator
#             _, theMotion, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)

#             # Balanced reward structure
#             if action == 1 and doMatch:    # Correct acceptance
#                 reward = 1.0
#             elif action == 1 and not doMatch:  # False acceptance
#                 reward = -1.0  
#             elif action == 0 and doMatch:  # False rejection
#                 reward = -0.5
#             else:  # Correct rejection
#                 reward = 0.2

#             # Store experience
#             next_state = state  # In this setup, next state is the same
#             done = False
#             replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)
#             total_reward += reward

#             # Logging
#             writer.add_scalar("Agent/q_value_accept", q_values[1].item(), steps_done)
#             writer.add_scalar("Agent/q_value_reject", q_values[0].item(), steps_done)
#             writer.add_scalar("Agent/action", action, steps_done)
#             writer.add_scalar("Agent/reward", reward, steps_done)
#             writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)

#             if (action == 1 and doMatch) or (action == 0 and not doMatch):
#                 correct_decisions += 1
#             writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)
#             writer.add_scalar("Agent/epsilon", epsilon, steps_done)

#             steps_done += 1

#             # Training step
#             if len(replay_buffer) >= BATCH_SIZE and steps_done % 4 == 0:  # Train every 4 steps
#                 states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
#                 states = torch.tensor(states, dtype=torch.float32)
#                 actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
#                 rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
#                 dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
#                 next_states = torch.tensor(next_states, dtype=torch.float32)

#                 # DQN update with target network
#                 with torch.no_grad():
#                     next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
#                     targets = rewards + (GAMMA * next_q_values * (1 - dones))

#                 current_q = agent(states).gather(1, actions)
#                 loss = criterion(current_q, targets)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 writer.add_scalar("Loss/train", loss.item(), steps_done)

#             # Update target network
#             if steps_done % TARGET_UPDATE == 0:
#                 target_net.load_state_dict(agent.state_dict())

#         allID.append(curID)
#         allImages.append(curImage)

#     # Decay epsilon
#     epsilon = max(EPS_END, epsilon * EPS_DECAY)
#     writer.add_scalar("Reward/episode", total_reward, episode)

#     # Save model checkpoint
#     if episode % EVAL_INTERVAL == 0:
#         torch.save({
#             'episode': episode,
#             'model_state_dict': agent.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'epsilon': epsilon,
#             'steps': steps_done
#         }, os.path.join(save_dir, f"agent_ep{episode}.pt"))

# print("Training complete.")
# writer.close()




# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import os
# from collections import deque
# import random
# import datetime

# from datasimulator import DataSimulator
# from loopmodel import LoopModel
# from graphoptimizer import GraphOptimizer
# from util import compute_absolute_trajectory_error

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # RL Agent with Q-values for two actions: [reject, accept]
# class RLLoopAgent(nn.Module):
#     def __init__(self, obs_size):
#         super(RLLoopAgent, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)  # Q-values for actions 0 (reject), 1 (accept)
#         )

#     def forward(self, x):
#         return self.net(x)

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return np.array(states), actions, rewards, np.array(next_states), dones

#     def __len__(self):
#         return len(self.buffer)

# EPISODES = 2
# BATCH_SIZE = 32
# GAMMA = 0.99
# LR = 1e-4
# REPLAY_CAPACITY = 1000
# EPS_START = 1.0
# EPS_END = 0.1
# EPS_DECAY = 0.995
# EVAL_INTERVAL = 1

# obs_size = 10
# agent = RLLoopAgent(obs_size)
# optimizer = optim.Adam(agent.parameters(), lr=LR)
# replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
# criterion = nn.MSELoss()

# save_dir = f"runs/rl_loops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# os.makedirs(save_dir, exist_ok=True)
# writer = SummaryWriter(log_dir=save_dir)

# loop_model = LoopModel()
# loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

# # Add loop_model prediction into the features
# def extract_features(i_ref, i_cur):
#     diff = np.abs(i_ref - i_cur).mean()
#     loop_pred = loop_model.predict([i_ref[np.newaxis], i_cur[np.newaxis]])[0][0]
#     return np.array([diff] * (obs_size - 1) + [loop_pred])

# steps_done = 0
# epsilon = EPS_START
# correct_decisions = 0

# DS_NOISES = [[0.625, np.pi / (180 * 4)], [2.5, np.pi / 180], [5, 2 * np.pi / 180]]

# for episode in range(EPISODES):
#     motionSigma, angleSigma = random.choice(DS_NOISES)
#     ds = DataSimulator("/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True, motionSigma=motionSigma, angleSigma=angleSigma)
#     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)

#     preID, preImage = ds.get_image()
#     allID = [preID]
#     allImages = [preImage]
#     total_reward = 0

#     while ds.update():
#         curID, curImage = ds.get_image()
#         optimizer_nn.add_odometry(curID, np.zeros((3, 1)), np.eye(3))

#         candidateIDs = allID[:-1]
#         candidateImages = allImages[:-1]

#         for idx, candID in enumerate(candidateIDs):
#             i_ref = candidateImages[idx]
#             features = extract_features(i_ref, curImage)
#             state = torch.tensor(features, dtype=torch.float32)

#             with torch.no_grad():
#                 q_values = agent(state)
#                 if random.random() > epsilon:
#                     action = torch.argmax(q_values).item()
#                 else:
#                     action = random.choice([0, 1])

#             _, theMotion, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)

#             if action == 1:
#                 reward = 1.0 if doMatch else -1.0
#             else:
#                 reward = -0.1 if doMatch else 0.0

#             next_state = state
#             done = False
#             replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)
#             total_reward += reward

#             writer.add_scalar("Agent/q_value_accept", q_values[1].item(), steps_done)
#             writer.add_scalar("Agent/action", action, steps_done)
#             writer.add_scalar("Agent/reward", reward, steps_done)
#             writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)

#             if (action == 1 and doMatch) or (action == 0 and not doMatch):
#                 correct_decisions += 1
#             writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)

#             steps_done += 1

#         allID.append(curID)
#         allImages.append(curImage)

#         if len(replay_buffer) >= BATCH_SIZE:
#             states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
#             states = torch.tensor(states, dtype=torch.float32)
#             actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
#             rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

#             q_vals = agent(states)
#             q_vals_chosen = q_vals.gather(1, actions)
#             loss = criterion(q_vals_chosen, rewards)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             writer.add_scalar("Loss/train", loss.item(), steps_done)

#     epsilon = max(EPS_END, epsilon * EPS_DECAY)
#     writer.add_scalar("Reward/episode", total_reward, episode)

#     if episode % EVAL_INTERVAL == 0:
#         torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

# print("Training complete.")
# writer.close()

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.tensorboard import SummaryWriter
# # import os
# # from collections import deque
# # import random
# # import datetime

# # from datasimulator import DataSimulator
# # from loopmodel import LoopModel
# # from graphoptimizer import GraphOptimizer
# # from util import compute_absolute_trajectory_error

# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # class RLLoopAgent(nn.Module):
# #     def __init__(self, obs_size, action_size):
# #         super(RLLoopAgent, self).__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(obs_size, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, action_size),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         return self.net(x)

# # class ReplayBuffer:
# #     def __init__(self, capacity):
# #         self.buffer = deque(maxlen=capacity)

# #     def push(self, state, action, reward, next_state, done, doMatch):
# #         self.buffer.append((state, action, reward, next_state, done, doMatch))

# #     def sample(self, batch_size):
# #         batch = random.sample(self.buffer, batch_size)
# #         states, actions, rewards, next_states, dones, matches = zip(*batch)
# #         return np.array(states), actions, rewards, np.array(next_states), dones, matches

# #     def __len__(self):
# #         return len(self.buffer)

# # EPISODES = 2
# # BATCH_SIZE = 32
# # GAMMA = 0.99
# # LR = 1e-4
# # REPLAY_CAPACITY = 1000
# # EPS_START = 1.0
# # EPS_END = 0.1
# # EPS_DECAY = 0.995
# # EVAL_INTERVAL = 2

# # obs_size = 10
# # action_size = 1
# # agent = RLLoopAgent(obs_size, action_size)
# # optimizer = optim.Adam(agent.parameters(), lr=LR)
# # replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
# # criterion = nn.BCELoss()

# # save_dir = f"runs/rl_loops_ate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # os.makedirs(save_dir, exist_ok=True)
# # writer = SummaryWriter(log_dir=save_dir)

# # loop_model = LoopModel()
# # loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")
# # #loop_model.load("/home/mundus/svillhaue213/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

# # def extract_features(i_ref, i_cur):
# #     diff = np.abs(i_ref - i_cur).mean()
# #     return np.array([diff] * obs_size)

# # steps_done = 0
# # epsilon = EPS_START
# # correct_decisions = 0

# # # for episode in range(EPISODES):
# # #     ds = DataSimulator("/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
# # #     #ds = DataSimulator("/home/mundus/svillhaue213/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
# # #     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)

# # DS_NOISES = [[0.625, np.pi / (180 * 4)], [2.5, np.pi / 180], [5, 2 * np.pi / 180]]

# # for episode in range(EPISODES):
# #     motionSigma, angleSigma = random.choice(DS_NOISES)
# #     ds = DataSimulator(
# #         "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM",
# #         loadImages=True,
# #         motionSigma=motionSigma,
# #         angleSigma=angleSigma
# #     )

# #     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)

# #     preID, preImage = ds.get_image()
# #     allID = [preID]
# #     allImages = [preImage]
# #     total_reward = 0


# #     while ds.update():
# #         curID, curImage = ds.get_image()
# #         optimizer_nn.add_odometry(curID, np.zeros((3,1)), np.eye(3))

# #         candidateIDs = allID[:-1]
# #         candidateImages = allImages[:-1]

# #         for idx, candID in enumerate(candidateIDs):
# #             i_ref = candidateImages[idx]
# #             features = extract_features(i_ref, curImage)
# #             state = torch.tensor(features, dtype=torch.float32)

# #             with torch.no_grad():
# #                 loop_pred = loop_model.predict([i_ref[np.newaxis], curImage[np.newaxis]])[0][0]
# #                 q_value = agent(state)
# #                 action = 1 if random.random() > epsilon else int(random.random() < 0.5)

# #             _, theMotion, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)

# #             reward = 0.0
# #             if action == 1:
# #                 try:
# #                     optimizer_copy = optimizer_nn.copy()
# #                     scaledMotion = theMotion.reshape((3,1)) * np.array([10, 10, 1]).reshape((3,1))
# #                     optimizer_copy.add_loop(candID, curID, scaledMotion, ds.odoCovariance)
# #                     optimizer_copy.validate()
# #                     optimizer_copy.optimize()

# #                     new_traj = np.array([v.pose for v in optimizer_copy.theVertices]).T
# #                     base_traj = np.array([v.pose for v in optimizer_nn.theVertices]).T
# #                     gt_traj = ds.gtOdom[0:3, 1:]

# #                     _, ate_before, _ = compute_absolute_trajectory_error(base_traj, compose_trajectory(gt_traj))
# #                     _, ate_after, _ = compute_absolute_trajectory_error(new_traj, compose_trajectory(gt_traj))
# #                     reward = (ate_before - ate_after) * 10.0
# #                 except:
# #                     reward = -5.0
# #             elif doMatch:
# #                 reward = -0.1

# #             next_state = state
# #             done = False

# #             replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done, doMatch)
# #             total_reward += reward

# #             writer.add_scalar("Agent/q_value", q_value.item(), steps_done)
# #             writer.add_scalar("Agent/action", action, steps_done)
# #             writer.add_scalar("Agent/reward", reward, steps_done)
# #             writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)

# #             if (action == 1 and doMatch) or (action == 0 and not doMatch):
# #                 correct_decisions += 1
# #             writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)

# #             steps_done += 1

# #         allID.append(curID)
# #         allImages.append(curImage)

# #         if len(replay_buffer) >= BATCH_SIZE:
# #             states, actions, rewards, next_states, dones, matches = replay_buffer.sample(BATCH_SIZE)
# #             states = torch.tensor(states, dtype=torch.float32)
# #             actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
# #             rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
# #             targets = torch.tensor([[1.0] if m else [0.0] for m in matches], dtype=torch.float32)

# #             q_vals = agent(states)
# #             loss = criterion(q_vals, targets)

# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #             writer.add_scalar("Loss/train", loss.item(), steps_done)

# #     epsilon = max(EPS_END, epsilon * EPS_DECAY)
# #     writer.add_scalar("Reward/episode", total_reward, episode)

# #     if episode % EVAL_INTERVAL == 0:
# #         torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

# # print("Training complete.")
# # writer.close()



# # # # Standard libraries for computation, machine learning, and logging
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.tensorboard import SummaryWriter
# # # import os
# # # from collections import deque
# # # import random
# # # import datetime

# # # # Import SLAM-related components
# # # from datasimulator import DataSimulator
# # # from loopmodel import LoopModel
# # # from graphoptimizer import GraphOptimizer
# # # from util import compute_absolute_trajectory_error

# # # # Disable GPU use for TensorFlow (used inside LoopModel) to avoid memory issues
# # # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # # # Define a simple neural network agent with 3 hidden layers and a sigmoid output
# # # class RLLoopAgent(nn.Module):
# # #     def __init__(self, obs_size, action_size):
# # #         super(RLLoopAgent, self).__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(obs_size, 128),
# # #             nn.ReLU(),
# # #             nn.Linear(128, 64),
# # #             nn.ReLU(),
# # #             nn.Linear(64, action_size),
# # #             nn.Sigmoid()  # Outputs probability of accepting a loop
# # #         )

# # #     def forward(self, x):
# # #         return self.net(x)

# # # # Define a replay buffer to store past experiences for training
# # # class ReplayBuffer:
# # #     def __init__(self, capacity):
# # #         self.buffer = deque(maxlen=capacity)

# # #     def push(self, state, action, reward, next_state, done, doMatch):
# # #         self.buffer.append((state, action, reward, next_state, done, doMatch))

# # #     def sample(self, batch_size):
# # #         batch = random.sample(self.buffer, batch_size)
# # #         states, actions, rewards, next_states, dones, matches = zip(*batch)
# # #         return np.array(states), actions, rewards, np.array(next_states), dones, matches

# # #     def __len__(self):
# # #         return len(self.buffer)

# # # # Training parameters
# # # EPISODES = 1
# # # MAX_STEPS = 50
# # # BATCH_SIZE = 34
# # # GAMMA = 0.99
# # # LR = 1e-4
# # # REPLAY_CAPACITY = 50
# # # EPS_START = 1.0  # Start fully random
# # # EPS_END = 0.1    # End mostly greedy
# # # EPS_DECAY = 0.995
# # # EVAL_INTERVAL = 10

# # # # Create agent and related objects
# # # obs_size = 10
# # # action_size = 1
# # # agent = RLLoopAgent(obs_size, action_size)
# # # optimizer = optim.Adam(agent.parameters(), lr=LR)
# # # replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
# # # criterion = nn.BCELoss()  # Binary cross-entropy loss

# # # # Set up TensorBoard logging
# # # save_dir = f"runs/rl_loops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # # os.makedirs(save_dir, exist_ok=True)
# # # writer = SummaryWriter(log_dir=save_dir)

# # # # Load pretrained neural network loop detector
# # # loop_model = LoopModel()
# # # loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

# # # # Define a simple feature extraction: average image difference
# # # def extract_features(i_ref, i_cur):
# # #     diff = np.abs(i_ref - i_cur).mean()
# # #     return np.array([diff] * obs_size)

# # # # Training loop
# # # steps_done = 0
# # # epsilon = EPS_START
# # # correct_decisions = 0

# # # for episode in range(EPISODES):
# # #     # Load simulator and SLAM graph
# # #     ds = DataSimulator("/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
# # #     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)

# # #     preID, preImage = ds.get_image()
# # #     allID = [preID]
# # #     allImages = [preImage]
# # #     total_reward = 0

# # #     while ds.update():  # For each timestep
# # #         curID, curImage = ds.get_image()
# # #         optimizer_nn.add_odometry(curID, np.zeros((3,1)), np.eye(3))

# # #         candidateIDs = allID[:-1]
# # #         candidateImages = allImages[:-1]

# # #         for idx, candID in enumerate(candidateIDs):
# # #             # Prepare input for agent
# # #             i_ref = candidateImages[idx]
# # #             features = extract_features(i_ref, curImage)
# # #             state = torch.tensor(features, dtype=torch.float32)

# # #             with torch.no_grad():
# # #                 # Predict loop likelihood using pretrained NN
# # #                 loop_pred = loop_model.predict([i_ref[np.newaxis], curImage[np.newaxis]])[0][0]
# # #                 q_value = agent(state)
# # #                 # Epsilon-greedy: either use agent or explore randomly
# # #                 action = 1 if random.random() > epsilon else int(random.random() < 0.5)

# # #             # Use simulator to get ground truth loop match and motion
# # #             _, theMotion, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)

# # #             # Reward scheme
# # #             if action == 1:
# # #                 # Accepting a loop: reward if correct, punish if wrong
# # #                 reward = 1.0 if doMatch else -1.0
# # #             else:
# # #                 # Rejecting a loop: small penalty if it was actually valid
# # #                 reward = -0.1 if doMatch else 0.0

# # #             next_state = state
# # #             done = False  # Not used here

# # #             # Store experience
# # #             replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done, doMatch)
# # #             total_reward += reward

# # #             # Log to TensorBoard
# # #             writer.add_scalar("Agent/q_value", q_value.item(), steps_done)
# # #             writer.add_scalar("Agent/action", action, steps_done)
# # #             writer.add_scalar("Agent/reward", reward, steps_done)
# # #             writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)
# # #             if (action == 1 and doMatch) or (action == 0 and not doMatch):
# # #                 correct_decisions += 1
# # #             writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)

# # #             steps_done += 1

# # #         allID.append(curID)
# # #         allImages.append(curImage)

# # #         # Update the agent if enough experience has been collected
# # #         if len(replay_buffer) >= BATCH_SIZE:
# # #             states, actions, rewards, next_states, dones, matches = replay_buffer.sample(BATCH_SIZE)
# # #             states = torch.tensor(states, dtype=torch.float32)
# # #             actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
# # #             rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
# # #             targets = torch.tensor([[1.0] if m else [0.0] for m in matches], dtype=torch.float32)

# # #             q_vals = agent(states)
# # #             loss = criterion(q_vals, targets)

# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()

# # #             writer.add_scalar("Loss/train", loss.item(), steps_done)

# # #     epsilon = max(EPS_END, epsilon * EPS_DECAY)
# # #     writer.add_scalar("Reward/episode", total_reward, episode)

# # #     if episode % EVAL_INTERVAL == 0:
# # #         torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

# # # print("Training complete.")
# # # writer.close()




# # # # train_rl_agent.py (Updated with full TensorBoard logging)
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.tensorboard import SummaryWriter
# # # import os
# # # from collections import deque
# # # import random
# # # import datetime

# # # from datasimulator import DataSimulator
# # # from loopmodel import LoopModel
# # # from graphoptimizer import GraphOptimizer

# # # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # # class RLLoopAgent(nn.Module):
# # #     def __init__(self, obs_size, action_size):
# # #         super(RLLoopAgent, self).__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(obs_size, 128),
# # #             nn.ReLU(),
# # #             nn.Linear(128, 64),
# # #             nn.ReLU(),
# # #             nn.Linear(64, action_size),
# # #             nn.Sigmoid()
# # #         )

# # #     def forward(self, x):
# # #         return self.net(x)

# # # class ReplayBuffer:
# # #     def __init__(self, capacity):
# # #         self.buffer = deque(maxlen=capacity)

# # #     def push(self, state, action, reward, next_state, done, doMatch):
# # #         self.buffer.append((state, action, reward, next_state, done, doMatch))

# # #     def sample(self, batch_size):
# # #         batch = random.sample(self.buffer, batch_size)
# # #         states, actions, rewards, next_states, dones, matches = zip(*batch)
# # #         return np.array(states), actions, rewards, np.array(next_states), dones, matches

# # #     def __len__(self):
# # #         return len(self.buffer)

# # # EPISODES = 1
# # # MAX_STEPS = 50
# # # BATCH_SIZE = 34
# # # GAMMA = 0.99
# # # LR = 1e-4
# # # REPLAY_CAPACITY = 50
# # # EPS_START = 1.0
# # # EPS_END = 0.1
# # # EPS_DECAY = 0.995
# # # EVAL_INTERVAL = 10

# # # obs_size = 10
# # # action_size = 1
# # # agent = RLLoopAgent(obs_size, action_size)
# # # optimizer = optim.Adam(agent.parameters(), lr=LR)
# # # replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
# # # criterion = nn.BCELoss()

# # # save_dir = f"runs/rl_loops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # # os.makedirs(save_dir, exist_ok=True)
# # # writer = SummaryWriter(log_dir=save_dir)

# # # loop_model = LoopModel()
# # # loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

# # # def extract_features(i_ref, i_cur):
# # #     diff = np.abs(i_ref - i_cur).mean()
# # #     return np.array([diff] * obs_size)

# # # steps_done = 0
# # # epsilon = EPS_START
# # # correct_decisions = 0

# # # for episode in range(EPISODES):
# # #     ds = DataSimulator("/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
# # #     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)

# # #     preID, preImage = ds.get_image()
# # #     allID = [preID]
# # #     allImages = [preImage]
# # #     total_reward = 0

# # #     while ds.update():
# # #         curID, curImage = ds.get_image()
# # #         optimizer_nn.add_odometry(curID, np.zeros((3,1)), np.eye(3))

# # #         candidateIDs = allID[:-1]
# # #         candidateImages = allImages[:-1]

# # #         for idx, candID in enumerate(candidateIDs):
# # #             i_ref = candidateImages[idx]
# # #             features = extract_features(i_ref, curImage)
# # #             state = torch.tensor(features, dtype=torch.float32)

# # #             with torch.no_grad():
# # #                 loop_pred = loop_model.predict([i_ref[np.newaxis], curImage[np.newaxis]])[0][0]
# # #                 q_value = agent(state)
# # #                 action = 1 if random.random() > epsilon else int(random.random() < 0.5)

# # #             _, _, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)
# # #             reward = 1.0 if doMatch and action == 1 else -1.0 if not doMatch and action == 1 else 0.0
# # #             next_state = state
# # #             done = False

# # #             replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done, doMatch)
# # #             total_reward += reward

# # #             # Log per step
# # #             writer.add_scalar("Agent/q_value", q_value.item(), steps_done)
# # #             writer.add_scalar("Agent/action", action, steps_done)
# # #             writer.add_scalar("Agent/reward", reward, steps_done)
# # #             writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)

# # #             if (action == 1 and doMatch) or (action == 0 and not doMatch):
# # #                 correct_decisions += 1
# # #             writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)

# # #             steps_done += 1

# # #         allID.append(curID)
# # #         allImages.append(curImage)

# # #         if len(replay_buffer) >= BATCH_SIZE:
# # #             states, actions, rewards, next_states, dones, matches = replay_buffer.sample(BATCH_SIZE)
# # #             states = torch.tensor(states, dtype=torch.float32)
# # #             actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
# # #             rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
# # #             targets = torch.tensor([[1.0] if m else [0.0] for m in matches], dtype=torch.float32)

# # #             q_vals = agent(states)
# # #             loss = criterion(q_vals, targets)

# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()

# # #             writer.add_scalar("Loss/train", loss.item(), steps_done)

# # #     epsilon = max(EPS_END, epsilon * EPS_DECAY)
# # #     writer.add_scalar("Reward/episode", total_reward, episode)

# # #     if episode % EVAL_INTERVAL == 0:
# # #         torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

# # # print("Training complete.")
# # # writer.close()




# # # # train_rl_agent.py
# # # # This script trains and saves the RL agent. 

# # # # Import libraries
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.tensorboard import SummaryWriter
# # # import os
# # # from collections import deque
# # # import random
# # # import datetime

# # # from datasimulator import DataSimulator
# # # from loopmodel import LoopModel
# # # from graphoptimizer import GraphOptimizer

# # # # To avoid crashes, disable TensorFlow GPU
# # # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# # # # ============================
# # # # RL Agent Definition
# # # # ============================
# # # class RLLoopAgent(nn.Module):
# # #     def __init__(self, obs_size, action_size):
# # #         super(RLLoopAgent, self).__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(obs_size, 128),
# # #             nn.ReLU(),
# # #             nn.Linear(128, 64),
# # #             nn.ReLU(),
# # #             nn.Linear(64, action_size),
# # #             nn.Sigmoid()  # Output = loop acceptance probability
# # #         )

# # #     def forward(self, x):
# # #         return self.net(x)


# # # # ============================
# # # # Experience Replay
# # # # ============================
# # # class ReplayBuffer:
# # #     def __init__(self, capacity):
# # #         self.buffer = deque(maxlen=capacity)

# # #     def push(self, state, action, reward, next_state, done):
# # #         self.buffer.append((state, action, reward, next_state, done))

# # #     def sample(self, batch_size):
# # #         batch = random.sample(self.buffer, batch_size)
# # #         states, actions, rewards, next_states, dones = zip(*batch)
# # #         return np.array(states), actions, rewards, np.array(next_states), dones

# # #     def __len__(self):
# # #         return len(self.buffer)


# # # # ============================
# # # # Training Hyperparameters
# # # # ============================
# # # EPISODES = 1 # 3
# # # MAX_STEPS = 50 # 300
# # # BATCH_SIZE = 34 #2 #64
# # # GAMMA = 0.99
# # # LR = 1e-4
# # # REPLAY_CAPACITY = 50 #500 #10000
# # # EPS_START = 1.0
# # # EPS_END = 0.1
# # # EPS_DECAY = 0.995
# # # EVAL_INTERVAL = 10

# # # # ============================
# # # # Setup
# # # # ============================
# # # obs_size = 10  # Feature vector size per loop candidate
# # # action_size = 1

# # # # agent = RLLoopAgent(obs_size, action_size).cuda()
# # # agent = RLLoopAgent(obs_size, action_size)  # Use CPU for testing

# # # optimizer = optim.Adam(agent.parameters(), lr=LR)
# # # replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
# # # criterion = nn.BCELoss()

# # # save_dir = f"runs/rl_loops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# # # os.makedirs(save_dir, exist_ok=True)
# # # writer = SummaryWriter(log_dir=save_dir)

# # # # Load LoopModel and Data
# # # loop_model = LoopModel()
# # # loop_model.load("/home/svillhauer/Desktop/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")
# # # # loop_model.theModel.eval() # Can't use this, since our model uses keras

# # # # ============================
# # # # Feature Extraction Function
# # # # ============================
# # # def extract_features(i_ref, i_cur):
# # #     diff = np.abs(i_ref - i_cur).mean()
# # #     feat = np.array([diff] * obs_size)
# # #     return feat

# # # # ============================
# # # # Training Loop
# # # # ============================
# # # steps_done = 0
# # # epsilon = EPS_START

# # # for episode in range(EPISODES):
# # #     ds = DataSimulator("/home/svillhauer/Desktop/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
# # #     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)
    
# # #     preID, preImage = ds.get_image()
# # #     allID = [preID]
# # #     allImages = [preImage]
# # #     total_reward = 0

# # #     while ds.update():
# # #         curID, curImage = ds.get_image()
# # #         optimizer_nn.add_odometry(curID, np.zeros((3,1)), np.eye(3))

# # #         candidateIDs = allID[:-1]
# # #         candidateImages = allImages[:-1]

# # #         for idx, candID in enumerate(candidateIDs):
# # #             i_ref = candidateImages[idx]
# # #             features = extract_features(i_ref, curImage)
# # #             state = torch.tensor(features, dtype=torch.float32) # .cuda()

# # #             with torch.no_grad():
# # #                 # loop_pred = loop_model.predict_one(i_ref, curImage)
# # #                 loop_pred = loop_model.predict([i_ref[np.newaxis], curImage[np.newaxis]])[0][0] # Because model uses sigmoid for binary classification 
# # #                 q_value = agent(state)
# # #                 action = 1 if random.random() > epsilon else int(random.random() < 0.5)

# # #             #reward = 1.0 if (loop_pred > 0.5 and action == 1) else -0.1
# # #             # After you get `doMatch` from dataSimulator.match_images(...)
# # #             reward = 1.0 if doMatch and action == 1 else -1.0 if not doMatch and action == 1 else 0.0 # Reward +1: for correctly accepting a valid loop
# # #                                                                                                     # Penalty -1: for falsely accepting a bad loop (false positive)
# # #                                                                                                     # Reward 0: for rejections (either good or bad), as long as you're mainly penalizing FP
# # #             next_state = state  # Or update with new image pairs if needed
# # #             done = False

# # #             replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)
# # #             total_reward += reward

# # #         allID.append(curID)
# # #         allImages.append(curImage)

# # #         if len(replay_buffer) >= BATCH_SIZE:
# # #             states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
# # #             states = torch.tensor(states, dtype=torch.float32) #.cuda()
# # #             actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1) # .cuda()
# # #             rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) # .cuda()

# # #             q_vals = agent(states)
# # #             target = torch.tensor([[1.0] if doMatch else [0.0] for ...])  # target is 1 if it's truly a loop
# # #             loss = criterion(q_vals, target)

# # #             #loss = criterion(q_vals, actions)

# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()

# # #             writer.add_scalar("Loss/train", loss.item(), steps_done)

# # #         steps_done += 1

# # #     epsilon = max(EPS_END, epsilon * EPS_DECAY)
# # #     writer.add_scalar("Reward/episode", total_reward, episode)

# # #     if episode % EVAL_INTERVAL == 0:
# # #         torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

# # # print("Training complete.")
# # # writer.close()
