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
from util import compute_absolute_trajectory_error

# Detect and use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RLLoopAgent(nn.Module):
    def __init__(self, obs_size, action_size):
        super(RLLoopAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, doMatch):
        self.buffer.append((state, action, reward, next_state, done, doMatch))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, matches = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones, matches

    def __len__(self):
        return len(self.buffer)

# Training settings
EPISODES = 2
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
REPLAY_CAPACITY = 1000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
EVAL_INTERVAL = 2

obs_size = 10
action_size = 1
agent = RLLoopAgent(obs_size, action_size).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
criterion = nn.BCELoss()

save_dir = f"runs/rl_loops_ate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=save_dir)

loop_model = LoopModel()
loop_model.load("/home/mundus/svillhaue213/AI_project/SNNLOOP-main/MODELSTESTING/LOOP_AUTO_128_128_16_EPOCHS100_DENSE_128_16")

def extract_features(i_ref, i_cur):
    diff = np.abs(i_ref - i_cur).mean()
    return np.array([diff] * obs_size)

steps_done = 0
epsilon = EPS_START
correct_decisions = 0

# for episode in range(EPISODES):
#     ds = DataSimulator("/home/mundus/svillhaue213/AI_project/UCAMGEN-main/SAMPLE_RANDOM", loadImages=True)
#     optimizer_nn = GraphOptimizer(initialID=0, minLoops=5, doFilter=False)


DS_NOISES = [[0.625, np.pi / (180 * 4)], [2.5, np.pi / 180], [5, 2 * np.pi / 180]]

for episode in range(EPISODES):
    motionSigma, angleSigma = random.choice(DS_NOISES)
    ds = DataSimulator(
        "/home/mundus/svillhaue213/AI_project/UCAMGEN-main/SAMPLE_RANDOM",
        loadImages=True,
        motionSigma=motionSigma,
        angleSigma=angleSigma
    )

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
            state = torch.tensor(features, dtype=torch.float32, device=device)

            with torch.no_grad():
                loop_pred = loop_model.predict([i_ref[np.newaxis], curImage[np.newaxis]])[0][0]
                q_value = agent(state)
                action = 1 if random.random() > epsilon else int(random.random() < 0.5)

            _, theMotion, doMatch = ds.match_images(candID, curID, addNoise=True, simulateFailures=True)

            reward = 0.0
            if action == 1:
                try:
                    optimizer_copy = optimizer_nn.copy()
                    scaledMotion = theMotion.reshape((3,1)) * np.array([10, 10, 1]).reshape((3,1))
                    optimizer_copy.add_loop(candID, curID, scaledMotion, ds.odoCovariance)
                    optimizer_copy.validate()
                    optimizer_copy.optimize()

                    new_traj = np.array([v.pose for v in optimizer_copy.theVertices]).T
                    base_traj = np.array([v.pose for v in optimizer_nn.theVertices]).T
                    gt_traj = ds.gtOdom[0:3, 1:]

                    _, ate_before, _ = compute_absolute_trajectory_error(base_traj, compose_trajectory(gt_traj))
                    _, ate_after, _ = compute_absolute_trajectory_error(new_traj, compose_trajectory(gt_traj))
                    reward = (ate_before - ate_after) * 10.0
                except:
                    reward = -5.0
            elif doMatch:
                reward = -0.1

            next_state = state.detach()
            done = False

            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, doMatch)
            total_reward += reward

            writer.add_scalar("Agent/q_value", q_value.item(), steps_done)
            writer.add_scalar("Agent/action", action, steps_done)
            writer.add_scalar("Agent/reward", reward, steps_done)
            writer.add_scalar("Agent/ground_truth_doMatch", int(doMatch), steps_done)

            if (action == 1 and doMatch) or (action == 0 and not doMatch):
                correct_decisions += 1
            writer.add_scalar("Agent/correct_decision_rate", correct_decisions / (steps_done + 1), steps_done)

            steps_done += 1

        allID.append(curID)
        allImages.append(curImage)

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones, matches = replay_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            targets = torch.tensor([[1.0] if m else [0.0] for m in matches], dtype=torch.float32, device=device)

            q_vals = agent(states)
            loss = criterion(q_vals, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), steps_done)

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    writer.add_scalar("Reward/episode", total_reward, episode)

    if episode % EVAL_INTERVAL == 0:
        torch.save(agent.state_dict(), os.path.join(save_dir, f"agent_ep{episode}.pt"))

print("Training complete.")
writer.close()
