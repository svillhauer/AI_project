import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from swinloopmodel import SwinLoopModel
from graphoptimizer_gpu import GraphOptimizerGPU
from transform2d import compose_trajectory
from imagematcher import ImageMatcher
from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error
import matplotlib.pyplot as plt



# ========== CONFIGURATION ==========
csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"
model_name = "microsoft/swin-tiny-patch4-window7-224"
output_dir = "./swin_slam_results"
os.makedirs(output_dir, exist_ok=True)

# Convert trajectories to numpy arrays
X_est = np.array(optimizer.poses).squeeze().T
X_gt = compose_trajectory(poses)  # Ground truth

# Plot both trajectories
plt.figure(figsize=(10, 10))
plt.plot(X_gt[0], X_gt[1], 'g--', label="Ground Truth Trajectory")   # dashed green line
plt.plot(X_est[0], X_est[1], 'k-', label="Optimized Trajectory")     # solid black line

# Plot loop closures
for i, j, label, pred, passed, added in loop_stats:
    if added:
        x1, y1 = optimizer.poses[i][:2]
        x2, y2 = optimizer.poses[j][:2]
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5)

# Optional: red arrows for loop closures
for i, j, label, pred, passed, added in loop_stats:
    if added:
        x1, y1 = optimizer.poses[i][:2]
        x2, y2 = optimizer.poses[j][:2]
        plt.arrow(x1, y1, x2 - x1, y2 - y1, color='red', head_width=20, alpha=0.5)

plt.axis('equal')
plt.grid(True)
plt.title("Swin-based SLAM: Optimized vs Ground Truth")
plt.legend()
plt.savefig(os.path.join(output_dir, "swin_slam_with_groundtruth.png"))
plt.show()

