import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm  # Added for progress bar
from swinloopmodel import SwinLoopModel
from graphoptimizer import GraphOptimizer
from transform2d import compose_trajectory
from imagematcher import ImageMatcher
from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error

# ========== CONFIGURATION ==========

csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_TEST/swin_siamese_best.pth"
model_name = "microsoft/swin-tiny-patch4-window7-224"
output_dir = "./swin_slam_results"
os.makedirs(output_dir, exist_ok=True)

PIX_TO_WORLD = np.array([10, 10, 1]).reshape((3, 1))
loopCovariance = np.diag([0.01, 0.01, 0.01])
odo_cov = np.diag([0.01, 0.01, 0.01])
threshold = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DATA LOADING ==========
print("[INFO] Loading loop ground truth...")
loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
loop_pairs = set(zip(loop_df["img1"], loop_df["img2"]))

print("[INFO] Loading odometry...")
poses_array = loadcsv(odom_file, delimiter=",")
if poses_array.shape[1] != 3:
    poses_array = poses_array.reshape(-1, 3)
poses = poses_array.tolist()
num_frames = len(poses)

# ========== INIT COMPONENTS ==========
print("[INFO] Initializing model and matcher...")
model = SwinLoopModel(model_path=model_path, model_name=model_name, device=device)
theMatcher = ImageMatcher()
optimizer = GraphOptimizer(initialID=0, initialPose=np.array(poses[0]).reshape(3, 1))

for i in range(1, len(poses)):
    Xodo = (np.array(poses[i]) - np.array(poses[i - 1])).reshape(3, 1)
    optimizer.add_odometry(i, Xodo, odo_cov)

loop_stats = []

# ========== LOOP SIMULATION WITH PROGRESS ==========
print("[INFO] Simulating loop closures...")
for i in tqdm(range(num_frames), desc="Loop closure simulation"):
    for j in range(i):
        img1_name = f"IMAGE{j+1:05d}.png"
        img2_name = f"IMAGE{i+1:05d}.png"
        doMatch = int((img1_name, img2_name) in loop_pairs or (img2_name, img1_name) in loop_pairs)

        img1 = Image.open(os.path.join(img_dir, img1_name)).convert("RGB")
        img2 = Image.open(os.path.join(img_dir, img2_name)).convert("RGB")
        score = model.predict((img1, img2))[0][0]
        is_loop = score > threshold

        predicted = int(is_loop)
        passed_ransac = 0
        added = 0

        if is_loop:
            img1_np = np.asarray(img1)
            img2_np = np.asarray(img2)
            theMatcher.define_images(img1_np, img2_np)
            theMatcher.estimate()

            if not theMatcher.hasFailed:
                matcherMotion = theMatcher.theMotion.reshape((3, 1)) * PIX_TO_WORLD
                optimizer.add_loop(j, i, matcherMotion, loopCovariance)
                passed_ransac = 1
                added = 1

        loop_stats.append([j, i, doMatch, predicted, passed_ransac, added])

# ========== VALIDATE + OPTIMIZE ==========
print("[INFO] Validating candidate loops...")
selected_loops = optimizer.validate()
print(f"[INFO] Loops retained after validation: {len(selected_loops) if selected_loops else 0}")

print("[INFO] Optimizing pose graph...")
optimizer.optimize()

# ========== METRICS & EVALUATION ==========
loop_array = np.array(loop_stats)
TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))
acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)

print("[INFO] Quality Metrics:")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, Fallout={fall:.4f}")

# ========== TRAJECTORY COMPARISON ==========
print("[INFO] Generating trajectories and evaluating errors...")
X_est = compose_trajectory(optimizer.poses)
X_gt = compose_trajectory(poses)

avg_err = evaluate_trajectory(X_est, np.array(poses).T)
ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)

print(f"[INFO] Average error per traveled distance: {avg_err:.4f}")
print(f"[INFO] ATE: mean={mean_ate:.4f}, std={std_ate:.4f}")

# ========== SAVE ==========
pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added"]).to_csv(
    os.path.join(output_dir, "swin_loop_stats.csv"), index=False)

print("[INFO] SLAM simulation complete. Stats saved.")


import matplotlib.pyplot as plt

# Plot trajectory
X_est = np.array(optimizer.poses).squeeze().T  # Estimated poses
plt.figure(figsize=(10, 10))
plt.plot(X_est[0], X_est[1], 'k-', label="Optimized Trajectory")

# Plot loop closures (in blue)
for i, j, label, pred, passed, added in loop_stats:
    if added:
        x1, y1 = optimizer.poses[i][:2]
        x2, y2 = optimizer.poses[j][:2]
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5)

# Optionally add loop closure arrows (in red)
for i, j, label, pred, passed, added in loop_stats:
    if added:
        x1, y1 = optimizer.poses[i][:2]
        x2, y2 = optimizer.poses[j][:2]
        plt.arrow(x1, y1, x2 - x1, y2 - y1, color='red', head_width=20, alpha=0.5)

plt.axis('equal')
plt.grid(True)
plt.title("Swin-based SLAM Loop Closures")
plt.legend()
plt.savefig(os.path.join(output_dir, "swin_slam_plot.png"))
plt.show()



# import os
# import numpy as np
# import torch
# import pandas as pd
# from PIL import Image
# from swinloopmodel import SwinLoopModel
# from graphoptimizer import GraphOptimizer
# from transform2d import compose_trajectory
# from imagematcher import ImageMatcher
# from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error

# # ========== CONFIGURATION ==========

# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"  # Your original dataset CSV
# #output_dir = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_TEST"         # Where to save train/val/test CSVs
# img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"    # Path to images 
# odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
# model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_TEST/swin_siamese_best.pth"
# model_name = "microsoft/swin-tiny-patch4-window7-224"
# output_dir = "./swin_slam_results"
# os.makedirs(output_dir, exist_ok=True)

# PIX_TO_WORLD = np.array([10, 10, 1]).reshape((3, 1))
# loopCovariance = np.diag([0.01, 0.01, 0.01])
# odo_cov = np.diag([0.01, 0.01, 0.01])
# threshold = 0.5

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== DATA LOADING ==========
# print("[INFO] Loading loop ground truth...")
# loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# loop_pairs = set(zip(loop_df["img1"], loop_df["img2"]))

# print("[INFO] Loading odometry...")
# poses_array = loadcsv(odom_file, delimiter=",")
# if poses_array.shape[1] != 3:
#     poses_array = poses_array.reshape(-1, 3)
# poses = poses_array.tolist()
# num_frames = len(poses)

# # ========== INIT COMPONENTS ==========
# print("[INFO] Initializing model and matcher...")
# model = SwinLoopModel(model_path=model_path, model_name=model_name, device=device)
# theMatcher = ImageMatcher()
# optimizer = GraphOptimizer(initialID=0, initialPose=np.array(poses[0]).reshape(3, 1))

# for i in range(1, len(poses)):
#     Xodo = (np.array(poses[i]) - np.array(poses[i - 1])).reshape(3, 1)
#     optimizer.add_odometry(i, Xodo, odo_cov)

# loop_stats = []
# print("[INFO] Simulating loop closures...")
# for i in range(num_frames):
#     for j in range(i):
#         # img1_name = f"frame{j:06d}.png"
#         # img2_name = f"frame{i:06d}.png"
#         img1_name = f"IMAGE{j+1:05d}.png"
#         img2_name = f"IMAGE{i+1:05d}.png"
#         doMatch = int((img1_name, img2_name) in loop_pairs or (img2_name, img1_name) in loop_pairs)

#         img1 = Image.open(os.path.join(img_dir, img1_name)).convert("RGB")
#         img2 = Image.open(os.path.join(img_dir, img2_name)).convert("RGB")
#         score = model.predict((img1, img2))[0][0]
#         is_loop = score > threshold

#         predicted = int(is_loop)
#         passed_ransac = 0
#         added = 0

#         if is_loop:
#             img1_np = np.asarray(img1)
#             img2_np = np.asarray(img2)
#             theMatcher.define_images(img1_np, img2_np)
#             theMatcher.estimate()

#             if not theMatcher.hasFailed:
#                 matcherMotion = theMatcher.theMotion.reshape((3, 1)) * PIX_TO_WORLD
#                 optimizer.add_loop(j, i, matcherMotion, loopCovariance)
#                 passed_ransac = 1
#                 added = 1

#         loop_stats.append([j, i, doMatch, predicted, passed_ransac, added])

# # ========== VALIDATE + OPTIMIZE ==========
# print("[INFO] Validating candidate loops...")
# selected_loops = optimizer.validate()
# print(f"[INFO] Loops retained after validation: {len(selected_loops) if selected_loops else 0}")

# print("[INFO] Optimizing pose graph...")
# optimizer.optimize()

# # ========== METRICS & EVALUATION ==========
# loop_array = np.array(loop_stats)
# TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
# FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
# TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
# FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))
# acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)

# print("[INFO] Quality Metrics:")
# print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
# print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, Fallout={fall:.4f}")

# # ========== TRAJECTORY COMPARISON ==========
# print("[INFO] Generating trajectories and evaluating errors...")
# X_est = compose_trajectory(optimizer.poses)
# X_gt = compose_trajectory(poses)

# avg_err = evaluate_trajectory(X_est, np.array(poses).T)
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)

# print(f"[INFO] Average error per traveled distance: {avg_err:.4f}")
# print(f"[INFO] ATE: mean={mean_ate:.4f}, std={std_ate:.4f}")

# # ========== SAVE ==========
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added"]).to_csv(
#     os.path.join(output_dir, "swin_loop_stats.csv"), index=False)

# print("[INFO] SLAM simulation complete. Stats saved.")

# # import os
# # import torch
# # import pandas as pd
# # from PIL import Image
# # from swinloopmodel import SwinLoopModel
# # from graphoptimizer import GraphOptimizer
# # from transform2d import compose_trajectory
# # #from util import read_odometry_file, plot_trajectory
# # from util import loadcsv
# # from util import progress_bar  # optional if you want to add it later


# # # ========== CONFIG ==========
# # csv_path = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# # img_dir = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
# # odom_file = "/Users/sarahvillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
# # model_path = "/Users/sarahvillhauer/Desktop/AI_project/SNNLOOP-main/swin_siamese_best.pth"
# # model_name = "microsoft/swin-tiny-patch4-window7-224"
# # output_dir = "./swin_slam_results"
# # os.makedirs(output_dir, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # threshold = 0.5  # prediction score threshold

# # # ========== LOAD LOOP PAIRS ==========
# # loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# # loop_pairs = set(zip(loop_df["img1"], loop_df["img2"]))

# # # ========== INITIALIZE SWIN LOOP MODEL ==========
# # model = SwinLoopModel(model_path=model_path, model_name=model_name, device=device)

# # # ========== LOOP PREDICTION FUNCTION ==========
# # def predict_loop(img1_name, img2_name):
# #     img1 = Image.open(os.path.join(img_dir, img1_name)).convert("RGB")
# #     img2 = Image.open(os.path.join(img_dir, img2_name)).convert("RGB")
# #     score = model.predict((img1, img2))[0][0]
# #     return score > threshold, score

# # # ========== LOAD ODOMETRY & INIT GRAPH ==========
# # poses_array = loadcsv(odom_file, delimiter=",")
# # poses = poses_array.tolist()  # Convert numpy array to list of [x, y, theta]

# # num_frames = len(poses)

# # optimizer = GraphOptimizer()
# # optimizer.add_odometry_sequence(poses)

# # # ========== SIMULATE LOOP CLOSURES ==========
# # added_loops = 0
# # for i in range(num_frames):
# #     for j in range(i):
# #         img1_name = f"frame{j:06d}.png"
# #         img2_name = f"frame{i:06d}.png"
# #         if (img1_name, img2_name) in loop_pairs or (img2_name, img1_name) in loop_pairs:
# #             is_loop, score = predict_loop(img1_name, img2_name)
# #             if is_loop:
# #                 optimizer.add_loop_edge(j, i)
# #                 added_loops += 1

# # print(f"Added {added_loops} loop closures using Swin predictions.")

# # # ========== OPTIMIZE GRAPH ==========
# # optimizer.optimize()

# # # ========== PLOT TRAJECTORY ==========
# # optimized_traj = compose_trajectory(optimizer.poses)
# # odom_traj = compose_trajectory(poses)

# # plot_trajectory([odom_traj, optimized_traj],
# #                 labels=["Odometry", "Optimized"],
# #                 colors=["red", "green"],
# #                 title="Swin-based SLAM Result",
# #                 save_path=os.path.join(output_dir, "trajectory_comparison.png"))

# # print("SLAM simulation complete. Trajectory plot saved.")
