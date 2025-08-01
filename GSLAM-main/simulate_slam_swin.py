import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from swinloopmodel import SwinLoopModel
from graphoptimizer_gpu import GraphOptimizerGPU
from transform2d_gpu import compose_trajectory
from imagematcher import ImageMatcher
from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error
import matplotlib.pyplot as plt

def align_trajectory_umeyama(est_traj, gt_traj, correct_scale=True):
    est_traj = np.asarray(est_traj)
    gt_traj = np.asarray(gt_traj)
    assert est_traj.shape == gt_traj.shape
    est_centroid = np.mean(est_traj, axis=0)
    gt_centroid = np.mean(gt_traj, axis=0)
    est_centered = est_traj - est_centroid
    gt_centered = gt_traj - gt_centroid
    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    if correct_scale:
        scale = np.trace(np.diag(S)) / np.trace(est_centered.T @ est_centered)
    else:
        scale = 1.0
    t = gt_centroid - scale * R @ est_centroid
    aligned_traj = (scale * R @ est_traj.T).T + t
    return aligned_traj, (scale, R, t)


# ========== CONFIGURATION ==========
csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"
model_name = "microsoft/swin-tiny-patch4-window7-224"
output_dir = "./swin_slam_results"
os.makedirs(output_dir, exist_ok=True)


# Constants with explicit dtype and gradient tracking
PIX_TO_WORLD = torch.tensor([10.0, 10.0, 1.0], dtype=torch.float32, device='cuda').view(3, 1)
loopCovariance = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda'))
odo_cov = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda'))
threshold = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== CHECKPOINT PATHS ==========
loop_stats_path = os.path.join(output_dir, "loop_stats.npy")
feature_cache_path = os.path.join(output_dir, "feature_cache.pt")

# ========== DATA LOADING ==========
print("[INFO] Loading ground truth and odometry...")
loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
loop_pairs = set(zip(loop_df["img1"], loop_df["img2"])).union(set(zip(loop_df["img2"], loop_df["img1"])))

# Load and convert poses to (3,1) tensors with float32
poses_array = loadcsv(odom_file, delimiter=",").reshape(-1, 3).astype(np.float32)
poses = [torch.tensor(p, device=device, dtype=torch.float32).view(3, 1) for p in poses_array]


# Match frames to available images
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith("IMAGE")])
num_frames = min(len(poses), len(img_files))

# Load ground truth positions
# Print basic statistics
gt_positions = np.array([p.cpu().numpy() for p in poses[:num_frames]])
print(f"Ground truth shape: {gt_positions.shape}")
print(f"Min values: {np.min(gt_positions, axis=0)}")
print(f"Max values: {np.max(gt_positions, axis=0)}")
print(f"First 5 poses:\n{gt_positions[:5]}")
print(f"Last 5 poses:\n{gt_positions[-5:]}")


# ========== INITIALIZATION ==========
print("[INFO] Initializing components...")
model = SwinLoopModel(model_path=model_path, model_name=model_name).to(device)
model.eval()
theMatcher = ImageMatcher()

# Initialize optimizer with proper ID tracking
optimizer = GraphOptimizerGPU(
    initialID=0,
    initialPose=poses[0].clone().detach(),
    device=device,
    minLoops=5,
    maxAngularDistance=torch.tensor(np.pi/4, dtype=torch.float32),
    maxMotionDistance=500.0
)

# ========== ODOMETRY CONSTRAINTS ==========
print("\n[PROGRESS] Adding odometry constraints...")
for i in tqdm(range(1, num_frames), desc="Processing odometry"):
    Xodo = (poses[i] - poses[i-1]).view(3, 1)
    optimizer.add_odometry(i, Xodo, odo_cov)

# ========== CHECKPOINT HANDLING ==========
if os.path.exists(loop_stats_path) and os.path.exists(feature_cache_path):
    print("\n[INFO] Loading precomputed features and loop closures...")
    loop_stats = np.load(loop_stats_path, allow_pickle=True).tolist()
    feature_cache = torch.load(feature_cache_path, map_location=device)
    
    # Handle backward compatibility for loop_stats entries
    for entry in loop_stats:
        # Backward compatibility for old entries
        if len(entry) == 6:
            j, i, doMatch, predicted, passed_ransac, added = entry
            matcherMotion_np = None
        else:
            j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np = entry
            
        if added and matcherMotion_np is not None:
            matcherMotion = torch.tensor(matcherMotion_np, device=device, dtype=torch.float32)
            optimizer.add_loop(j, i, matcherMotion, loopCovariance)
else:
    # ========== FEATURE CACHING ==========
    print("\n[PROGRESS] Precomputing image features...")
    feature_cache = {}

    with torch.no_grad():
        for idx in tqdm(range(num_frames), desc="Processing images"):
            img_name = f"IMAGE{idx+1:05d}.png"
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            feature = model.extract_features(img).to(device)
            feature_cache[idx] = feature.view(-1)  # Store as 1D tensor

    # ========== LOOP CLOSURE DETECTION ==========
    print("\n[PROGRESS] Detecting loop closures...")
    loop_stats = []

    for i in tqdm(range(num_frames), desc="Current frame"):
        if i not in feature_cache:
            continue
            
        for j in tqdm(range(i), desc="Previous frames", leave=False):
            if j not in feature_cache:
                continue

            # GPU feature comparison
            with torch.no_grad():
                feat1 = feature_cache[j].view(1, -1)
                feat2 = feature_cache[i].view(1, -1)
                score = model.compare_features(feat1, feat2).item()

            # Ground truth check
            img_i = f"IMAGE{i+1:05d}.png"
            img_j = f"IMAGE{j+1:05d}.png"
            doMatch = int((img_j, img_i) in loop_pairs)

            # Loop closure logic
            is_loop = score > threshold
            predicted = int(is_loop)
            passed_ransac = 0
            added = 0
            matcherMotion_np = None

            if is_loop:
                img1 = np.array(Image.open(os.path.join(img_dir, img_j)))
                img2 = np.array(Image.open(os.path.join(img_dir, img_i)))
                
                theMatcher.define_images(img1, img2)
                theMatcher.estimate()

                if not theMatcher.hasFailed:
                    matcherMotion = torch.tensor(theMatcher.theMotion, 
                                               device=device, dtype=torch.float32).view(3, 1) * PIX_TO_WORLD
                    optimizer.add_loop(j, i, matcherMotion, loopCovariance)
                    passed_ransac = 1
                    added = 1
                    matcherMotion_np = matcherMotion.cpu().numpy()

            loop_stats.append([j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np])

    # Save checkpoint
    np.save(loop_stats_path, np.array(loop_stats, dtype=object))
    torch.save(feature_cache, feature_cache_path)

# ========== POSE GRAPH OPTIMIZATION ==========
print("\n[INFO] Optimizing pose graph...")
optimizer.validate()

# Add gradient verification
print("\n[DEBUG] Gradient checks before optimization:")
print(f"First pose requires grad: {optimizer.poses[0].requires_grad}")
print(f"First pose grad_fn: {optimizer.poses[0].grad_fn}")

optimizer.optimize()

# # ========== EVALUATION METRICS ==========
# print("\n[RESULTS] Final metrics:")
# loop_array = np.array(loop_stats)
# TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
# FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
# TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
# FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))

# ========== EVALUATION METRICS ==========
print("\n[RESULTS] Final metrics:")

# Convert to DataFrame first
loop_df = pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added", "motion"])

# Now convert relevant columns to numpy arrays
TP = np.sum((loop_df["label"] == 1) & (loop_df["predicted"] == 1))
FP = np.sum((loop_df["label"] == 0) & (loop_df["predicted"] == 1))
TN = np.sum((loop_df["label"] == 0) & (loop_df["predicted"] == 0))
FN = np.sum((loop_df["label"] == 1) & (loop_df["predicted"] == 0))


acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

# Trajectory evaluation
# X_est = compose_trajectory([pose.cpu().numpy() for pose in optimizer.get_poses()])
# X_gt = compose_trajectory([p.cpu().numpy() for p in poses[:num_frames]])
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)
# print(f"Absolute Trajectory Error: {mean_ate:.3f} ± {std_ate:.3f}")

# # ========== TRAJECTORY EVALUATION ==========
# # Get optimized poses (already NumPy arrays)
# X_est = compose_trajectory(optimizer.get_poses())

# # Convert ground truth poses from PyTorch tensors to NumPy
# X_gt = compose_trajectory([
#     p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p 
#     for p in poses[:num_frames]
# ])

# # Compute error metrics
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)
# print(f"Absolute Trajectory Error: {mean_ate:.3f} ± {std_ate:.3f}")


# # ========== SAVE OUTPUTS ==========
# print("\n[INFO] Saving results...")
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added", "motion"])\
#   .to_csv(os.path.join(output_dir, "loop_stats.csv"), index=False)

# # Save trajectory plot
# plt.figure(figsize=(10, 10))
# plt.plot(X_est[0], X_est[1], 'b-', label="Optimized Trajectory")
# plt.plot(X_gt[0], X_gt[1], 'k--', label="Ground Truth")
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"))

# # Zoomed in trajectory plot 
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.plot(X_est[0], X_est[1], 'b-', label="Optimized Trajectory")
# ax.plot(X_gt[0], X_gt[1], 'k--', label="Ground Truth")
# ax.legend()
# ax.grid(True)
# ax.set_xlim(-1000, 3000)
# ax.set_ylim(-4000, 0)
# ax.set_xlabel("X axis")
# ax.set_ylabel("Y axis")
# ax.set_aspect('equal', adjustable='box')  # or adjustable='datalim'
# plt.savefig(os.path.join(output_dir, "trajectory_comparison_zoom.png"))
# plt.close()


# print("[SUCCESS] SLAM simulation complete!")

# ========== TRAJECTORY EVALUATION ==========

# Get optimized poses (already NumPy arrays)
X_est = compose_trajectory(optimizer.get_poses())
X_gt = compose_trajectory([
    p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p 
    for p in poses[:num_frames]
])

# Use only the (x, y) part for alignment (if your data is 3xN)
X_est_xy = X_est[:2].T  # Shape (N, 2)
X_gt_xy = X_gt[:2].T    # Shape (N, 2)

# If your trajectories are (3, N), transpose to (N, 3)
if X_est_xy.shape[0] != X_gt_xy.shape[0]:
    min_len = min(X_est_xy.shape[0], X_gt_xy.shape[0])
    X_est_xy = X_est_xy[:min_len]
    X_gt_xy = X_gt_xy[:min_len]

# Align estimated to ground truth
X_est_aligned, (scale, R, t) = align_trajectory_umeyama(X_est_xy, X_gt_xy, correct_scale=True)

# Plot the ground truth
plt.figure(figsize=(10, 10))
plt.plot(gt_positions[:, 0], gt_positions[:, 1], 'r.', markersize=1)
plt.title("Raw Ground Truth Trajectory")
plt.axis('equal')
plt.grid(True)
plt.savefig("raw_groundtruth.png")


# Plot aligned trajectories
plt.figure(figsize=(10, 10))
plt.plot(X_est_aligned[:,0], X_est_aligned[:,1], 'b-', label="Aligned Optimized Trajectory")
plt.plot(X_gt_xy[:,0], X_gt_xy[:,1], 'k--', label="Ground Truth")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "trajectory_comparison_aligned.png"))

# Zoomed in trajectory plot (aligned)
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(X_est_aligned[:,0], X_est_aligned[:,1], 'b-', label="Aligned Optimized Trajectory")
ax.plot(X_gt_xy[:,0], X_gt_xy[:,1], 'k--', label="Ground Truth")
ax.legend()
ax.grid(True)
ax.set_xlim(-1000, 3000)
ax.set_ylim(-4000, 0)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_aspect('equal', adjustable='box')
plt.savefig(os.path.join(output_dir, "trajectory_comparison_zoom_aligned.png"))
plt.close()

# Compute error metrics on aligned trajectory
ate = np.sqrt(np.mean(np.linalg.norm(X_est_aligned - X_gt_xy, axis=1) ** 2))
print(f"Aligned Absolute Trajectory Error: {ate:.3f}")


# import os
# import numpy as np
# import torch
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from swinloopmodel import SwinLoopModel
# from graphoptimizer_gpu import GraphOptimizerGPU
# from transform2d_gpu import compose_trajectory
# from imagematcher import ImageMatcher
# from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error
# import matplotlib.pyplot as plt

# # ========== CONFIGURATION ==========
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
# odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
# model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"
# model_name = "microsoft/swin-tiny-patch4-window7-224"
# output_dir = "./swin_slam_results"
# os.makedirs(output_dir, exist_ok=True)

# # Constants with explicit dtype and gradient tracking
# PIX_TO_WORLD = torch.tensor([10.0, 10.0, 1.0], dtype=torch.float32, device='cuda').view(3, 1)
# loopCovariance = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda'))
# odo_cov = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda'))
# threshold = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== CHECKPOINT PATHS ==========
# loop_stats_path = os.path.join(output_dir, "loop_stats.npy")
# feature_cache_path = os.path.join(output_dir, "feature_cache.pt")

# # ========== DATA LOADING ==========
# print("[INFO] Loading ground truth and odometry...")
# loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# loop_pairs = set(zip(loop_df["img1"], loop_df["img2"])).union(set(zip(loop_df["img2"], loop_df["img1"])))

# # Load and convert poses to (3,1) tensors with gradient tracking
# poses_array = loadcsv(odom_file, delimiter=",").reshape(-1, 3).astype(np.float32)
# poses = [torch.tensor(p, device=device, dtype=torch.float32, requires_grad=True).view(3, 1) for p in poses_array]

# # Match frames to available images
# img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith("IMAGE")])
# num_frames = min(len(poses), len(img_files))

# # ========== INITIALIZATION ==========
# print("[INFO] Initializing components...")
# model = SwinLoopModel(model_path=model_path, model_name=model_name).to(device)
# model.eval()
# theMatcher = ImageMatcher()

# # Initialize optimizer with proper gradient tracking
# optimizer = GraphOptimizerGPU(
#     initialID=0,
#     initialPose=poses[0].clone().requires_grad_(True),  # Remove .detach()
#     device=device,
#     minLoops=5,
#     maxAngularDistance=torch.tensor(np.pi/4, dtype=torch.float32),
#     maxMotionDistance=500.0
# )

# # ========== ODOMETRY CONSTRAINTS ==========
# print("\n[PROGRESS] Adding odometry constraints...")
# for i in tqdm(range(1, num_frames), desc="Processing odometry"):
#     Xodo = (poses[i] - poses[i-1]).view(3, 1).requires_grad_(True)  # Add gradient tracking
#     optimizer.add_odometry(i, Xodo, odo_cov)

# # ========== CHECKPOINT HANDLING ==========
# if os.path.exists(loop_stats_path) and os.path.exists(feature_cache_path):
#     print("\n[INFO] Loading precomputed features and loop closures...")
#     loop_stats = np.load(loop_stats_path, allow_pickle=True).tolist()
#     feature_cache = torch.load(feature_cache_path, map_location=device)
    
#     # Reconstruct loop closures with gradient tracking
#     for entry in loop_stats:
#         if len(entry) == 6:
#             j, i, doMatch, predicted, passed_ransac, added = entry
#             matcherMotion_np = None
#         else:
#             j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np = entry

#         if added and matcherMotion_np is not None:
#             # Rebuild tensor with gradient tracking
#             matcherMotion = torch.tensor(matcherMotion_np, 
#                                        device=device, 
#                                        dtype=torch.float32,
#                                        requires_grad=True)
#             optimizer.add_loop(j, i, matcherMotion, loopCovariance)
# else:
#     # ========== FEATURE CACHING ==========
#     print("\n[PROGRESS] Precomputing image features...")
#     feature_cache = {}

#     with torch.no_grad():
#         for idx in tqdm(range(num_frames), desc="Processing images"):
#             img_name = f"IMAGE{idx+1:05d}.png"
#             img_path = os.path.join(img_dir, img_name)
            
#             if not os.path.exists(img_path):
#                 continue
                
#             img = Image.open(img_path).convert("RGB")
#             feature = model.extract_features(img).to(device)
#             feature_cache[idx] = feature.view(-1).requires_grad_(True)  # Track gradients

#     # ========== LOOP CLOSURE DETECTION ==========
#     print("\n[PROGRESS] Detecting loop closures...")
#     loop_stats = []

#     for i in tqdm(range(num_frames), desc="Current frame"):
#         if i not in feature_cache:
#             continue
            
#         for j in tqdm(range(i), desc="Previous frames", leave=False):
#             if j not in feature_cache:
#                 continue

#             with torch.no_grad():
#                 feat1 = feature_cache[j].view(1, -1)
#                 feat2 = feature_cache[i].view(1, -1)
#                 score = model.compare_features(feat1, feat2).item()

#             img_i = f"IMAGE{i+1:05d}.png"
#             img_j = f"IMAGE{j+1:05d}.png"
#             doMatch = int((img_j, img_i) in loop_pairs)

#             is_loop = score > threshold
#             predicted = int(is_loop)
#             passed_ransac = 0
#             added = 0
#             matcherMotion_np = None

#             if is_loop:
#                 img1 = np.array(Image.open(os.path.join(img_dir, img_j)))
#                 img2 = np.array(Image.open(os.path.join(img_dir, img_i)))
                
#                 theMatcher.define_images(img1, img2)
#                 theMatcher.estimate()

#                 if not theMatcher.hasFailed:
#                     matcherMotion = torch.tensor(theMatcher.theMotion, 
#                                                device=device, 
#                                                dtype=torch.float32,
#                                                requires_grad=True).view(3, 1) * PIX_TO_WORLD
#                     optimizer.add_loop(j, i, matcherMotion, loopCovariance)
#                     passed_ransac = 1
#                     added = 1
#                     matcherMotion_np = matcherMotion.detach().cpu().numpy()

#             loop_stats.append([j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np])

#     # Save checkpoint with gradient info
#     np.save(loop_stats_path, np.array(loop_stats, dtype=object))
#     torch.save(feature_cache, feature_cache_path)

# # ========== POSE GRAPH OPTIMIZATION ==========
# print("\n[INFO] Optimizing pose graph...")
# optimizer.validate()

# # Add gradient verification
# print("\n[DEBUG] Gradient checks before optimization:")
# print(f"First pose requires grad: {optimizer.poses[0].requires_grad}")
# print(f"First pose grad_fn: {optimizer.poses[0].grad_fn}")

# optimizer.optimize()

# # ========== EVALUATION METRICS ==========
# print("\n[RESULTS] Final metrics:")
# loop_array = np.array(loop_stats)
# TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
# FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
# TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
# FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))

# acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)
# print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

# # Trajectory evaluation
# X_est = compose_trajectory([pose.detach().cpu().numpy() for pose in optimizer.get_poses()])
# X_gt = compose_trajectory([p.detach().cpu().numpy() for p in poses[:num_frames]])
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)
# print(f"Absolute Trajectory Error: {mean_ate:.3f} ± {std_ate:.3f}")

# # ========== SAVE OUTPUTS ==========
# print("\n[INFO] Saving results...")
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added", "motion"])\
#   .to_csv(os.path.join(output_dir, "loop_stats.csv"), index=False)

# # Save trajectory plot
# plt.figure(figsize=(10, 10))
# plt.plot(X_est[0], X_est[1], 'b-', label="Optimized Trajectory")
# plt.plot(X_gt[0], X_gt[1], 'k--', label="Ground Truth")
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"))

# print("[SUCCESS] SLAM simulation complete!")



# import os
# import numpy as np
# import torch
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from swinloopmodel import SwinLoopModel
# from graphoptimizer_gpu import GraphOptimizerGPU
# from transform2d_gpu import compose_trajectory
# from imagematcher import ImageMatcher
# from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error
# import matplotlib.pyplot as plt

# # ========== CONFIGURATION ==========
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
# odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
# model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"
# model_name = "microsoft/swin-tiny-patch4-window7-224"
# output_dir = "./swin_slam_results"
# os.makedirs(output_dir, exist_ok=True)

# # Constants with explicit dtype
# PIX_TO_WORLD = torch.tensor([10.0, 10.0, 1.0], dtype=torch.float32, device='cuda').view(3, 1)
# loopCovariance = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda'))
# odo_cov = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device='cuda'))
# threshold = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== CHECKPOINT PATHS ==========
# loop_stats_path = os.path.join(output_dir, "loop_stats.npy")
# feature_cache_path = os.path.join(output_dir, "feature_cache.pt")

# # ========== DATA LOADING ==========
# print("[INFO] Loading ground truth and odometry...")
# loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# loop_pairs = set(zip(loop_df["img1"], loop_df["img2"])).union(set(zip(loop_df["img2"], loop_df["img1"])))

# # Load and convert poses to (3,1) tensors with float32
# poses_array = loadcsv(odom_file, delimiter=",").reshape(-1, 3).astype(np.float32)
# poses = [torch.tensor(p, device=device, dtype=torch.float32).view(3, 1) for p in poses_array]

# # Match frames to available images
# img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith("IMAGE")])
# num_frames = min(len(poses), len(img_files))

# # ========== INITIALIZATION ==========
# print("[INFO] Initializing components...")
# model = SwinLoopModel(model_path=model_path, model_name=model_name).to(device)
# model.eval()
# theMatcher = ImageMatcher()

# # Initialize optimizer with proper ID tracking
# optimizer = GraphOptimizerGPU(
#     initialID=0,
#     initialPose=poses[0].clone().detach(),
#     device=device,
#     minLoops=5,
#     maxAngularDistance=torch.tensor(np.pi/4, dtype=torch.float32),
#     maxMotionDistance=500.0
# )

# # ========== ODOMETRY CONSTRAINTS ==========
# print("\n[PROGRESS] Adding odometry constraints...")
# for i in tqdm(range(1, num_frames), desc="Processing odometry"):
#     Xodo = (poses[i] - poses[i-1]).view(3, 1)
#     optimizer.add_odometry(i, Xodo, odo_cov)

# # ========== CHECKPOINT HANDLING ==========
# if os.path.exists(loop_stats_path) and os.path.exists(feature_cache_path):
#     print("\n[INFO] Loading precomputed features and loop closures...")
#     loop_stats = np.load(loop_stats_path, allow_pickle=True).tolist()
#     feature_cache = torch.load(feature_cache_path, map_location=device)
    
#     # ---- MODIFY CACHED MOTIONS HERE IF NEEDED ----
#     for entry in loop_stats:
#         # Backward compatibility for old (6) and new (7) element entries
#         if len(entry) == 6:
#             j, i, doMatch, predicted, passed_ransac, added = entry
#             matcherMotion_np = None
#         else:
#             j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np = entry

#         # Example: scale all motions by 0.95 (remove if not needed)
#         if added and matcherMotion_np is not None:
#             matcherMotion_np = matcherMotion_np * 0.95
#             entry[6] = matcherMotion_np  # update in cache

#             matcherMotion = torch.tensor(matcherMotion_np, device=device, dtype=torch.float32)
#             optimizer.add_loop(j, i, matcherMotion, loopCovariance)
# # -------------------------------------------------
# else:
#     # ========== FEATURE CACHING ==========
#     print("\n[PROGRESS] Precomputing image features...")
#     feature_cache = {}

#     with torch.no_grad():
#         for idx in tqdm(range(num_frames), desc="Processing images"):
#             img_name = f"IMAGE{idx+1:05d}.png"
#             img_path = os.path.join(img_dir, img_name)
            
#             if not os.path.exists(img_path):
#                 continue
                
#             img = Image.open(img_path).convert("RGB")
#             feature = model.extract_features(img).to(device)
#             feature_cache[idx] = feature.view(-1)

#     # ========== LOOP CLOSURE DETECTION ==========
#     print("\n[PROGRESS] Detecting loop closures...")
#     loop_stats = []

#     for i in tqdm(range(num_frames), desc="Current frame"):
#         if i not in feature_cache:
#             continue
            
#         for j in tqdm(range(i), desc="Previous frames", leave=False):
#             if j not in feature_cache:
#                 continue

#             # GPU feature comparison
#             with torch.no_grad():
#                 feat1 = feature_cache[j].view(1, -1)
#                 feat2 = feature_cache[i].view(1, -1)
#                 score = model.compare_features(feat1, feat2).item()

#             # Ground truth check
#             img_i = f"IMAGE{i+1:05d}.png"
#             img_j = f"IMAGE{j+1:05d}.png"
#             doMatch = int((img_j, img_i) in loop_pairs)

#             # Loop closure logic
#             is_loop = score > threshold
#             predicted = int(is_loop)
#             passed_ransac = 0
#             added = 0
#             matcherMotion_np = None

#             if is_loop:
#                 img1 = np.array(Image.open(os.path.join(img_dir, img_j)))
#                 img2 = np.array(Image.open(os.path.join(img_dir, img_i)))
                
#                 theMatcher.define_images(img1, img2)
#                 theMatcher.estimate()

#                 if not theMatcher.hasFailed:
#                     matcherMotion = torch.tensor(theMatcher.theMotion, 
#                                                device=device, dtype=torch.float32).view(3, 1) * PIX_TO_WORLD
#                     optimizer.add_loop(j, i, matcherMotion, loopCovariance)
#                     passed_ransac = 1
#                     added = 1
#                     matcherMotion_np = matcherMotion.cpu().numpy()

#             # Always store 7 elements for future compatibility
#             loop_stats.append([j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np])

#     # Save checkpoint
#     np.save(loop_stats_path, np.array(loop_stats, dtype=object))
#     torch.save(feature_cache, feature_cache_path)

# # ========== POSE GRAPH OPTIMIZATION ==========
# print("\n[INFO] Optimizing pose graph...")
# optimizer.validate()
# optimizer.optimize()

# # ========== EVALUATION METRICS ==========
# print("\n[RESULTS] Final metrics:")
# loop_array = np.array(loop_stats)
# TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
# FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
# TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
# FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))

# acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)
# print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

# # Trajectory evaluation
# X_est = compose_trajectory([pose.cpu().numpy() for pose in optimizer.get_poses()])
# X_gt = compose_trajectory([p.cpu().numpy() for p in poses[:num_frames]])
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)
# print(f"Absolute Trajectory Error: {mean_ate:.3f} ± {std_ate:.3f}")

# # ========== SAVE OUTPUTS ==========
# print("\n[INFO] Saving results...")
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added", "motion"])\
#   .to_csv(os.path.join(output_dir, "loop_stats.csv"), index=False)

# # Save trajectory plot
# plt.figure(figsize=(10, 10))
# plt.plot(X_est[0], X_est[1], 'b-', label="Optimized Trajectory")
# plt.plot(X_gt[0], X_gt[1], 'k--', label="Ground Truth")
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"))

# print("[SUCCESS] SLAM simulation complete!")

