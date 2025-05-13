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

# ========== CONFIGURATION ==========
csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
odom_file = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/ODOM.csv"
model_path = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"
model_name = "microsoft/swin-tiny-patch4-window7-224"
output_dir = "./swin_slam_results"
os.makedirs(output_dir, exist_ok=True)

# Constants with explicit dtype
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
        # Check if entry has old format (6 elements)
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
            feature_cache[idx] = feature.view(-1)

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

            # Always store 7 elements for future compatibility
            loop_stats.append([j, i, doMatch, predicted, passed_ransac, added, matcherMotion_np])

    # Save checkpoint
    np.save(loop_stats_path, np.array(loop_stats, dtype=object))
    torch.save(feature_cache, feature_cache_path)

# ========== POSE GRAPH OPTIMIZATION ==========
print("\n[INFO] Optimizing pose graph...")
optimizer.validate()
optimizer.optimize()

# ========== EVALUATION METRICS ==========
print("\n[RESULTS] Final metrics:")
loop_array = np.array(loop_stats)
TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))

acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

# Trajectory evaluation
X_est = compose_trajectory([pose.cpu().numpy() for pose in optimizer.get_poses()])
X_gt = compose_trajectory([p.cpu().numpy() for p in poses[:num_frames]])
ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)
print(f"Absolute Trajectory Error: {mean_ate:.3f} ± {std_ate:.3f}")

# ========== SAVE OUTPUTS ==========
print("\n[INFO] Saving results...")
pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added", "motion"])\
  .to_csv(os.path.join(output_dir, "loop_stats.csv"), index=False)

# Save trajectory plot
plt.figure(figsize=(10, 10))
plt.plot(X_est[0], X_est[1], 'b-', label="Optimized Trajectory")
plt.plot(X_gt[0], X_gt[1], 'k--', label="Ground Truth")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"))

print("[SUCCESS] SLAM simulation complete!")



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

# # Constants
# PIX_TO_WORLD = torch.tensor([10.0, 10.0, 1.0], device='cuda').view(3, 1)
# loopCovariance = torch.diag(torch.tensor([0.01, 0.01, 0.01], device='cuda'))
# odo_cov = torch.diag(torch.tensor([0.01, 0.01, 0.01], device='cuda'))
# threshold = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== CHECKPOINT PATHS ==========
# loop_stats_path = os.path.join(output_dir, "loop_stats.npy")
# feature_cache_path = os.path.join(output_dir, "feature_cache.pt")

# # ========== DATA LOADING ==========
# print("[INFO] Loading ground truth and odometry...")
# loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# loop_pairs = set(zip(loop_df["img1"], loop_df["img2"])).union(set(zip(loop_df["img2"], loop_df["img1"])))

# # Load and convert poses to (3,1) tensors
# poses_array = loadcsv(odom_file, delimiter=",").reshape(-1, 3)
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
#     maxAngularDistance=torch.tensor(np.pi/4),
#     maxMotionDistance=500.0
# )

# # ========== ODOMETRY CONSTRAINTS ==========
# print("\n[PROGRESS] Adding odometry constraints...")
# for i in tqdm(range(1, num_frames), desc="Processing odometry"):
#     Xodo = (poses[i] - poses[i-1]).view(3, 1)
#     optimizer.add_odometry(i, Xodo, odo_cov)

# # ========== CHECKPOINT LOADING ==========
# if os.path.exists(loop_stats_path) and os.path.exists(feature_cache_path):
#     print("\n[INFO] Loading precomputed features and loop closures...")
#     loop_stats = np.load(loop_stats_path, allow_pickle=True).tolist()
#     feature_cache = torch.load(feature_cache_path, map_location=device)
    
#     # Add precomputed loop closures to optimizer
#     for entry in loop_stats:
#         if entry[5]:  # If 'added' is True
#             j, i, _, _, _, _ = entry
#             # Note: This assumes loopCovariance is constant and matcherMotion can be recalculated
#             # For full checkpointing, you'd need to save actual motion parameters
#             optimizer.add_loop(j, i, torch.zeros(3,1, device=device), loopCovariance)
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
#             feature_cache[idx] = feature.view(-1)  # Store as 1D tensor

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

#             if is_loop:
#                 img1 = np.array(Image.open(os.path.join(img_dir, img_j)))
#                 img2 = np.array(Image.open(os.path.join(img_dir, img_i)))
                
#                 theMatcher.define_images(img1, img2)
#                 theMatcher.estimate()

#                 if not theMatcher.hasFailed:
#                     matcherMotion = torch.tensor(theMatcher.theMotion, 
#                                                device=device).view(3, 1) * PIX_TO_WORLD
#                     optimizer.add_loop(j, i, matcherMotion, loopCovariance)
#                     passed_ransac = 1
#                     added = 1

#             loop_stats.append([j, i, doMatch, predicted, passed_ransac, added])

#     # Save checkpoint
#     np.save(loop_stats_path, np.array(loop_stats))
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
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added"])\
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

# # Enable GPU acceleration settings FIRST
# torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 tensor cores
# torch.set_float32_matmul_precision('high')    # Balance speed/accuracy for SLAM

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

# # Constants
# PIX_TO_WORLD = torch.tensor([10.0, 10.0, 1.0], device='cuda').view(3, 1)
# loopCovariance = torch.diag(torch.tensor([0.01, 0.01, 0.01], device='cuda'))
# odo_cov = torch.diag(torch.tensor([0.01, 0.01, 0.01], device='cuda'))
# threshold = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== DATA LOADING ==========
# print("[INFO] Loading ground truth and odometry...")
# loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# loop_pairs = set(zip(loop_df["img1"], loop_df["img2"])).union(set(zip(loop_df["img2"], loop_df["img1"])))

# # Load and convert poses to (3,1) tensors
# poses_array = loadcsv(odom_file, delimiter=",").reshape(-1, 3)
# poses = [torch.tensor(p, device=device, dtype=torch.float32).view(3, 1) for p in poses_array]

# # Match frames to available images
# img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith("IMAGE")])
# num_frames = min(len(poses), len(img_files))

# # ========== INITIALIZATION ==========
# print("[INFO] Initializing components...")
# model = SwinLoopModel(model_path=model_path, model_name=model_name).to(device)
# model.eval()
# theMatcher = ImageMatcher()

# # Initialize optimizer with proper (3,1) tensor
# initial_pose = poses[0].clone().detach()
# #optimizer = GraphOptimizerGPU(initialPose=initial_pose, device=device)
# # Initialize with explicit ID tracking
# optimizer = GraphOptimizerGPU(
#     initialID=0,  # Explicit initial ID
#     initialPose=torch.tensor(poses[0], device=device).view(3, 1),
#     device=device
# )


# # ========== ODOMETRY CONSTRAINTS ==========
# print("\n[PROGRESS] Adding odometry constraints...")
# for i in tqdm(range(1, num_frames), desc="Processing odometry"):
#     Xodo = (poses[i] - poses[i-1]).view(3, 1)
#     optimizer.add_odometry(i, Xodo, odo_cov)

# # ========== FEATURE CACHING ==========
# print("\n[PROGRESS] Precomputing image features...")
# feature_cache = {}

# with torch.no_grad():
#     for idx in tqdm(range(num_frames), desc="Processing images"):
#         img_name = f"IMAGE{idx+1:05d}.png"
#         img_path = os.path.join(img_dir, img_name)
        
#         if not os.path.exists(img_path):
#             continue
            
#         img = Image.open(img_path).convert("RGB")
#         feature = model.extract_features(img).to(device)  # Keep on GPU
#         feature_cache[idx] = feature.view(-1)  # Store as 1D tensor

# # ========== GPU-ACCELERATED LOOP CLOSURE DETECTION ==========
# print("\n[PROGRESS] Detecting loop closures...")
# loop_stats = []

# for i in tqdm(range(num_frames), desc="Current frame"):
#     if i not in feature_cache:
#         continue
        
#     for j in tqdm(range(i), desc="Previous frames", leave=False):
#         if j not in feature_cache:
#             continue

#         # GPU feature comparison
#         with torch.no_grad():
#             feat1 = feature_cache[j].view(1, -1)
#             feat2 = feature_cache[i].view(1, -1)
#             score = model.compare_features(feat1, feat2).item()

#         # Ground truth check
#         img_i = f"IMAGE{i+1:05d}.png"
#         img_j = f"IMAGE{j+1:05d}.png"
#         doMatch = int((img_j, img_i) in loop_pairs)

#         # Loop closure logic
#         is_loop = score > threshold
#         predicted = int(is_loop)
#         passed_ransac = 0
#         added = 0

#         if is_loop:
#             # Load images only for positive detections
#             img1 = np.array(Image.open(os.path.join(img_dir, img_j)))
#             img2 = np.array(Image.open(os.path.join(img_dir, img_i)))
            
#             theMatcher.define_images(img1, img2)
#             theMatcher.estimate()

#             if not theMatcher.hasFailed:
#                 matcherMotion = torch.tensor(theMatcher.theMotion, device=device).view(3, 1) * PIX_TO_WORLD
#                 optimizer.add_loop(j, i, matcherMotion, loopCovariance)
#                 passed_ransac = 1
#                 added = 1

#         loop_stats.append([j, i, doMatch, predicted, passed_ransac, added])

# # ========== GPU-ACCELERATED POSE GRAPH OPTIMIZATION ==========
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
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added"])\
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
# from graphoptimizer import GraphOptimizer
# from transform2d import compose_trajectory
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

# # Constants
# PIX_TO_WORLD = np.array([10, 10, 1]).reshape((3, 1))
# loopCovariance = np.diag([0.01, 0.01, 0.01])
# odo_cov = np.diag([0.01, 0.01, 0.01])
# threshold = 0.5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== DATA LOADING ==========
# print("[INFO] Loading ground truth and odometry...")
# loop_df = pd.read_csv(csv_path).rename(columns={"first": "img1", "second": "img2", "match": "label"})
# loop_pairs = set(zip(loop_df["img1"], loop_df["img2"])).union(set(zip(loop_df["img2"], loop_df["img1"])))

# # Load and align pose data
# poses_array = loadcsv(odom_file, delimiter=",").reshape(-1, 3)
# poses = poses_array.tolist()

# # Match frames to available images
# img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith("IMAGE")])
# num_frames = min(len(poses), len(img_files))

# # ========== INITIALIZATION ==========
# print("[INFO] Initializing components...")
# model = SwinLoopModel(model_path=model_path, model_name=model_name).to(device)
# model.eval()
# theMatcher = ImageMatcher()
# optimizer = GraphOptimizer(initialID=0, initialPose=np.array(poses[0]).reshape(3, 1))

# # Initialize odometry constraints
# for i in range(1, num_frames):
#     Xodo = (np.array(poses[i]) - np.array(poses[i-1])).reshape(3, 1)
#     optimizer.add_odometry(i, Xodo, odo_cov)

# # ========== FEATURE CACHING ==========
# print("\n[PROGRESS] Precomputing image features...")
# feature_cache = {}

# with torch.no_grad():
#     for idx in tqdm(range(num_frames), desc="Processing images"):
#         img_name = f"IMAGE{idx+1:05d}.png"
#         img_path = os.path.join(img_dir, img_name)
        
#         if not os.path.exists(img_path):
#             continue
            
#         img = Image.open(img_path).convert("RGB")
#         feature = model.extract_features(img).cpu()  # Keep features on CPU
#         feature_cache[idx] = feature

# # ========== OPTIMIZED LOOP CLOSURE DETECTION ==========
# print("\n[PROGRESS] Detecting loop closures...")
# loop_stats = []

# for i in tqdm(range(num_frames), desc="Current frame"):
#     if i not in feature_cache:
#         continue
        
#     for j in tqdm(range(i), desc="Previous frames", leave=False):
#         if j not in feature_cache:
#             continue

#         # Feature comparison
#         with torch.no_grad():
#             score = model.compare_features(feature_cache[j].to(device),
#                               feature_cache[i].to(device)).item()

#             # score = model.compare_features(
#             #     feature_cache[j].to(device),
#             #     feature_cache[i].to(device)
#             # ).item()

#         # Ground truth check
#         img_i = f"IMAGE{i+1:05d}.png"
#         img_j = f"IMAGE{j+1:05d}.png"
#         doMatch = int((img_j, img_i) in loop_pairs)

#         # Loop closure logic
#         is_loop = score > threshold
#         predicted = int(is_loop)
#         passed_ransac = 0
#         added = 0

#         if is_loop:
#             # Only load images for positive detections
#             img1 = np.array(Image.open(os.path.join(img_dir, img_j)))
#             img2 = np.array(Image.open(os.path.join(img_dir, img_i)))
            
#             theMatcher.define_images(img1, img2)
#             theMatcher.estimate()

#             if not theMatcher.hasFailed:
#                 matcherMotion = theMatcher.theMotion.reshape((3, 1)) * PIX_TO_WORLD
#                 optimizer.add_loop(j, i, matcherMotion, loopCovariance)
#                 passed_ransac = 1
#                 added = 1

#         loop_stats.append([j, i, doMatch, predicted, passed_ransac, added])

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
# X_est = compose_trajectory(optimizer.poses)
# X_gt = compose_trajectory(poses[:num_frames])
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)
# print(f"Absolute Trajectory Error: {mean_ate:.3f} ± {std_ate:.3f}")

# # ========== SAVE OUTPUTS ==========
# print("\n[INFO] Saving results...")
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added"])\
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
# from graphoptimizer import GraphOptimizer
# from transform2d import compose_trajectory
# from imagematcher import ImageMatcher
# from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error
# import matplotlib.pyplot as plt

# # ========== CONFIGURATION ==========
# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
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

# # Match number of frames to available images
# img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith("IMAGE")])
# num_images = len(img_files)
# num_frames = min(len(poses), num_images)

# # ========== INIT COMPONENTS ==========
# print("[INFO] Initializing model and matcher...")
# model = SwinLoopModel(model_path=model_path, model_name=model_name, device=device)
# theMatcher = ImageMatcher()
# optimizer = GraphOptimizer(initialID=0, initialPose=np.array(poses[0]).reshape(3, 1))

# for i in range(1, num_frames):
#     Xodo = (np.array(poses[i]) - np.array(poses[i - 1])).reshape(3, 1)
#     optimizer.add_odometry(i, Xodo, odo_cov)

# loop_stats = []

# # ========== LOOP SIMULATION ==========
# print("[INFO] Simulating loop closures...")
# for i in tqdm(range(num_frames), desc="Loop closure simulation"):
#     for j in range(i):
#         img1_name = f"IMAGE{j+1:05d}.png"
#         img2_name = f"IMAGE{i+1:05d}.png"
#         img1_path = os.path.join(img_dir, img1_name)
#         img2_path = os.path.join(img_dir, img2_name)

#         # Skip if either image does not exist
#         if not os.path.exists(img1_path) or not os.path.exists(img2_path):
#             continue

#         doMatch = int((img1_name, img2_name) in loop_pairs or (img2_name, img1_name) in loop_pairs)

#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")
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

# # ========== METRICS ==========
# loop_array = np.array(loop_stats)
# TP = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 1))
# FP = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 1))
# TN = np.sum((loop_array[:, 2] == 0) & (loop_array[:, 3] == 0))
# FN = np.sum((loop_array[:, 2] == 1) & (loop_array[:, 3] == 0))
# acc, prec, rec, fall = compute_quality_metrics(TP, FP, TN, FN)

# print("[INFO] Quality Metrics:")
# print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
# print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, Fallout={fall:.4f}")

# # ========== TRAJECTORY ERROR ==========
# print("[INFO] Evaluating trajectory...")
# X_est = compose_trajectory(optimizer.poses)
# X_gt = compose_trajectory(poses)

# avg_err = evaluate_trajectory(X_est, np.array(poses).T)
# ate, mean_ate, std_ate = compute_absolute_trajectory_error(X_est, X_gt)

# print(f"[INFO] Average error per traveled distance: {avg_err:.4f}")
# print(f"[INFO] ATE: mean={mean_ate:.4f}, std={std_ate:.4f}")

# # ========== SAVE STATS ==========
# pd.DataFrame(loop_stats, columns=["i", "j", "label", "predicted", "passed_ransac", "added"]).to_csv(
#     os.path.join(output_dir, "swin_loop_stats.csv"), index=False)

# with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
#     f.write(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}\n")
#     f.write(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, Fallout={fall:.4f}\n")
#     f.write(f"Avg Error per Distance: {avg_err:.4f}\n")
#     f.write(f"ATE Mean={mean_ate:.4f}, Std={std_ate:.4f}\n")

# print("[INFO] SLAM simulation complete. Stats saved.")

# # ========== PLOT TRAJECTORY ==========
# X_est = np.array(optimizer.poses).squeeze().T
# plt.figure(figsize=(10, 10))
# plt.plot(X_est[0], X_est[1], 'k-', label="Optimized Trajectory")

# # Plot loop closures
# for i, j, label, pred, passed, added in loop_stats:
#     if added:
#         x1, y1 = optimizer.poses[i][:2]
#         x2, y2 = optimizer.poses[j][:2]
#         plt.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5)

# # Optional: arrows
# for i, j, label, pred, passed, added in loop_stats:
#     if added:
#         x1, y1 = optimizer.poses[i][:2]
#         x2, y2 = optimizer.poses[j][:2]
#         plt.arrow(x1, y1, x2 - x1, y2 - y1, color='red', head_width=20, alpha=0.5)

# plt.axis('equal')
# plt.grid(True)
# plt.title("Swin-based SLAM Loop Closures")
# plt.legend()
# plt.savefig(os.path.join(output_dir, "swin_slam_plot.png"))
# plt.show()





# import os
# import numpy as np
# import torch
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm  # Added for progress bar
# from swinloopmodel import SwinLoopModel
# from graphoptimizer import GraphOptimizer
# from transform2d import compose_trajectory
# from imagematcher import ImageMatcher
# from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error

# # ========== CONFIGURATION ==========

# csv_path = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/OVERLAP_PAIRS.csv"
# img_dir = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET/IMAGES"
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

# # ========== LOOP SIMULATION WITH PROGRESS ==========
# print("[INFO] Simulating loop closures...")
# for i in tqdm(range(num_frames), desc="Loop closure simulation"):
#     for j in range(i):
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


# import matplotlib.pyplot as plt

# # Plot trajectory
# X_est = np.array(optimizer.poses).squeeze().T  # Estimated poses
# plt.figure(figsize=(10, 10))
# plt.plot(X_est[0], X_est[1], 'k-', label="Optimized Trajectory")

# # Plot loop closures (in blue)
# for i, j, label, pred, passed, added in loop_stats:
#     if added:
#         x1, y1 = optimizer.poses[i][:2]
#         x2, y2 = optimizer.poses[j][:2]
#         plt.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5)

# # Optionally add loop closure arrows (in red)
# for i, j, label, pred, passed, added in loop_stats:
#     if added:
#         x1, y1 = optimizer.poses[i][:2]
#         x2, y2 = optimizer.poses[j][:2]
#         plt.arrow(x1, y1, x2 - x1, y2 - y1, color='red', head_width=20, alpha=0.5)

# plt.axis('equal')
# plt.grid(True)
# plt.title("Swin-based SLAM Loop Closures")
# plt.legend()
# plt.savefig(os.path.join(output_dir, "swin_slam_plot.png"))
# plt.show()



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
