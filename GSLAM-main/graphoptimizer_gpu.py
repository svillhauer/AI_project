import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from swinloopmodel import SwinLoopModel
from transform2d_gpu import compose_references, invert_reference, normalize_angle, compose_trajectory
from imagematcher import ImageMatcher
from util import loadcsv, compute_quality_metrics, evaluate_trajectory, compute_absolute_trajectory_error
import matplotlib.pyplot as plt

# ========================================================================
# GraphOptimizerGPU Class (Corrected Version)
# ========================================================================
class GraphOptimizerGPU:
    def __init__(self, initialID, initialPose, device='cuda',
                 minLoops=5, maxAngularDistance=np.pi/4, 
                 maxMotionDistance=500, errorThreshold=500, maxSameID=2):
        self.device = device
        
        # Initialize with ParameterList for leaf tensors
        self.poses = torch.nn.ParameterList([
            torch.nn.Parameter(initialPose.clone().detach().to(device), requires_grad=True)
        ])
        self.ids = [initialID]
        self.edges = []
        self.candidate_edges = []
        
        # Configuration parameters
        self.minLoops = minLoops
        self.maxAngularDistance = torch.tensor(np.pi/4, device=device, dtype=torch.float32)
        self.maxMotionDistance = maxMotionDistance
        self.errorThreshold = errorThreshold
        self.maxSameID = maxSameID

    def add_odometry(self, newPoseID, Xodo, Podo):
        # Convert inputs to tensors (maintain leaf status)
        Xodo = Xodo if isinstance(Xodo, torch.Tensor) else torch.tensor(Xodo, device=self.device, dtype=torch.float32)
        Podo = Podo if isinstance(Podo, torch.Tensor) else torch.tensor(Podo, device=self.device, dtype=torch.float32)
        
        # Create new pose as leaf parameter
        new_pose = torch.nn.Parameter(
            compose_references(self.poses[-1], Xodo.view(3, 1)),
            requires_grad=True
        )
        self.poses.append(new_pose)
        self.ids.append(newPoseID)
        
        # Add edge with inverse covariance
        self.edges.append({
            'type': 'odom',
            'from': len(self.poses)-2,
            'to': len(self.poses)-1,
            'measurement': Xodo.detach(),
            'information': torch.inverse(Podo.view(3, 3))
        })

    def add_loop(self, fromID, toID, Xloop, Ploop):
        # Convert to tensors with gradient tracking
        Xloop = Xloop if isinstance(Xloop, torch.Tensor) else torch.tensor(Xloop, device=self.device, dtype=torch.float32)
        Ploop = Ploop if isinstance(Ploop, torch.Tensor) else torch.tensor(Ploop, device=self.device, dtype=torch.float32)
        
        # Find indices for the IDs
        from_idx = self.ids.index(fromID)
        to_idx = self.ids.index(toID)
        
        # Add candidate loop closure
        self.candidate_edges.append({
            'from': from_idx,
            'to': to_idx,
            'measurement': Xloop.view(3, 1),
            'information': torch.inverse(Ploop.view(3, 3))
        })

    def validate(self):
        if len(self.candidate_edges) < self.minLoops:
            return []

        valid_edges = []
        for edge in self.candidate_edges:
            from_pose = self.poses[edge['from']]
            to_pose = self.poses[edge['to']]
            
            # Compute relative pose with gradient tracking
            rel_pose = compose_references(invert_reference(from_pose), to_pose)
            delta = rel_pose - edge['measurement']
            
            # Check thresholds
            angular_diff = torch.abs(normalize_angle(delta[2]))
            distance_diff = torch.norm(delta[:2])
            
            if angular_diff < self.maxAngularDistance and distance_diff < self.maxMotionDistance:
                valid_edges.append(edge)
                
        return valid_edges

    def optimize(self):
        # Directly optimize the ParameterList
        optimizer = torch.optim.LBFGS(self.poses, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            total_error = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            for edge in self.edges + self.validate():
                from_idx = edge['from']
                to_idx = edge['to']
                from_pose = self.poses[from_idx]
                to_pose = self.poses[to_idx]
                
                # Compute predicted measurement
                pred = compose_references(invert_reference(from_pose), to_pose)
                error = pred - edge['measurement']
                error[2] = normalize_angle(error[2])
                
                # Calculate error term
                error_term = torch.matmul(error.T, torch.matmul(edge['information'], error))
                total_error += error_term.squeeze()
            
            total_error.backward()
            return total_error
        
        optimizer.step(closure)

    def get_poses(self):
        return [pose.detach().cpu().numpy() for pose in self.poses]


# import torch
# import numpy as np
# from transform2d_gpu import compose_references, invert_reference, normalize_angle



# class GraphOptimizerGPU:
#     def __init__(self, initialID, initialPose, device='cuda',
#                  minLoops=5, maxAngularDistance=np.pi/4, 
#                  maxMotionDistance=500, errorThreshold=500, maxSameID=2):
#         self.device = device
        
#         # Initialize with gradient tracking
#         self.poses = [initialPose.clone().detach().requires_grad_(True).to(device)]
#         self.ids = [initialID]
#         self.edges = []
#         self.candidate_edges = []
        
#         # Configuration parameters
#         self.minLoops = minLoops
#         self.maxAngularDistance = torch.tensor(maxAngularDistance, device=device, dtype=torch.float32)
#         self.maxMotionDistance = maxMotionDistance
#         self.errorThreshold = errorThreshold
#         self.maxSameID = maxSameID

#     def add_odometry(self, newPoseID, Xodo, Podo):
#         # Convert inputs to tensors with gradient tracking
#         Xodo = Xodo.detach().requires_grad_(True) if isinstance(Xodo, torch.Tensor) else torch.tensor(Xodo, device=self.device, dtype=torch.float32, requires_grad=True)
#         Podo = Podo.detach().requires_grad_(True) if isinstance(Podo, torch.Tensor) else torch.tensor(Podo, device=self.device, dtype=torch.float32, requires_grad=True)
        
#         # Ensure proper tensor shapes
#         Xodo = Xodo.view(3, 1)
#         Podo = Podo.view(3, 3)
        
#         # Compute new pose with gradient tracking
#         new_pose = compose_references(self.poses[-1], Xodo)
#         new_pose.retain_grad()  # Critical for gradient flow
#         self.poses.append(new_pose)
#         self.ids.append(newPoseID)
        
#         # Add edge with inverse covariance
#         self.edges.append({
#             'type': 'odom',
#             'from': len(self.poses)-2,
#             'to': len(self.poses)-1,
#             'measurement': Xodo.detach(),
#             'information': torch.inverse(Podo.detach())
#         })

#     def add_loop(self, fromID, toID, Xloop, Ploop):
#         # Convert to tensors with gradient tracking
#         Xloop = Xloop.detach().requires_grad_(True) if isinstance(Xloop, torch.Tensor) else torch.tensor(Xloop, device=self.device, dtype=torch.float32, requires_grad=True)
#         Ploop = Ploop.detach().requires_grad_(True) if isinstance(Ploop, torch.Tensor) else torch.tensor(Ploop, device=self.device, dtype=torch.float32, requires_grad=True)
        
#         # Find indices for the IDs
#         from_idx = self.ids.index(fromID)
#         to_idx = self.ids.index(toID)
        
#         # Add candidate loop closure
#         self.candidate_edges.append({
#             'from': from_idx,
#             'to': to_idx,
#             'measurement': Xloop.view(3, 1),
#             'information': torch.inverse(Ploop.view(3, 3))
#         })

#     def validate(self):
#         if len(self.candidate_edges) < self.minLoops:
#             return []

#         valid_edges = []
#         for edge in self.candidate_edges:
#             from_pose = self.poses[edge['from']]
#             to_pose = self.poses[edge['to']]
            
#             # Compute relative pose with gradient tracking
#             rel_pose = compose_references(invert_reference(from_pose), to_pose)
#             delta = rel_pose - edge['measurement']
            
#             # Check thresholds
#             angular_diff = torch.abs(normalize_angle(delta[2]))
#             distance_diff = torch.norm(delta[:2])
            
#             if angular_diff < self.maxAngularDistance and distance_diff < self.maxMotionDistance:
#                 valid_edges.append(edge)
                
#         return valid_edges

#     def optimize(self):
#         # Convert poses to parameter tensor (maintain gradients)
#         pose_tensor = torch.stack(self.poses, dim=0).requires_grad_(True)
        
#         # Combine all edges
#         all_edges = self.edges + self.validate()
        
#         # Set up optimizer
#         optimizer = torch.optim.LBFGS([pose_tensor], line_search_fn='strong_wolfe')

#         def closure():
#             optimizer.zero_grad()
#             total_error = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
#             for edge in all_edges:
#                 from_idx = edge['from']
#                 to_idx = edge['to']
#                 meas = edge['measurement']
#                 info = edge['information']
                
#                 # Direct tensor access with gradient tracking
#                 from_pose = pose_tensor[from_idx].view(3, 1)
#                 to_pose = pose_tensor[to_idx].view(3, 1)
                
#                 # Compute predicted measurement
#                 pred = compose_references(invert_reference(from_pose), to_pose)
                
#                 # Compute error
#                 error = pred - meas
#                 error[2] = normalize_angle(error[2])
                
#                 # Calculate error term
#                 error_term = torch.matmul(error.T, torch.matmul(info, error))
#                 total_error += error_term.squeeze()
            
#             # Force backward pass
#             if total_error.requires_grad: 
#                 total_error.backward(retain_graph=True)
#             return total_error
        
#         # Run optimization
#         optimizer.step(closure)

        
#         # Update poses with optimized values
#         with torch.no_grad():
#             for i in range(len(self.poses)):
#                 self.poses[i] = pose_tensor[i].clone()
                
#         self.candidate_edges = []

#     def get_poses(self):
#         return [pose.detach().cpu().numpy() for pose in self.poses]
