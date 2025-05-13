import torch
import numpy as np
from transform2d_gpu import compose_references, invert_reference, normalize_angle

class GraphOptimizerGPU:
    def __init__(self, initialID, initialPose, device='cuda',
                 minLoops=5, maxAngularDistance=np.pi/4, 
                 maxMotionDistance=500, errorThreshold=500, maxSameID=2):
        self.device = device
        self.poses = [initialPose.view(3, 1).to(device)]
        self.ids = [initialID]
        self.edges = []
        self.candidate_edges = []
        
        # Configuration parameters with defaults
        self.minLoops = minLoops
        self.maxAngularDistance = maxAngularDistance
        self.maxMotionDistance = maxMotionDistance
        self.errorThreshold = errorThreshold
        self.maxSameID = maxSameID

    def add_odometry(self, newPoseID, Xodo, Podo):
        # Ensure proper tensor shapes
        Xodo = Xodo.view(3, 1) if isinstance(Xodo, torch.Tensor) else torch.tensor(Xodo, device=self.device).view(3, 1)
        Podo = Podo.view(3, 3) if isinstance(Podo, torch.Tensor) else torch.tensor(Podo, device=self.device).view(3, 3)
        
        # Compute new pose
        new_pose = compose_references(self.poses[-1], Xodo)
        self.poses.append(new_pose)
        self.ids.append(newPoseID)
        
        # Add edge with inverse covariance
        self.edges.append({
            'type': 'odom',
            'from': len(self.poses)-2,
            'to': len(self.poses)-1,
            'measurement': Xodo,
            'information': torch.inverse(Podo)
        })

    def add_loop(self, fromID, toID, Xloop, Ploop):
        # Ensure proper tensor shapes
        Xloop = Xloop.view(3, 1) if isinstance(Xloop, torch.Tensor) else torch.tensor(Xloop, device=self.device).view(3, 1)
        Ploop = Ploop.view(3, 3) if isinstance(Ploop, torch.Tensor) else torch.tensor(Ploop, device=self.device).view(3, 3)
        
        # Find indices for the IDs
        from_idx = self.ids.index(fromID)
        to_idx = self.ids.index(toID)
        
        # Add candidate loop closure
        self.candidate_edges.append({
            'from': from_idx,
            'to': to_idx,
            'measurement': Xloop,
            'information': torch.inverse(Ploop)
        })

    def validate(self):
        if len(self.candidate_edges) < self.minLoops:
            return []

        valid_edges = []
        for edge in self.candidate_edges:
            from_pose = self.poses[edge['from']]
            to_pose = self.poses[edge['to']]
            
            # Compute relative pose
            rel_pose = compose_references(invert_reference(from_pose), to_pose)
            delta = rel_pose - edge['measurement']
            
            # Check thresholds
            angular_diff = torch.abs(normalize_angle(delta[2]))
            distance_diff = torch.norm(delta[:2])
            
            if angular_diff < self.maxAngularDistance and distance_diff < self.maxMotionDistance:
                valid_edges.append(edge)
                
        return valid_edges

    def optimize(self):
        # Convert poses to parameter tensor
        pose_tensor = torch.stack(self.poses, dim=0).requires_grad_(True)
        
        # Combine all edges
        all_edges = self.edges + self.validate()
        
        # Set up optimizer
        optimizer = torch.optim.LBFGS([pose_tensor], line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            total_error = torch.tensor(0.0, device=self.device)
            
            for edge in all_edges:
                from_idx = edge['from']
                to_idx = edge['to']
                meas = edge['measurement']
                info = edge['information']
                
                # Compute predicted measurement
                pred = compose_references(
                    invert_reference(pose_tensor[from_idx]), 
                    pose_tensor[to_idx]
                )
                
                # Compute error term correctly
                error = pred - meas
                error[2] = normalize_angle(error[2])
                
                # Correct matrix multiplication order: error^T @ info @ error
                error_term = torch.matmul(error.T, torch.matmul(info, error))
                total_error += error_term.squeeze()
                
            total_error.backward()
            return total_error
        
        # Run optimization
        optimizer.step(closure)
        
        # Update poses with optimized values
        with torch.no_grad():
            for i in range(len(self.poses)):
                self.poses[i] = pose_tensor[i].clone()
                
        self.candidate_edges = []

    def get_poses(self):
        return [pose.cpu().numpy() for pose in self.poses]
