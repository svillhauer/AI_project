#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Modified version with PyTorch GPU support
###############################################################################

import math
import numpy as np
import torch

###############################################################################
# ANGLE NORMALIZATION
###############################################################################
def normalize_angle(angle):
    """Normalize angle to [-π, π] (supports NumPy and PyTorch)"""
    if isinstance(angle, torch.Tensor):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    return (angle + math.pi) % (2 * math.pi) - math.pi

###############################################################################
# POSE COMPOSITION (GRADIENT-FRIENDLY VERSION)
###############################################################################
def compose_references(X1, X2):
    """Compose two 2D transformations with gradient tracking"""
    # Handle PyTorch tensors
    if isinstance(X1, torch.Tensor) or isinstance(X2, torch.Tensor):
        device = X1.device if isinstance(X1, torch.Tensor) else X2.device
        
        # Ensure proper tensor dimensions and gradient tracking
        X1 = X1.view(3, 1).detach().requires_grad_(True) if X1.requires_grad else X1.view(3, 1)
        X2 = X2.view(3, 1).detach().requires_grad_(True) if X2.requires_grad else X2.view(3, 1)
        
        theta = X1[2, 0]
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        dx = X2[0, 0]
        dy = X2[1, 0]
        dtheta = X2[2, 0]
        
        # Maintain computation graph using tensor operations
        x = X1[0, 0] + dx * c - dy * s
        y = X1[1, 0] + dx * s + dy * c
        theta = normalize_angle(theta + dtheta)
        
        return torch.stack([x, y, theta]).view(3, 1).to(device).requires_grad_(True)
    
    # NumPy implementation
    c = np.cos(X1[2, 0])
    s = np.sin(X1[2, 0])
    return np.array([
        [X1[0, 0] + X2[0, 0]*c - X2[1, 0]*s],
        [X1[1, 0] + X2[0, 0]*s + X2[1, 0]*c],
        [normalize_angle(X1[2, 0] + X2[2, 0])]
    ])

###############################################################################
# POSE INVERSION (GRADIENT-FRIENDLY VERSION)
###############################################################################
def invert_reference(X):
    """Invert a 2D transformation with gradient tracking"""
    if isinstance(X, torch.Tensor):
        c = torch.cos(X[2, 0])
        s = torch.sin(X[2, 0])
        return torch.stack([
            -X[0, 0]*c - X[1, 0]*s,
            X[0, 0]*s - X[1, 0]*c,
            normalize_angle(-X[2, 0])
        ]).view(3, 1).requires_grad_(True)
    
    # NumPy implementation
    c = np.cos(X[2, 0])
    s = np.sin(X[2, 0])
    return np.array([
        [-X[0, 0]*c - X[1, 0]*s],
        [X[0, 0]*s - X[1, 0]*c],
        [normalize_angle(-X[2, 0])]
    ])

###############################################################################
# TRAJECTORY COMPOSITION
###############################################################################
def compose_trajectory(poses):
    """Convert relative poses to absolute trajectory"""
    if not poses:
        return np.zeros((3, 0))
    
    if isinstance(poses[0], torch.Tensor):
        traj = [poses[0].clone().requires_grad_(True)]
        for pose in poses[1:]:
            traj.append(compose_references(traj[-1], pose.requires_grad_(True)))
        return torch.cat(traj, dim=1)
    else:
        traj = [poses[0].copy()]
        for pose in poses[1:]:
            traj.append(compose_references(traj[-1], pose))
        return np.hstack(traj)

###############################################################################
# LEAST SQUARES MOTION ESTIMATION
###############################################################################
def least_squares_cartesian(Sref, Scur):
    """Motion estimation (supports NumPy/PyTorch)"""
    if isinstance(Sref, torch.Tensor) or isinstance(Scur, torch.Tensor):
        device = Sref.device if isinstance(Sref, torch.Tensor) else Scur.device
        
        # Ensure gradient tracking
        Sref = Sref.requires_grad_(True) if not Sref.requires_grad else Sref
        Scur = Scur.requires_grad_(True) if not Scur.requires_grad else Scur
        
        mx = torch.mean(Scur[0, :])
        my = torch.mean(Scur[1, :])
        mx2 = torch.mean(Sref[0, :])
        my2 = torch.mean(Sref[1, :])
        
        Sxx = torch.sum((Scur[0, :]-mx)*(Sref[0, :]-mx2))
        Syy = torch.sum((Scur[1, :]-my)*(Sref[1, :]-my2))
        Sxy = torch.sum((Scur[0, :]-mx)*(Sref[1, :]-my2))
        Syx = torch.sum((Scur[1, :]-my)*(Sref[0, :]-mx2))
        
        o = torch.atan2(Sxy-Syx, Sxx+Syy)
        x = mx2 - (mx*torch.cos(o) - my*torch.sin(o))
        y = my2 - (mx*torch.sin(o) + my*torch.cos(o))
        return torch.stack([x, y, o]).view(3, 1).to(device).requires_grad_(True)
    
    # NumPy implementation
    mx = np.mean(Scur[0, :])
    my = np.mean(Scur[1, :])
    mx2 = np.mean(Sref[0, :])
    my2 = np.mean(Sref[1, :])
    
    Sxx = np.sum((Scur[0, :]-mx)*(Sref[0, :]-mx2))
    Syy = np.sum((Scur[1, :]-my)*(Sref[1, :]-my2))
    Sxy = np.sum((Scur[0, :]-mx)*(Sref[1, :]-my2))
    Syx = np.sum((Scur[1, :]-my)*(Sref[0, :]-mx2))
    
    o = math.atan2(Sxy-Syx, Sxx+Syy)
    x = mx2 - (mx*math.cos(o) - my*math.sin(o))
    y = my2 - (mx*math.sin(o) + my*math.cos(o))
    return np.array([[x], [y], [o]])
