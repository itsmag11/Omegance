"""
Omega Schedulers for Dynamic Detail Control

This module provides various scheduling strategies for the Omega parameter
to achieve dynamic detail control during diffusion model inference.
"""

import numpy as np

def exponential_scheduler(initial_lr, min_lr, gamma, steps):
    """
    Exponential decay scheduler for Omega values.
    
    Args:
        initial_lr: Starting Omega value
        min_lr: Minimum Omega value
        gamma: Decay factor
        steps: Number of timesteps
        
    Returns:
        List of Omega values following exponential decay
    """
    return [(initial_lr - min_lr) * (gamma ** step) + min_lr for step in range(steps)]

def cosine_scheduler(initial_lr, min_lr, steps, alpha=1.0):
    """
    Cosine annealing scheduler for Omega values.
    
    Args:
        initial_lr: Starting Omega value
        min_lr: Minimum Omega value
        steps: Number of timesteps
        alpha: Cosine annealing factor
        
    Returns:
        List of Omega values following cosine annealing
    """
    lr_schedule = []
    for step in range(steps):
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(alpha * np.pi * step / steps))
        lr = min(lr, initial_lr if step == 0 else lr_schedule[-1])
        lr_schedule.append(lr)
    return lr_schedule

def step_scheduler(center, abs, change, steps):
    """
    Step scheduler for Omega values with discrete changes.
    
    Args:
        center: Center Omega value
        abs: Absolute deviation from center
        change: Timestep at which to change
        steps: Number of timesteps
        
    Returns:
        List of Omega values with step changes
    """
    schedule = []
    for step in range(steps):
        if step <= change:
            lr = center - abs
        if step > (change + 10):
            lr = center + abs
        else:
            lr = center
        schedule.append(lr)
    return schedule

# Mirrored versions of the functions with respect to y = 1.0
# These are used for Omega scheduling where values are mirrored around 1.0

def exponential_scheduler_mirrored(initial_lr, min_lr, gamma, steps):
    """
    Mirrored exponential scheduler for Omega values.
    Values are mirrored around 1.0 (y = 2 - x).
    
    Args:
        initial_lr: Starting Omega value
        min_lr: Minimum Omega value
        gamma: Decay factor
        steps: Number of timesteps
        
    Returns:
        List of mirrored Omega values following exponential decay
    """
    original = exponential_scheduler(initial_lr, min_lr, gamma, steps)
    return [2 - lr for lr in original]

def cosine_scheduler_mirrored(initial_lr, min_lr, steps, alpha=1.0):
    """
    Mirrored cosine scheduler for Omega values.
    Values are mirrored around 1.0 (y = 2 - x).
    
    Args:
        initial_lr: Starting Omega value
        min_lr: Minimum Omega value
        steps: Number of timesteps
        alpha: Cosine annealing factor
        
    Returns:
        List of mirrored Omega values following cosine annealing
    """
    original = cosine_scheduler(initial_lr, min_lr, steps, alpha)
    return [2 - lr for lr in original]

def step_scheduler_mirrored(center, abs, change, steps):
    """
    Mirrored step scheduler for Omega values.
    Values are mirrored around 1.0 (y = 2 - x).
    
    Args:
        center: Center Omega value
        abs: Absolute deviation from center
        change: Timestep at which to change
        steps: Number of timesteps
        
    Returns:
        List of mirrored Omega values with step changes
    """
    original = step_scheduler(center, abs, change, steps)
    return [2 - lr for lr in original]
