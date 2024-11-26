import numpy as np

def exponential_scheduler(initial_lr, min_lr, gamma, steps):
    return [(initial_lr - min_lr) * (gamma ** step) + min_lr for step in range(steps)]

def cosine_scheduler(initial_lr, min_lr, steps, alpha=1.0):
    lr_schedule = []
    for step in range(steps):
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(alpha * np.pi * step / steps))
        lr = min(lr, initial_lr if step == 0 else lr_schedule[-1])
        lr_schedule.append(lr)
    return lr_schedule

# def cosine_increase_scheduler(start, end, steps, alpha):
#     schedule = []
#     for step in range(steps):
#         cosine_decay = 0.5 * (1 + np.cos(np.pi * step / steps))
#         lr = end + (start - end) * (cosine_decay ** alpha)
#         schedule.append(lr)
#     return schedule

def step_scheduler(center, abs, change, steps):
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
def exponential_scheduler_mirrored(initial_lr, min_lr, gamma, steps):
    original = exponential_scheduler(initial_lr, min_lr, gamma, steps)
    return [2 - lr for lr in original]

def cosine_scheduler_mirrored(initial_lr, min_lr, steps, alpha=1.0):
    original = cosine_scheduler(initial_lr, min_lr, steps, alpha)
    return [2 - lr for lr in original]

def step_scheduler_mirrored(center, abs, change, steps):
    original = step_scheduler(center, abs, change, steps)
    return [2 - lr for lr in original]
