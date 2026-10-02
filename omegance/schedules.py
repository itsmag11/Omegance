"""Temporal omega schedules.

A schedule is a sequence of *effective* omega scales (values around 1.0), one per denoising step.
Values > 1 suppress detail at that step, values < 1 enhance it.
"""

from typing import Callable, List, Sequence, Union

import numpy as np

ScheduleLike = Union[str, Sequence[float], Callable[[float], float]]


def exponential_schedule(start: float, end: float, gamma: float, steps: int) -> List[float]:
    return [(start - end) * (gamma**i) + end for i in range(steps)]


def cosine_schedule(start: float, end: float, steps: int, alpha: float = 1.0) -> List[float]:
    schedule = []
    for i in range(steps):
        value = end + 0.5 * (start - end) * (1 + np.cos(alpha * np.pi * i / steps))
        value = min(value, start if i == 0 else schedule[-1])
        schedule.append(float(value))
    return schedule


def _mirror(schedule: List[float]) -> List[float]:
    return [2.0 - v for v in schedule]


_GAP = 0.05

PRESETS = {
    # rich detail early, fading back to neutral
    "exp1": lambda n: _mirror(exponential_schedule(1.0 + _GAP, 1.0, 0.9, n)),
    # rich detail early, smoother towards the end
    "exp2": lambda n: _mirror(exponential_schedule(1.0 + 2 * _GAP, 1.0 - 2 * _GAP, 0.9, n)),
    # neutral early, increasingly smooth towards the end
    "cos1": lambda n: _mirror(cosine_schedule(1.0, 1.0 - _GAP, n, 1.5)),
    # smooth early, increasingly detailed towards the end
    "cos2": lambda n: cosine_schedule(1.0 + 2 * _GAP, 1.0 - 2 * _GAP, n, 1.5),
}


def build_schedule(schedule: ScheduleLike, num_steps: int) -> List[float]:
    """Turns a preset name, a list of scales, or a callable `f(progress in [0, 1]) -> scale`
    into a list of `num_steps` effective omega scales."""
    if isinstance(schedule, str):
        if schedule not in PRESETS:
            raise ValueError(f"Unknown omega schedule {schedule!r}. Choose from {sorted(PRESETS)}.")
        return PRESETS[schedule](num_steps)
    if callable(schedule):
        return [float(schedule(i / max(num_steps - 1, 1))) for i in range(num_steps)]
    values = [float(v) for v in schedule]
    if len(values) == num_steps:
        return values
    src = np.linspace(0.0, 1.0, len(values))
    dst = np.linspace(0.0, 1.0, num_steps)
    return np.interp(dst, src, values).tolist()
