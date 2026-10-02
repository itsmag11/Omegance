"""Omegance as a drop-in patch for diffusers schedulers.

Omegance only changes how the scheduler consumes the model prediction, so instead of forking pipelines we
patch `scheduler.step` and leave every diffusers pipeline untouched.
"""

import functools
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .schedules import ScheduleLike, build_schedule

OmegaLike = Union[float, np.ndarray, torch.Tensor]

EPS_BOUNDS = (0.95, 1.05)
FLOW_BOUNDS = (0.9, 1.1)
STEEPNESS = 0.1

_PATCHED_CLASSES = {}


def rescale_omega(omega, bounds: Tuple[float, float], k: float = STEEPNESS):
    """Maps a user-facing omega (roughly [-10, 10], 0 = no change) to an effective scale in `bounds`."""
    low, high = bounds
    if isinstance(omega, torch.Tensor):
        return low + (high - low) * torch.sigmoid(k * omega)
    return low + (high - low) / (1.0 + math.exp(-k * float(omega)))


def omega_mask(mask, low: float, high: float) -> np.ndarray:
    """Builds a spatial omega map from a mask image: black (0) -> `low`, white (1) -> `high`.

    `mask` can be a PIL image, a numpy array or a tensor, in [0, 255] or [0, 1]. Grey values interpolate linearly.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=-1) if mask.shape[-1] in (1, 3, 4) else mask[0]
    if mask.max() > 1.0:
        mask = mask / 255.0
    return low + mask * (high - low)


def _is_flow_matching(scheduler) -> bool:
    return any("FlowMatch" in cls.__name__ for cls in type(scheduler).__mro__)


class OmeganceMixin:
    """State and logic shared by every patched scheduler. Use `apply_omegance` rather than this class directly."""

    _omegance_base = None

    def set_omega(
        self,
        omega: OmegaLike = 0.0,
        schedule: Optional[ScheduleLike] = None,
        active_range: Tuple[float, float] = (0.0, 1.0),
        bounds: Optional[Tuple[float, float]] = None,
    ):
        """Configures Omegance.

        Args:
            omega: Global control as a float (positive = smoother / fewer details, negative = richer details,
                0 = unchanged), or spatial control as a 2D map of omega values (see `omega_mask`).
            schedule: Temporal control. A preset name (`"exp1"`, `"exp2"`, `"cos1"`, `"cos2"`), a list of effective
                scales around 1.0 (resampled to the number of steps), or a callable `f(progress) -> scale`.
                Overrides `omega` when set.
            active_range: Fraction of the denoising trajectory `(start, end)` in which `omega` is applied, where 0 is
                the first step (largest timestep, pure noise) and 1 is the last step. E.g. `(0.0, 0.5)` = early half.
            bounds: Range of the effective scale that `omega` is squashed into. Defaults to (0.95, 1.05) for
                noise-prediction models and (0.9, 1.1) for flow-matching models.
        """
        if isinstance(omega, (np.ndarray, torch.Tensor)) and omega.ndim > 0:
            if omega.ndim != 2:
                raise ValueError(f"A spatial omega map must be 2D (H, W), got shape {tuple(omega.shape)}.")
            omega = torch.as_tensor(omega).detach().to("cpu", torch.float32)
        else:
            omega = float(omega)

        self._omega = omega
        self._omega_schedule = schedule
        self._omega_active_range = tuple(active_range)
        self._omega_bounds = tuple(bounds) if bounds is not None else (
            FLOW_BOUNDS if _is_flow_matching(self) else EPS_BOUNDS
        )
        self._omega_map_cache = {}
        self._omega_schedule_cache = {}
        return self

    @property
    def omega(self):
        return self._omega

    def _omegance_step_index(self, timestep) -> int:
        step_index = getattr(self, "step_index", None)
        if step_index is not None:
            return step_index
        begin_index = getattr(self, "begin_index", None)
        if begin_index is not None:
            return begin_index
        timesteps = self.timesteps
        t = timestep.to(timesteps.device) if isinstance(timestep, torch.Tensor) else timestep
        matches = (timesteps == t).nonzero()
        return int(matches[0]) if len(matches) else 0

    def _omegance_scale(self, timestep, model_output: torch.Tensor):
        """Effective omega scale for this step: a float or a tensor broadcastable to `model_output`."""
        num_steps = len(self.timesteps)
        index = self._omegance_step_index(timestep)

        if self._omega_schedule is not None:
            if num_steps not in self._omega_schedule_cache:
                self._omega_schedule_cache[num_steps] = build_schedule(self._omega_schedule, num_steps)
            return self._omega_schedule_cache[num_steps][min(index, num_steps - 1)]

        start, end = self._omega_active_range
        progress = index / num_steps
        if progress < start or (progress >= end and end < 1.0):
            return 1.0

        if not isinstance(self._omega, torch.Tensor):
            return rescale_omega(self._omega, self._omega_bounds)

        key = (tuple(model_output.shape), model_output.device)
        if key not in self._omega_map_cache:
            omega_map = _fit_map(self._omega, model_output).to(model_output.device)
            self._omega_map_cache[key] = rescale_omega(omega_map, self._omega_bounds)
        return self._omega_map_cache[key]

    def _omegance_transform(self, model_output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        scale = self._omegance_scale(timestep, model_output)
        if isinstance(scale, float) and scale == 1.0:
            return model_output

        dtype = model_output.dtype
        out = model_output.float()
        if isinstance(scale, torch.Tensor):
            scale = scale.float()

        if _is_flow_matching(self):
            mean = out.mean()
            out = (out - mean) * scale + mean
            return out.to(dtype)

        prediction_type = getattr(self.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            out = out * scale
        elif prediction_type == "v_prediction":
            out = self._omegance_scale_v(out, timestep, sample.float(), scale)
        else:
            raise NotImplementedError(f"Omegance does not support prediction_type={prediction_type!r}.")
        return out.to(dtype)

    def _omegance_scale_v(self, v, timestep, sample, scale):
        """Scales the implied noise prediction of a v-prediction model and converts back to v."""
        if float(getattr(self, "init_noise_sigma", 1.0)) > 1.0:
            # k-diffusion style (Euler, Heun, ...): sample = x0 + sigma * eps
            if getattr(self, "step_index", None) is None and hasattr(self, "_init_step_index"):
                self._init_step_index(timestep)
            sigma = self.sigmas[self.step_index].to(v.device).float()
            c = (sigma**2 + 1) ** 0.5
            x0 = -v * sigma / c + sample / (sigma**2 + 1)
            eps = (sample - x0) / sigma
            x0_new = sample - sigma * eps * scale
            return (x0_new - sample / (sigma**2 + 1)) * (-c / sigma)

        # DDPM style (DDIM, DPM-Solver, ...): sample = sqrt(a) * x0 + sqrt(1 - a) * eps
        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        alpha_prod = self.alphas_cumprod[t].to(v.device).float()
        sqrt_a, sqrt_1ma = alpha_prod**0.5, (1 - alpha_prod) ** 0.5
        eps = sqrt_a * v + sqrt_1ma * sample
        return (eps * scale - sqrt_1ma * sample) / sqrt_a


def _fit_map(omega_map: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
    """Resizes a 2D omega map so that it broadcasts against `model_output`."""
    grid = omega_map[None, None]
    if model_output.ndim >= 4:
        h, w = model_output.shape[-2:]
        grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return grid.view(*([1] * (model_output.ndim - 2)), h, w)
    if model_output.ndim == 3:
        # packed sequence latents (e.g. Flux): (batch, seq_len, channels)
        h, w = _grid_for_sequence(model_output.shape[1], omega_map.shape[0] / omega_map.shape[1])
        grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        return grid.reshape(1, h * w, 1)
    raise ValueError(f"Spatial omega is not supported for model outputs of shape {tuple(model_output.shape)}.")


def _grid_for_sequence(seq_len: int, aspect: float) -> Tuple[int, int]:
    best = None
    for h in range(1, int(math.isqrt(seq_len)) + 1):
        if seq_len % h:
            continue
        for hh, ww in ((h, seq_len // h), (seq_len // h, h)):
            err = abs(math.log(hh / ww) - math.log(aspect))
            if best is None or err < best[0]:
                best = (err, hh, ww)
    return best[1], best[2]


def _patched_class(base: type) -> type:
    if base in _PATCHED_CLASSES:
        return _PATCHED_CLASSES[base]
    base_step = base.step

    @functools.wraps(base_step)
    def step(self, model_output, timestep, sample, *args, **kwargs):
        model_output = self._omegance_transform(model_output, timestep, sample)
        return base_step(self, model_output, timestep, sample, *args, **kwargs)

    # Keep the base name/module so `save_pretrained` writes a config that plain diffusers can load.
    cls = type(
        base.__name__,
        (OmeganceMixin, base),
        {"step": step, "_omegance_base": base, "__module__": base.__module__, "__qualname__": base.__qualname__},
    )
    _PATCHED_CLASSES[base] = cls
    return cls


def apply_omegance(
    pipe_or_scheduler,
    omega: OmegaLike = 0.0,
    schedule: Optional[ScheduleLike] = None,
    active_range: Tuple[float, float] = (0.0, 1.0),
    bounds: Optional[Tuple[float, float]] = None,
):
    """Enables Omegance on a diffusers pipeline (or a scheduler) in place and returns the patched scheduler.

    Example:
        >>> pipe = StableDiffusionXLPipeline.from_pretrained(...)
        >>> apply_omegance(pipe, omega=-5)       # richer details
        >>> pipe.scheduler.set_omega(5)          # smoother, change any time
    """
    scheduler = getattr(pipe_or_scheduler, "scheduler", pipe_or_scheduler)
    if not isinstance(scheduler, OmeganceMixin):
        scheduler.__class__ = _patched_class(type(scheduler))
    return scheduler.set_omega(omega=omega, schedule=schedule, active_range=active_range, bounds=bounds)


def remove_omegance(pipe_or_scheduler):
    """Restores the original scheduler class."""
    scheduler = getattr(pipe_or_scheduler, "scheduler", pipe_or_scheduler)
    if isinstance(scheduler, OmeganceMixin):
        scheduler.__class__ = scheduler._omegance_base
    return scheduler
