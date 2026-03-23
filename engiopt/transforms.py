"""Transformations for the data."""

from collections.abc import Callable

from datasets import Dataset
from engibench.core import Problem
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn.functional as f


def flatten_dict_factory(problem: Problem, device: th.device) -> Callable:
    """Factory function to create a flatten_dict function."""

    def flatten_dict(x):
        """Convert each design in the batch to a flattened tensor."""
        flattened = []
        for design in x:
            # Move to CPU for numpy conversion, then back to device
            design_cpu = {k: v.cpu().numpy() if isinstance(v, th.Tensor) else v for k, v in design.items()}
            flattened_array = spaces.flatten(problem.design_space, design_cpu)
            flattened.append(th.tensor(flattened_array, device=device))
        return th.stack(flattened)

    return flatten_dict


def resize_to(data: th.Tensor, h: int, w: int, mode: str = "bicubic") -> th.Tensor:
    """Resize 2D data back to any desired (h, w). Data should be a Tensor in the format (B, C, H, W)."""
    low_dim = 3
    if data.ndim == low_dim:
        data = data.unsqueeze(1)  # (B, 1, H, W)
    return f.interpolate(data, size=(h, w), mode=mode)


def get_scalar_condition_keys(problem: Problem, dataset: Dataset, *, drop_constants: bool = False) -> list[str]:
    """Return condition keys that are scalar, present in dataset, and (optionally) non-constant.

    Filters ``problem.conditions_keys`` to only those that:
    1. Exist as columns in *dataset*.
    2. Are scalar-valued (``ndim == 0``).
    3. (When *drop_constants* is True) Have non-zero standard deviation.

    Args:
        problem: An EngiBench problem instance.
        dataset: A HuggingFace Dataset split (e.g. ``problem.dataset["train"]``).
        drop_constants: If True, drop columns whose std is 0 across the dataset.

    Returns:
        A list of condition key names suitable for use as model inputs.
    """
    scalar_keys: list[str] = []
    for key in problem.conditions_keys:
        if key not in dataset.column_names:
            continue
        if np.asarray(dataset[0][key]).ndim > 0:
            continue  # skip image / array conditions
        scalar_keys.append(key)

    if drop_constants and scalar_keys:
        conds = th.stack([th.as_tensor(dataset[c][:]).float() for c in scalar_keys], dim=1)
        std = conds.std(dim=0)
        dropped = [c for i, c in enumerate(scalar_keys) if std[i] == 0]
        if dropped:
            print(f"Info: Dropping constant scalar conditions (std=0): {dropped}")
        scalar_keys = [c for i, c in enumerate(scalar_keys) if std[i] > 0]

    return scalar_keys


def get_image_condition_keys(problem: Problem, dataset: Dataset) -> list[str]:
    """Return condition keys that are image/array-valued and present in the dataset.

    This is the complement of :func:`get_scalar_condition_keys` — it returns
    keys whose first sample has ``ndim > 0`` (e.g. 65x65 boundary matrices
    in thermoelastic2d).

    Args:
        problem: An EngiBench problem instance.
        dataset: A HuggingFace Dataset split (e.g. ``problem.dataset["train"]``).

    Returns:
        A list of condition key names that are array-valued.
    """
    img_keys: list[str] = []
    for key in problem.conditions_keys:
        if key not in dataset.column_names:
            continue
        if np.asarray(dataset[0][key]).ndim > 0:
            img_keys.append(key)
    return img_keys


def get_image_condition_shape(dataset: Dataset, img_keys: list[str]) -> tuple[int, ...]:
    """Return the spatial shape of the first image condition.

    Assumes all image conditions share the same spatial dimensions.

    Args:
        dataset: A HuggingFace Dataset split.
        img_keys: Image condition key names (from :func:`get_image_condition_keys`).

    Returns:
        Shape tuple, e.g. ``(65, 65)``.
    """
    return tuple(np.asarray(dataset[0][img_keys[0]]).shape)


def get_performance_target(problem: Problem, dataset: Dataset) -> th.Tensor:
    """Build a scalar performance target for each sample.

    For single-objective problems this returns ``dataset[objectives_keys[0]]``
    directly.  For multi-objective problems with a ``weight`` condition
    (e.g. thermoelastic2d) it returns the weighted sum used by the optimizer:
    ``weight * obj[0] + (1 - weight) * obj[1]``.

    Returns:
        Tensor of shape ``(N, 1)``.
    """
    n_objs = len(problem.objectives_keys)
    if n_objs == 1:
        return th.as_tensor(dataset[problem.objectives_keys[0]][:]).float().unsqueeze(-1)

    # Multi-objective: compute weighted sum
    obj_tensors = [th.as_tensor(dataset[k][:]).float() for k in problem.objectives_keys]

    if "weight" in dataset.column_names:
        w = th.as_tensor(dataset["weight"][:]).float()
        perf = w * obj_tensors[0] + (1.0 - w) * obj_tensors[1]
    else:
        # Equal weighting fallback
        perf = th.stack(obj_tensors, dim=-1).mean(dim=-1)

    print(f"Multi-objective performance target: weighted sum of {problem.objectives_keys[:2]}")
    return perf.unsqueeze(-1)


def normalize(ds: Dataset, condition_names: list[str]) -> tuple[Dataset, th.Tensor, th.Tensor]:
    """Normalize specified condition columns with global mean/std."""
    # stack condition columns into a single tensor (N, C) on CPU
    conds = th.stack([th.as_tensor(ds[c][:]).float() for c in condition_names], dim=1)
    mean = conds.mean(dim=0)
    std = conds.std(dim=0).clamp(min=1e-8)

    # normalize each condition column (HF expects numpy back)
    ds = ds.map(
        lambda batch: {
            c: ((th.as_tensor(batch[c][:]).float() - mean[i]) / std[i]).numpy() for i, c in enumerate(condition_names)
        },
        batched=True,
    )

    return ds, mean, std


def drop_constant(ds: Dataset, condition_names: list[str]) -> tuple[Dataset, list[str]]:
    """Drop constant condition columns (std=0) from dataset."""
    conds = th.stack([th.as_tensor(ds[c][:]).float() for c in condition_names], dim=1)
    std = conds.std(dim=0)

    kept = [c for i, c in enumerate(condition_names) if std[i] > 0]
    dropped = [c for i, c in enumerate(condition_names) if std[i] == 0]

    if dropped:
        print(f"Warning: Dropping constant condition columns (std=0): {dropped}")

    # remove dropped columns from dataset
    ds = ds.remove_columns(dropped)

    return ds, kept
