"""Contains function for sampling conditions from the dataset.

Also formats for use in problem.optimize and problem.simulate
"""

from datasets import Dataset
from engibench.core import Problem
import numpy as np
import torch as th

from engiopt.transforms import get_scalar_condition_keys


def sample_conditions(
    problem: Problem, n_samples: int, device: th.device, seed: int
) -> tuple[th.Tensor, Dataset, np.ndarray, np.ndarray]:
    """Samples conditions and designs from the dataset and prepares tensors for the generator.

    Only scalar (non-image), non-constant conditions are included in the returned tensor.
    The returned Dataset contains ALL available conditions for use as ``config`` in
    ``problem.optimize()`` / ``problem.simulate()``.

    Args:
        problem: The problem containing the dataset with conditions and designs.
        n_samples: Number of samples to draw.
        device: The device (e.g., 'cpu', 'mps', 'cuda') to place the tensors on.
        seed: Random seed for reproducibility.

    Returns:
        conditions_tensor: Scalar conditions as ``(n_samples, n_scalar_conds)`` tensor.
        sampled_conditions: A Hugging Face Dataset with all available conditions (for config).
        sampled_designs_np: A NumPy array of sampled optimal designs.
        selected_indices: The indices of the sampled conditions and designs.
    """
    ### Set up testing conditions ###
    rng = np.random.default_rng(seed)

    # Extract conditions — keep all dataset-available keys for config passing
    dataset = problem.dataset["test"]
    available_keys = [k for k in problem.conditions_keys if k in dataset.column_names]
    conditions_ds = dataset.select_columns(available_keys)

    # Sample conditions and test_ds designs at random indices
    selected_indices = rng.choice(len(dataset), n_samples, replace=True)
    sampled_conditions = conditions_ds.select(selected_indices)
    sampled_designs_np = np.array(dataset["optimal_design"])[selected_indices]

    # Create tensor from scalar-only conditions (for model input)
    scalar_keys = get_scalar_condition_keys(problem, dataset)
    if scalar_keys:
        conditions_tensor = th.stack(
            [th.tensor(np.array(sampled_conditions[key]), dtype=th.float32, device=device) for key in scalar_keys],
            dim=1,
        )
    else:
        conditions_tensor = th.zeros(n_samples, 0, dtype=th.float32, device=device)

    return conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices
