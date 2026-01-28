"""Utility functions for planning module, including SCOPF helpers."""

import torch


def create_n1_contingency_mask(num_lines: int, device='cpu', dtype=torch.float32):
    """
    Create identity matrix for full N-1 contingency analysis.

    For N-1 analysis, each line is removed one at a time. The mask is an
    identity matrix where mask[i, i] = 1.0 means line i is removed in
    contingency scenario i+1 (scenario 0 is base case, added by ADMM).

    Args:
        num_lines: Number of transmission lines
        device: Torch device ('cpu' or 'cuda')
        dtype: Torch dtype

    Returns:
        torch.Tensor: Shape (num_lines, num_lines), identity matrix

    Example:
        >>> mask = create_n1_contingency_mask(3)
        >>> mask
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
    """
    return torch.eye(num_lines, device=device, dtype=dtype)


def create_critical_line_contingency_mask(
    critical_line_indices: list[int],
    num_lines: int,
    device='cpu',
    dtype=torch.float32
):
    """
    Create contingency mask for subset of critical lines.

    Args:
        critical_line_indices: List of line indices to include in contingencies
        num_lines: Total number of lines in network
        device: Torch device
        dtype: Torch dtype

    Returns:
        torch.Tensor: Shape (len(critical_line_indices), num_lines)

    Example:
        >>> mask = create_critical_line_contingency_mask([0, 5, 10], num_lines=100)
        >>> mask.shape
        torch.Size([3, 100])
    """
    critical_line_indices = [int(i) for i in critical_line_indices]
    if any((i < 0 or i >= int(num_lines)) for i in critical_line_indices):
        bad = [i for i in critical_line_indices if (i < 0 or i >= int(num_lines))]
        raise ValueError(f"critical_line_indices out of range [0, {num_lines}): {bad}")

    num_contingencies = len(critical_line_indices)
    mask = torch.zeros((num_contingencies, num_lines), device=device, dtype=dtype)
    for i, line_idx in enumerate(critical_line_indices):
        mask[i, line_idx] = 1.0
    return mask
