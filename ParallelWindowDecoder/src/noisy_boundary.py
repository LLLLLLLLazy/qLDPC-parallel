from __future__ import annotations

import numpy as np

from .problem import PreparedProblem


def estimate_noisy_boundary_prior(
    problem: PreparedProblem,
    rows: slice,
    modeled_cols: slice,
    boundary_width: int | None = None,
) -> np.ndarray:
    outside_left = problem.chk[rows, : modeled_cols.start]
    outside_right = problem.chk[rows, modeled_cols.stop :]
    outside_priors = np.concatenate(
        (
            problem.priors[: modeled_cols.start],
            problem.priors[modeled_cols.stop :],
        )
    )
    if outside_priors.size == 0:
        prior = np.zeros(rows.stop - rows.start)
    else:
        outside = np.hstack((outside_left, outside_right))
        prior = np.asarray(outside * outside_priors).reshape(-1)

    if boundary_width is not None:
        mask = np.zeros_like(prior)
        mask[:boundary_width] = prior[:boundary_width]
        mask[-boundary_width:] = np.maximum(mask[-boundary_width:], prior[-boundary_width:])
        prior = mask

    return np.clip(prior, 1e-12, 0.49)


def add_noisy_boundary_columns(
    problem: PreparedProblem,
    mat: np.ndarray,
    prior: np.ndarray,
    rows: slice,
    modeled_cols: slice,
    boundary_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = rows.stop - rows.start
    noisy = np.zeros((row_count, 2 * boundary_width), dtype=np.uint8)
    noisy[:boundary_width, :boundary_width] = np.eye(boundary_width, dtype=np.uint8)
    noisy[-boundary_width:, boundary_width:] = np.eye(boundary_width, dtype=np.uint8)

    noisy_prior_full = estimate_noisy_boundary_prior(
        problem,
        rows,
        modeled_cols,
        boundary_width=boundary_width,
    )
    noisy_prior = np.concatenate(
        (
            noisy_prior_full[:boundary_width],
            noisy_prior_full[-boundary_width:],
        )
    )
    return np.hstack((mat, noisy)), np.concatenate((prior, noisy_prior))

