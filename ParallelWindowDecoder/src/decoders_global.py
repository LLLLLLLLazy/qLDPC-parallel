from __future__ import annotations

import math

import numpy as np

from .bp_osd import make_bp_osd
from .problem import PreparedProblem


def decode_global(problem: PreparedProblem, det_data: np.ndarray, max_iter: int, osd_order: int):
    decoder = make_bp_osd(problem.chk, problem.priors, max_iter, osd_order)
    e_hat = np.zeros((det_data.shape[0], problem.chk.shape[1]), dtype=np.uint8)
    for shot in range(det_data.shape[0]):
        e_hat[shot] = decoder.decode(det_data[shot])
    return e_hat


def decode_sliding(
    problem: PreparedProblem,
    det_data: np.ndarray,
    window_width: int,
    max_iter: int,
    osd_order: int,
    method: int = 1,
    shorten: bool = False,
    shorten_pre_max_iter: int = 8,
):
    """Sliding-window BP+OSD baseline matching SlidingWindowDecoder/osd.py."""
    shots = det_data.shape[0]
    total = np.zeros((shots, problem.chk.shape[1]), dtype=np.uint8)
    num_blocks = problem.num_detector_blocks
    n_half = problem.detector_block_size
    n = 2 * n_half
    F = 1

    anchors = [
        (block * n_half, problem.region_offsets[2 * block])
        for block in range(num_blocks)
    ]
    anchors.append((num_blocks * n_half, problem.chk.shape[1]))

    if method != 0:
        b = anchors[window_width]
        c = anchors[window_width - 1]
        if method == 1:
            c = (c[0], c[1] + (n_half * 3 if problem.z_basis else n))
        noisy_prior = np.sum(
            problem.chk[c[0] : b[0], c[1] : b[1]]
            * problem.priors[c[1] : b[1]],
            axis=1,
        )
        noisy_syndrome_priors = np.ones(n_half) * noisy_prior

    num_win = math.ceil((len(anchors) - window_width + F - 1) / F)
    residual = det_data.copy()
    top_left = 0

    for i in range(num_win):
        a = anchors[top_left]
        bottom_right = min(top_left + window_width, len(anchors) - 1)
        b = anchors[bottom_right]

        if i != num_win - 1 and method != 0:
            c = anchors[top_left + window_width - 1]
            if method == 1:
                c = (c[0], c[1] + (n_half * 3 if problem.z_basis else n))
            noisy_syndrome = np.zeros((b[0] - a[0], n_half), dtype=np.uint8)
            noisy_syndrome[-n_half:, :] = np.eye(n_half, dtype=np.uint8)
            mat = np.hstack((problem.chk[a[0] : b[0], a[1] : c[1]], noisy_syndrome))
            prior = np.concatenate((problem.priors[a[1] : c[1]], noisy_syndrome_priors))
        else:
            c = b
            mat = problem.chk[a[0] : b[0], a[1] : b[1]]
            prior = problem.priors[a[1] : b[1]]

        decoder = make_bp_osd(mat, prior, max_iter, osd_order, shorten, shorten_pre_max_iter)
        detector_win = residual[:, a[0] : b[0]]

        commit = anchors[top_left + F]
        for shot in range(shots):
            local_e = decoder.decode(detector_win[shot]).astype(np.uint8)
            if i == num_win - 1:
                total[shot, a[1] : b[1]] = local_e[: b[1] - a[1]]
            else:
                total[shot, a[1] : commit[1]] = local_e[: commit[1] - a[1]]

        residual = (det_data + total @ problem.chk.T) % 2
        top_left += F

    return total

