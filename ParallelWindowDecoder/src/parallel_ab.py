from __future__ import annotations

import itertools
import math

import numpy as np

from .bp_osd import (
    choose_boundary_feedback_flips,
    error_cost,
    make_bp_osd,
    run_a_candidate_payloads,
    run_a_repair_payloads,
    run_window_payloads,
    scale_priors_by_llr,
    unique_concat,
)
from .noisy_boundary import add_noisy_boundary_columns
from .problem import PreparedProblem, region_col_slice, row_slice


def s_range(start_block: int, stop_block: int) -> str:
    if stop_block <= start_block:
        return "empty"
    if stop_block == start_block + 1:
        return f"s{start_block + 1}"
    return f"s{start_block + 1}-s{stop_block}"


def decode_parallel_ab(
    problem: PreparedProblem,
    det_data: np.ndarray,
    b_width: int,
    a_radius: int,
    a_size: int | None,
    a_solve_size: int | None,
    max_iter: int,
    osd_order: int,
    oracle_errors: np.ndarray | None = None,
    a_noisy_boundary: bool = False,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    top_k_boundary: int = 1,
    soft_boundary_message: bool = False,
    soft_boundary_one_prior: float = 0.2,
    soft_boundary_zero_prior: float = 1e-4,
    window_shorten: bool = False,
    shorten_pre_max_iter: int = 8,
    b_noisy_boundary: bool = False,
    flag_triggered_single_flip: bool = False,
    bounded_two_flip: bool = False,
    two_flip_candidates: int = 4,
    a_boundary_weight_scale: float = 1.0,
    aba_boundary_feedback: bool = False,
    aba_max_flips: int = 2,
    aba_candidate_cols: int = 16,
    a_neighbor_rerank: bool = False,
    a_rerank_top_k: int = 2,
    a_neighbor_beta: float = 1.0,
    a_two_color_order: str = "none",
    a_shifted_ensemble: bool = False,
    a_shift_offsets: tuple[int, ...] = (-2, 0, 2),
    a_shifted_beta: float = 1.0,
    tsae_top_k: int = 1,
    tsae_boundary_top_k: int = 1,
    tsae_stitch_mode: str = "none",
    tsae_interface_branch: bool = False,
    tsae_interface_gated: bool = False,
    tsae_interface_cols_per_side: int = 1,
    tsae_interface_joint_score: bool = False,
    tsae_interface_zb_weight: float = 1.0,
    tsae_interface_commit_offset: int = 0,
    a_shifted_chain_dp: bool = False,
    a_shifted_joint_b_dp: bool = False,
    a_shifted_joint_b_dp_lag: int = 0,
    a_shifted_joint_flag_penalty: float = 1000.0,
    a_micro_sliding: bool = False,
    a_micro_sliding_order: str = "asc",
    oracle_diag: np.ndarray | None = None,
    z_boundary_repair: bool = False,
    z_repair_edge_width: int = 2,
    z_joint_retry: bool = False,
    seam_diagnostics: bool = False,
):
    if a_size is not None:
        return decode_staggered_ab(
            problem,
            det_data,
            n_b=b_width,
            n_a=a_size,
            n_a_solve=a_solve_size or a_size,
            max_iter=max_iter,
            osd_order=osd_order,
            oracle_errors=oracle_errors,
            a_noisy_boundary=a_noisy_boundary,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
            top_k_boundary=top_k_boundary,
            soft_boundary_message=soft_boundary_message,
            soft_boundary_one_prior=soft_boundary_one_prior,
            soft_boundary_zero_prior=soft_boundary_zero_prior,
            window_shorten=window_shorten,
            shorten_pre_max_iter=shorten_pre_max_iter,
            b_noisy_boundary=b_noisy_boundary,
            flag_triggered_single_flip=flag_triggered_single_flip,
            bounded_two_flip=bounded_two_flip,
            two_flip_candidates=two_flip_candidates,
            a_boundary_weight_scale=a_boundary_weight_scale,
            aba_boundary_feedback=aba_boundary_feedback,
            aba_max_flips=aba_max_flips,
            aba_candidate_cols=aba_candidate_cols,
            a_neighbor_rerank=a_neighbor_rerank,
            a_rerank_top_k=a_rerank_top_k,
            a_neighbor_beta=a_neighbor_beta,
            a_two_color_order=a_two_color_order,
            a_shifted_ensemble=a_shifted_ensemble,
            a_shift_offsets=a_shift_offsets,
            a_shifted_beta=a_shifted_beta,
            tsae_top_k=tsae_top_k,
            tsae_boundary_top_k=tsae_boundary_top_k,
            tsae_stitch_mode=tsae_stitch_mode,
            tsae_interface_branch=tsae_interface_branch,
            tsae_interface_gated=tsae_interface_gated,
            tsae_interface_cols_per_side=tsae_interface_cols_per_side,
            tsae_interface_joint_score=tsae_interface_joint_score,
            tsae_interface_zb_weight=tsae_interface_zb_weight,
            tsae_interface_commit_offset=tsae_interface_commit_offset,
            a_shifted_chain_dp=a_shifted_chain_dp,
            a_shifted_joint_b_dp=a_shifted_joint_b_dp,
            a_shifted_joint_b_dp_lag=a_shifted_joint_b_dp_lag,
            a_shifted_joint_flag_penalty=a_shifted_joint_flag_penalty,
            a_micro_sliding=a_micro_sliding,
            a_micro_sliding_order=a_micro_sliding_order,
            oracle_diag=oracle_diag,
            z_boundary_repair=z_boundary_repair,
            z_repair_edge_width=z_repair_edge_width,
            z_joint_retry=z_joint_retry,
            seam_diagnostics=seam_diagnostics,
        )

    shots = det_data.shape[0]
    total = np.zeros((shots, problem.chk.shape[1]), dtype=np.uint8)
    num_blocks = problem.num_detector_blocks
    boundaries = list(range(b_width, num_blocks, b_width))

    a_flagged = 0
    for boundary in boundaries:
        boundary_region = 2 * boundary - 1
        boundary_cols = region_col_slice(problem, boundary_region, boundary_region + 1)
        if oracle_errors is not None:
            total[:, boundary_cols] = oracle_errors[:, boundary_cols]
        else:
            if a_size is None:
                block_start = max(0, boundary - a_radius)
                block_stop = min(num_blocks, boundary + a_radius)
                col_region_start = 2 * block_start
                col_region_stop = 2 * block_stop - 1
            else:
                block_start = boundary
                block_stop = min(num_blocks, boundary + a_size)
                col_region_start = max(0, 2 * block_start - 1)
                col_region_stop = min(2 * block_stop, len(problem.region_offsets) - 1)
            rows = row_slice(problem, block_start, block_stop)
            cols = region_col_slice(problem, col_region_start, col_region_stop)
            local_boundary_start = boundary_cols.start - cols.start
            local_boundary_stop = boundary_cols.stop - cols.start

            decoder = make_bp_osd(
                problem.chk[rows, cols],
                problem.priors[cols],
                max_iter,
                osd_order,
                window_shorten,
                shorten_pre_max_iter,
            )
            for shot in range(shots):
                local_e = decoder.decode(det_data[shot, rows])
                if ((problem.chk[rows, cols] @ local_e + det_data[shot, rows]) % 2).any():
                    a_flagged += 1
                total[shot, boundary_cols] = local_e[local_boundary_start:local_boundary_stop]

    residual = (det_data + total @ problem.chk.T) % 2

    b_flagged = 0
    for block_start in range(0, num_blocks, b_width):
        block_stop = min(block_start + b_width, num_blocks)
        rows = row_slice(problem, block_start, block_stop)
        cols = region_col_slice(problem, 2 * block_start, 2 * block_stop - 1)
        mat = problem.chk[rows, cols]
        prior = problem.priors[cols]
        value_limit = mat.shape[1]
        if b_noisy_boundary:
            mat, prior = add_noisy_boundary_columns(
                problem,
                mat,
                prior,
                rows,
                cols,
                boundary_width=problem.detector_block_size,
            )
        decoder = make_bp_osd(mat, prior, max_iter, osd_order, window_shorten, shorten_pre_max_iter)
        for shot in range(shots):
            local_e = decoder.decode(residual[shot, rows]).astype(np.uint8)
            if ((mat @ local_e + residual[shot, rows]) % 2).any():
                b_flagged += 1
            total[shot, cols] = local_e[:value_limit]

    diagnostics = {
        "a_boundaries": len(boundaries),
        "a_source": "oracle" if oracle_errors is not None else "decoded",
        "a_size": a_size if a_size is not None else f"radius:{a_radius}",
        "a_flagged_local": a_flagged,
        "b_segments": math.ceil(num_blocks / b_width),
        "b_flagged_local": b_flagged,
        "b_noisy_boundary": b_noisy_boundary,
        "window_shorten": window_shorten,
        "shorten_pre_max_iter": shorten_pre_max_iter,
    }
    return total, diagnostics


def decode_staggered_ab(
    problem: PreparedProblem,
    det_data: np.ndarray,
    n_b: int,
    n_a: int,
    n_a_solve: int,
    max_iter: int,
    osd_order: int,
    oracle_errors: np.ndarray | None = None,
    a_noisy_boundary: bool = False,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    top_k_boundary: int = 1,
    soft_boundary_message: bool = False,
    soft_boundary_one_prior: float = 0.2,
    soft_boundary_zero_prior: float = 1e-4,
    window_shorten: bool = False,
    shorten_pre_max_iter: int = 8,
    b_noisy_boundary: bool = False,
    flag_triggered_single_flip: bool = False,
    bounded_two_flip: bool = False,
    two_flip_candidates: int = 4,
    a_boundary_weight_scale: float = 1.0,
    aba_boundary_feedback: bool = False,
    aba_max_flips: int = 2,
    aba_candidate_cols: int = 16,
    a_neighbor_rerank: bool = False,
    a_rerank_top_k: int = 2,
    a_neighbor_beta: float = 1.0,
    a_two_color_order: str = "none",
    a_shifted_ensemble: bool = False,
    a_shift_offsets: tuple[int, ...] = (-2, 0, 2),
    a_shifted_beta: float = 1.0,
    tsae_top_k: int = 1,
    tsae_boundary_top_k: int = 1,
    tsae_stitch_mode: str = "none",
    tsae_interface_branch: bool = False,
    tsae_interface_gated: bool = False,
    tsae_interface_cols_per_side: int = 1,
    tsae_interface_joint_score: bool = False,
    tsae_interface_zb_weight: float = 1.0,
    tsae_interface_commit_offset: int = 0,
    a_shifted_chain_dp: bool = False,
    a_shifted_joint_b_dp: bool = False,
    a_shifted_joint_b_dp_lag: int = 0,
    a_shifted_joint_flag_penalty: float = 1000.0,
    a_micro_sliding: bool = False,
    a_micro_sliding_order: str = "asc",
    oracle_diag: np.ndarray | None = None,
    z_boundary_repair: bool = False,
    z_repair_edge_width: int = 2,
    z_joint_retry: bool = False,
    seam_diagnostics: bool = False,
):
    shots = det_data.shape[0]
    total = np.zeros((shots, problem.chk.shape[1]), dtype=np.uint8)
    num_blocks = problem.num_detector_blocks
    num_regions = len(problem.region_offsets) - 1
    if n_a_solve < n_a:
        raise ValueError("n_a_solve must be >= n_a")
    if top_k_boundary > 1 and b_noisy_boundary:
        raise ValueError("top_k_boundary > 1 is not supported with b_noisy_boundary")
    if flag_triggered_single_flip and (
        oracle_errors is not None
        or soft_boundary_message
        or top_k_boundary > 1
        or b_noisy_boundary
    ):
        raise ValueError(
            "flag_triggered_single_flip requires decoded A, top_k_boundary=1, "
            "soft_boundary_message=False, and b_noisy_boundary=False"
        )
    if aba_boundary_feedback and (
        oracle_errors is not None
        or soft_boundary_message
        or top_k_boundary > 1
    ):
        raise ValueError(
            "aba_boundary_feedback requires decoded A, top_k_boundary=1, "
            "and soft_boundary_message=False"
        )
    if a_neighbor_rerank and (
        oracle_errors is not None
        or soft_boundary_message
        or top_k_boundary > 1
    ):
        raise ValueError(
            "a_neighbor_rerank requires decoded A, top_k_boundary=1, "
            "and soft_boundary_message=False"
        )
    if a_two_color_order not in ("none", "odd-first", "even-first"):
        raise ValueError("a_two_color_order must be one of: none, odd-first, even-first")
    if tsae_stitch_mode not in ("none", "component", "pairwise"):
        raise ValueError("tsae_stitch_mode must be one of: none, component, pairwise")
    if tsae_stitch_mode != "none" and not a_shifted_ensemble:
        raise ValueError("tsae_stitch_mode requires a_shifted_ensemble")
    if tsae_interface_branch and not a_shifted_ensemble:
        raise ValueError("tsae_interface_branch requires a_shifted_ensemble")
    if tsae_interface_branch and tsae_stitch_mode != "none":
        raise ValueError("tsae_interface_branch is mutually exclusive with tsae_stitch_mode")
    if tsae_interface_gated and not (tsae_interface_branch and a_shifted_joint_b_dp):
        raise ValueError("tsae_interface_gated requires tsae_interface_branch and a_shifted_joint_b_dp")
    if tsae_interface_cols_per_side < 1:
        raise ValueError("tsae_interface_cols_per_side must be at least 1")
    if tsae_interface_joint_score and not tsae_interface_branch:
        raise ValueError("tsae_interface_joint_score requires tsae_interface_branch")
    if tsae_interface_joint_score and tsae_interface_zb_weight < 0:
        raise ValueError("tsae_interface_zb_weight must be non-negative")
    if a_shifted_chain_dp and not a_shifted_ensemble:
        raise ValueError("a_shifted_chain_dp requires a_shifted_ensemble")
    if a_shifted_chain_dp and tsae_stitch_mode != "none":
        raise ValueError("a_shifted_chain_dp is mutually exclusive with tsae_stitch_mode")
    if a_shifted_joint_b_dp and not a_shifted_ensemble:
        raise ValueError("a_shifted_joint_b_dp requires a_shifted_ensemble")
    if a_shifted_joint_b_dp and (a_shifted_chain_dp or tsae_stitch_mode != "none"):
        raise ValueError("a_shifted_joint_b_dp is mutually exclusive with a_shifted_chain_dp and tsae_stitch_mode")
    if a_shifted_joint_b_dp_lag < 0:
        raise ValueError("a_shifted_joint_b_dp_lag must be non-negative")
    if a_shifted_joint_b_dp_lag and not a_shifted_joint_b_dp:
        raise ValueError("a_shifted_joint_b_dp_lag requires a_shifted_joint_b_dp")
    if a_shifted_joint_b_dp and a_shifted_joint_flag_penalty < 0:
        raise ValueError("a_shifted_joint_flag_penalty must be non-negative")
    if a_micro_sliding and not a_shifted_ensemble:
        raise ValueError("a_micro_sliding requires a_shifted_ensemble")
    if a_micro_sliding and (tsae_stitch_mode != "none" or a_shifted_chain_dp or a_shifted_joint_b_dp):
        raise ValueError("a_micro_sliding is mutually exclusive with tsae_stitch_mode and shifted DP modes")
    if tsae_interface_branch and a_micro_sliding:
        raise ValueError("tsae_interface_branch is mutually exclusive with a_micro_sliding")
    if a_micro_sliding_order not in ("asc", "desc", "center-out"):
        raise ValueError("a_micro_sliding_order must be one of: asc, desc, center-out")
    if a_two_color_order != "none" and (
        oracle_errors is not None
        or soft_boundary_message
        or top_k_boundary > 1
        or flag_triggered_single_flip
        or aba_boundary_feedback
        or a_neighbor_rerank
        or a_shifted_ensemble
    ):
        raise ValueError(
            "a_two_color_order requires decoded A, top_k_boundary=1, "
            "soft_boundary_message=False, and no A/B repair or rerank modes"
        )
    if two_flip_candidates < 2:
        raise ValueError("two_flip_candidates must be at least 2")
    if a_boundary_weight_scale <= 0:
        raise ValueError("a_boundary_weight_scale must be positive")
    if aba_max_flips < 1:
        raise ValueError("aba_max_flips must be at least 1")
    if aba_candidate_cols < 1:
        raise ValueError("aba_candidate_cols must be at least 1")
    if a_rerank_top_k < 2:
        raise ValueError("a_rerank_top_k must be at least 2")
    if tsae_top_k < 1:
        raise ValueError("tsae_top_k must be at least 1")
    if tsae_boundary_top_k < 1:
        raise ValueError("tsae_boundary_top_k must be at least 1")
    if a_neighbor_beta < 0:
        raise ValueError("a_neighbor_beta must be non-negative")
    if a_shifted_beta < 0:
        raise ValueError("a_shifted_beta must be non-negative")
    if not a_shift_offsets:
        raise ValueError("a_shift_offsets must not be empty")
    if z_repair_edge_width < 0:
        raise ValueError("z_repair_edge_width must be non-negative")
    if z_boundary_repair and z_joint_retry:
        raise ValueError("z_boundary_repair and z_joint_retry are mutually exclusive")
    use_buffer_aligned = n_a == 3 and n_a_solve == n_b and n_a_solve % 2 == 1
    if (z_boundary_repair or z_joint_retry) and (
        oracle_errors is not None
        or not use_buffer_aligned
        or not b_noisy_boundary
        or soft_boundary_message
        or top_k_boundary > 1
        or flag_triggered_single_flip
        or aba_boundary_feedback
    ):
        raise ValueError(
            "z repair/retry requires decoded buffer-aligned A/B, "
            "b_noisy_boundary=True, top_k_boundary=1, soft_boundary_message=False, "
            "and no flag/ABA repair modes"
        )
    if a_shifted_ensemble and (
        oracle_errors is not None
        or not use_buffer_aligned
        or soft_boundary_message
        or top_k_boundary > 1
        or flag_triggered_single_flip
        or aba_boundary_feedback
        or a_neighbor_rerank
        or a_two_color_order != "none"
    ):
        raise ValueError(
            "a_shifted_ensemble requires decoded buffer-aligned A/B, "
            "top_k_boundary=1, soft_boundary_message=False, and no A repair/rerank/two-color modes"
        )
    if use_buffer_aligned:
        buffer = (n_a_solve - 1) // 2
        step = n_a_solve + 1

        b_starts = []
        b_start = 1
        while b_start < num_blocks:
            b_starts.append(b_start)
            b_start += step

        a_starts = [0]
        next_a = buffer + 2
        while next_a < num_blocks:
            a_starts.append(next_a)
            next_a += step

        a_tasks = []
        for index, a_start in enumerate(a_starts):
            if index == 0:
                row_start = 0
                row_stop = min(buffer + 1, num_blocks)
                col_start = 0
                col_stop = min(2 * row_stop, num_regions)
                commit_start = 0
                commit_stop = min(2, num_regions)
            else:
                row_start = a_start
                row_stop = min(row_start + n_a_solve, num_blocks)
                col_start = max(0, 2 * row_start - 1)
                col_stop = min(2 * row_stop, num_regions)
                commit_start = min(2 * row_start + 2 * buffer - 1, num_regions)
                next_b_start = 1 + (index - 1) * step
                if next_b_start < num_blocks:
                    commit_stop = min(commit_start + n_a, num_regions)
                else:
                    commit_stop = num_regions

            if row_start >= row_stop or commit_start >= commit_stop:
                continue

            rows = row_slice(problem, row_start, row_stop)
            cols = region_col_slice(problem, col_start, col_stop)
            commit_cols = region_col_slice(problem, commit_start, commit_stop)
            a_tasks.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "commit_cols": commit_cols,
                    "row_label": s_range(row_start, row_stop),
                    "commit_label": f"e{commit_start}-e{commit_stop - 1}",
                }
            )

        b_tasks = []
        for b_start in b_starts:
            row_start = b_start
            row_stop = min(row_start + n_b, num_blocks)
            col_start = 2 * row_start
            col_stop = min(2 * row_stop - 1, num_regions)
            if row_start >= row_stop or col_start >= col_stop:
                continue
            rows = row_slice(problem, row_start, row_stop)
            cols = region_col_slice(problem, col_start, col_stop)
            b_tasks.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "row_label": s_range(row_start, row_stop),
                    "commit_label": f"e{col_start}-e{col_stop - 1}",
                }
            )
    else:
        left_buffer = (n_a_solve - n_a) // 2
        right_buffer = n_a_solve - n_a - left_buffer
        step = n_a + n_b - 2

        b_starts = []
        right_a_starts = set()
        b_start = n_a - 2
        while b_start >= 0 and b_start + n_b <= num_blocks:
            right_a_start = b_start + n_b - 1
            if right_a_start >= num_blocks:
                break
            b_starts.append(b_start)
            right_a_starts.add(right_a_start)
            b_start += step

        a_starts = [0]
        next_a = n_b
        while next_a < num_blocks:
            a_starts.append(next_a)
            next_a += step

        a_tasks = []
        for index, a_start in enumerate(a_starts):
            if index == 0:
                row_start = 0
                row_stop = min(n_a - 1 + right_buffer, num_blocks)
                col_start = 0
                col_stop = min(2 * row_stop, num_regions)
                commit_start = 0
                commit_stop = min(2, num_regions)
            else:
                next_b_start = a_start + n_a - 1
                is_final_a = next_b_start not in b_starts
                row_start = max(0, a_start - left_buffer)
                row_stop = (
                    num_blocks
                    if is_final_a
                    else min(a_start + n_a + right_buffer, num_blocks)
                )
                col_start = max(0, 2 * row_start - 1)
                col_stop = min(2 * row_stop, num_regions)
                commit_start = min(2 * a_start + 1, num_regions)
                if not is_final_a:
                    commit_stop = min(commit_start + n_a, num_regions)
                else:
                    commit_stop = num_regions

            if row_start >= row_stop or commit_start >= commit_stop:
                continue

            rows = row_slice(problem, row_start, row_stop)
            cols = region_col_slice(problem, col_start, col_stop)
            commit_cols = region_col_slice(problem, commit_start, commit_stop)
            a_tasks.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "commit_cols": commit_cols,
                    "row_label": s_range(row_start, row_stop),
                    "commit_label": f"e{commit_start}-e{commit_stop - 1}",
                }
            )

        b_tasks = []
        for b_start in b_starts:
            row_start = b_start
            row_stop = b_start + n_b
            left_boundary = 2 * row_start - 1
            right_boundary = 2 * (row_stop - 1) + 1
            col_start = left_boundary + 1
            col_stop = right_boundary
            rows = row_slice(problem, row_start, row_stop)
            cols = region_col_slice(problem, col_start, col_stop)
            b_tasks.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "row_label": s_range(row_start, row_stop),
                    "commit_label": f"e{col_start}-e{col_stop - 1}",
                }
            )

    a_boundary_penalty_cols: set[int] = set()
    if a_boundary_weight_scale != 1.0 and use_buffer_aligned:
        for b_index, task in enumerate(b_tasks):
            rows = task["rows"]
            for a_index in (b_index, b_index + 1):
                if a_index >= len(a_tasks):
                    continue
                commit_cols = a_tasks[a_index]["commit_cols"]
                global_cols = np.arange(commit_cols.start, commit_cols.stop)
                if global_cols.size == 0:
                    continue
                affects = np.asarray(problem.chk[rows, global_cols].sum(axis=0)).reshape(-1) > 0
                a_boundary_penalty_cols.update(int(col) for col in global_cols[affects])

    def build_a_payload(task_index: int, task: dict, syndrome_source: np.ndarray) -> dict:
        rows = task["rows"]
        cols = task["cols"]
        commit_cols = task["commit_cols"]
        mat = problem.chk[rows, cols]
        prior = problem.priors[cols].copy()
        if a_boundary_penalty_cols:
            global_cols = np.arange(cols.start, cols.stop)
            boundary_mask = np.isin(global_cols, list(a_boundary_penalty_cols))
            prior = scale_priors_by_llr(prior, boundary_mask, a_boundary_weight_scale)
        if a_noisy_boundary:
            mat, prior = add_noisy_boundary_columns(
                problem,
                mat,
                prior,
                rows,
                cols,
                boundary_width=problem.detector_block_size,
            )

        local_commit_start = commit_cols.start - cols.start
        local_commit_stop = commit_cols.stop - cols.start
        return {
            "task_index": task_index,
            "mat": mat,
            "prior": prior,
            "syndromes": syndrome_source[:, rows],
            "max_iter": max_iter,
            "osd_order": osd_order,
            "shorten": window_shorten,
            "shorten_pre_max_iter": shorten_pre_max_iter,
            "value_start": local_commit_start,
            "value_stop": local_commit_stop,
        }

    a_payloads = [
        build_a_payload(task_index, task, det_data)
        for task_index, task in enumerate(a_tasks)
    ]

    a_flagged = 0
    b_flagged = 0
    a_commit_reliability = None
    a_neighbor_rerank_selected_nonzero = 0
    a_neighbor_rerank_total_risk = 0.0
    a_neighbor_rerank_total_cost = 0.0
    a_shifted_selected_nonzero = 0
    a_shifted_total_risk = 0.0
    a_shifted_total_cost = 0.0
    a_shifted_candidate_count = 0
    a_shifted_valid_offsets: dict[int, list[int]] = {}
    tsae_stitch_attempts = 0
    tsae_stitch_accepted = 0
    tsae_stitch_fallback = 0
    tsae_stitch_non_payload = 0
    tsae_stitch_total_score = 0.0
    tsae_interface_branch_tasks = 0
    tsae_interface_branch_payloads = 0
    tsae_interface_branch_cols = 0
    tsae_interface_gated_shots = 0
    tsae_interface_gated_accepted = 0
    tsae_interface_gated_improved_to_zero = 0
    tsae_interface_gated_branch_payloads = 0
    tsae_interface_gated_b_decodes = 0
    joint_branch_groups_total = 0
    joint_branch_b_star_nontrivial = 0
    joint_branch_chose_center_offset = 0
    chain_dp_shots = 0
    chain_dp_selected_nonzero = 0
    chain_dp_final_total_risk = 0.0
    chain_dp_total_pair_weight = 0.0
    joint_b_dp_decoded = False
    joint_b_dp_shots = 0
    joint_b_dp_b_decodes = 0
    joint_b_dp_physical_residual = 0
    joint_b_dp_selected_cost = 0.0
    joint_b_dp_selected_nonzero = 0
    tsae_diag_in_pool = None
    tsae_diag_closest_hamming = None
    tsae_diag_pool_size = None
    tsae_diag_picked_offset = None
    tsae_diag_oracle_offset = None
    a_two_color_first_flagged = 0
    a_two_color_second_flagged = 0
    if oracle_errors is not None:
        for task in a_tasks:
            total[:, task["commit_cols"]] = oracle_errors[:, task["commit_cols"]]
    elif a_two_color_order != "none" and use_buffer_aligned:
        if a_two_color_order == "odd-first":
            first_indices = list(range(0, len(a_tasks), 2))
            second_indices = list(range(1, len(a_tasks), 2))
        else:
            first_indices = list(range(1, len(a_tasks), 2))
            second_indices = list(range(0, len(a_tasks), 2))

        residual_for_a = det_data.copy()
        for stage, indices in enumerate((first_indices, second_indices)):
            stage_payloads = [
                build_a_payload(task_index, a_tasks[task_index], residual_for_a)
                for task_index in indices
            ]
            stage_flagged = 0
            for task_index, commit_values, flagged in run_window_payloads(
                stage_payloads,
                parallel_workers,
                parallel_backend,
            ):
                total[:, a_tasks[task_index]["commit_cols"]] = commit_values
                stage_flagged += flagged
            if stage == 0:
                a_two_color_first_flagged = stage_flagged
            else:
                a_two_color_second_flagged = stage_flagged
            a_flagged += stage_flagged
            residual_for_a = (det_data + total @ problem.chk.T) % 2
        a_candidate_values = None
        a_candidate_costs = None
    elif flag_triggered_single_flip and use_buffer_aligned:
        a_commit_reliability = np.full((shots, problem.chk.shape[1]), np.inf, dtype=float)
        for task_index, commit_values, commit_reliability, flagged in run_a_repair_payloads(
            a_payloads,
            parallel_workers,
            parallel_backend,
        ):
            commit_cols = a_tasks[task_index]["commit_cols"]
            total[:, commit_cols] = commit_values
            a_commit_reliability[:, commit_cols] = commit_reliability
            a_flagged += flagged
        a_candidate_values = None
        a_candidate_costs = None
    elif a_neighbor_rerank and use_buffer_aligned:
        candidate_payloads = []
        for payload in a_payloads:
            candidate_payload = dict(payload)
            candidate_payload["top_k"] = a_rerank_top_k
            candidate_payloads.append(candidate_payload)
        a_candidate_values = [None] * len(a_tasks)
        a_candidate_costs = [None] * len(a_tasks)
        for task_index, values, costs, flagged in run_a_candidate_payloads(
            candidate_payloads,
            parallel_workers,
            parallel_backend,
        ):
            a_candidate_values[task_index] = values
            a_candidate_costs[task_index] = costs
            a_flagged += flagged

        for a_index, task in enumerate(a_tasks):
            commit_cols = task["commit_cols"]
            neighbor_b = []
            if a_index - 1 >= 0 and a_index - 1 < len(b_tasks):
                neighbor_b.append(a_index - 1)
            if a_index < len(b_tasks):
                neighbor_b.append(a_index)

            for shot in range(shots):
                best_k = 0
                best_score = np.inf
                best_risk = 0
                for cand in range(a_rerank_top_k):
                    candidate = a_candidate_values[a_index][shot, cand]
                    risk = 0
                    for b_index in neighbor_b:
                        rows = b_tasks[b_index]["rows"]
                        contribution = candidate @ problem.chk[rows, commit_cols].T
                        risk += int(((det_data[shot, rows] + contribution) % 2).sum())
                    score = a_candidate_costs[a_index][shot, cand] + a_neighbor_beta * risk
                    if score < best_score:
                        best_score = score
                        best_k = cand
                        best_risk = risk
                total[shot, commit_cols] = a_candidate_values[a_index][shot, best_k]
                a_neighbor_rerank_selected_nonzero += int(best_k != 0)
                a_neighbor_rerank_total_risk += float(best_risk)
                a_neighbor_rerank_total_cost += float(best_score)
        a_candidate_values = None
        a_candidate_costs = None
    elif a_shifted_ensemble and use_buffer_aligned:
        candidate_payloads = []
        payload_to_a: list[int] = []
        payload_offsets: list[int] = []

        def shifted_a_payload(
            task_index: int,
            task: dict,
            offset: int,
            payload_index: int,
        ) -> dict | None:
            base_rows = task["rows"]
            row_start = base_rows.start // problem.detector_block_size + offset
            row_count = (base_rows.stop - base_rows.start) // problem.detector_block_size
            row_stop = row_start + row_count
            if row_start < 0 or row_stop > num_blocks:
                return None
            col_start = max(0, 2 * row_start - 1)
            col_stop = min(2 * row_stop, num_regions)
            cols = region_col_slice(problem, col_start, col_stop)
            commit_cols = task["commit_cols"]
            if commit_cols.start < cols.start or commit_cols.stop > cols.stop:
                return None

            rows = row_slice(problem, row_start, row_stop)
            mat = problem.chk[rows, cols]
            prior = problem.priors[cols].copy()
            if a_boundary_penalty_cols:
                global_cols = np.arange(cols.start, cols.stop)
                boundary_mask = np.isin(global_cols, list(a_boundary_penalty_cols))
                prior = scale_priors_by_llr(prior, boundary_mask, a_boundary_weight_scale)
            if a_noisy_boundary:
                mat, prior = add_noisy_boundary_columns(
                    problem,
                    mat,
                    prior,
                    rows,
                    cols,
                    boundary_width=problem.detector_block_size,
                )
            return {
                "task_index": payload_index,
                "mat": mat,
                "prior": prior,
                "syndromes": det_data[:, rows],
                "max_iter": max_iter,
                "osd_order": osd_order,
                "shorten": window_shorten,
                "shorten_pre_max_iter": shorten_pre_max_iter,
                "value_start": commit_cols.start - cols.start,
                "value_stop": commit_cols.stop - cols.start,
                "top_k": (
                    tsae_top_k
                    if 0 < task_index < len(a_tasks) - 1
                    else tsae_boundary_top_k
                ),
                "rows_slice": rows,
                "cols_slice": cols,
                "real_col_count": cols.stop - cols.start,
            }

        def select_interface_branch_cols(a_index: int, task: dict) -> list[int]:
            if not (0 < a_index < len(a_tasks) - 1):
                return []
            commit_cols = task["commit_cols"]
            commit_region_start = int(np.searchsorted(problem.region_offsets, commit_cols.start))
            commit_region_stop = int(np.searchsorted(problem.region_offsets, commit_cols.stop))
            if commit_region_stop - commit_region_start < 3:
                return []

            selected: list[int] = []
            side_specs = []
            if a_index - 1 < len(b_tasks):
                side_specs.append((commit_region_start, b_tasks[a_index - 1]["rows"]))
            if a_index < len(b_tasks):
                side_specs.append((commit_region_stop - 1, b_tasks[a_index]["rows"]))

            for region, rows in side_specs:
                region_cols = region_col_slice(problem, region, region + 1)
                start = max(region_cols.start, commit_cols.start)
                stop = min(region_cols.stop, commit_cols.stop)
                if start >= stop:
                    continue
                global_cols = np.arange(start, stop)
                local = np.asarray(problem.chk[rows, global_cols], dtype=np.uint8)
                support = local.sum(axis=0).astype(float)
                prior = problem.priors[global_cols].astype(float)
                order = np.lexsort((-prior, -support))
                side_selected = [
                    int(global_cols[idx])
                    for idx in order[: min(tsae_interface_cols_per_side, order.size)]
                ]
                selected.extend(side_selected)
            return selected

        interface_cols_by_a: list[list[int]] = []
        for a_index, task in enumerate(a_tasks):
            interface_cols = (
                select_interface_branch_cols(a_index, task)
                if tsae_interface_branch
                else []
            )
            interface_cols_by_a.append(interface_cols)
            if interface_cols:
                tsae_interface_branch_tasks += 1
                tsae_interface_branch_cols += len(interface_cols)
            seen_offsets: set[int] = set()
            for offset in a_shift_offsets:
                if offset in seen_offsets:
                    continue
                seen_offsets.add(int(offset))
                payload = shifted_a_payload(a_index, task, int(offset), len(candidate_payloads))
                if payload is None:
                    continue
                if interface_cols and not tsae_interface_gated:
                    branch_positions = [
                        col - payload["cols_slice"].start
                        for col in interface_cols
                        if payload["cols_slice"].start <= col < payload["cols_slice"].stop
                    ]
                    if len(branch_positions) != len(interface_cols):
                        continue
                    for bits in itertools.product((0, 1), repeat=len(branch_positions)):
                        branched_payload = dict(payload)
                        branched_payload["task_index"] = len(candidate_payloads)
                        branched_payload["fixed_value_positions"] = branch_positions
                        branched_payload["fixed_value_bits"] = list(bits)
                        candidate_payloads.append(branched_payload)
                        payload_to_a.append(a_index)
                        payload_offsets.append(int(offset))
                        tsae_interface_branch_payloads += 1
                else:
                    candidate_payloads.append(payload)
                    payload_to_a.append(a_index)
                    payload_offsets.append(int(offset))
                a_shifted_valid_offsets.setdefault(a_index, []).append(int(offset))
            if a_index not in a_shifted_valid_offsets:
                payload = shifted_a_payload(a_index, task, 0, len(candidate_payloads))
                if payload is None:
                    raise ValueError(f"no valid shifted A payload for task {a_index}")
                candidate_payloads.append(payload)
                payload_to_a.append(a_index)
                payload_offsets.append(0)
                a_shifted_valid_offsets[a_index] = [0]

        if a_micro_sliding:
            payloads_by_a: list[list[int]] = [[] for _ in a_tasks]
            for payload_index, a_index in enumerate(payload_to_a):
                payloads_by_a[a_index].append(payload_index)

            def micro_order(payload_indices):
                offsets = [payload_offsets[p] for p in payload_indices]
                order_pairs = list(zip(offsets, payload_indices))
                if a_micro_sliding_order == "asc":
                    order_pairs.sort(key=lambda x: x[0])
                elif a_micro_sliding_order == "desc":
                    order_pairs.sort(key=lambda x: -x[0])
                else:  # center-out
                    order_pairs.sort(key=lambda x: abs(x[0]))
                return [p for _, p in order_pairs]

            decoders_per_payload = []
            for payload in candidate_payloads:
                decoders_per_payload.append(
                    make_bp_osd(
                        payload["mat"], payload["prior"],
                        max_iter, osd_order, window_shorten, shorten_pre_max_iter,
                    )
                )

            for shot in range(shots):
                for a_index in range(len(a_tasks)):
                    commit_cols = a_tasks[a_index]["commit_cols"]
                    payload_indices_ordered = micro_order(payloads_by_a[a_index])

                    if len(payload_indices_ordered) == 1:
                        p_idx = payload_indices_ordered[0]
                        payload = candidate_payloads[p_idx]
                        rows = payload["rows_slice"]
                        syndrome = np.asarray(det_data[shot, rows], dtype=np.uint8)
                        e_local = decoders_per_payload[p_idx].decode(syndrome).astype(np.uint8)
                        if ((payload["mat"] @ e_local + syndrome) % 2).any():
                            a_flagged += 1
                        v_start, v_stop = payload["value_start"], payload["value_stop"]
                        total[shot, commit_cols] = e_local[v_start:v_stop]
                        a_shifted_candidate_count += 1
                    else:
                        residual = np.asarray(det_data[shot], dtype=np.uint8).copy()
                        last_commit = None
                        for i, p_idx in enumerate(payload_indices_ordered):
                            payload = candidate_payloads[p_idx]
                            rows = payload["rows_slice"]
                            cols = payload["cols_slice"]
                            n_real = payload["real_col_count"]
                            v_start = payload["value_start"]
                            v_stop = payload["value_stop"]
                            local_syndrome = np.asarray(residual[rows], dtype=np.uint8)
                            e_local = decoders_per_payload[p_idx].decode(local_syndrome).astype(np.uint8)
                            if ((payload["mat"] @ e_local + local_syndrome) % 2).any():
                                a_flagged += 1

                            if i < len(payload_indices_ordered) - 1:
                                masked = e_local[:n_real].copy()
                                masked[v_start:v_stop] = 0
                                contribution = (np.asarray(problem.chk[:, cols]) @ masked) & 1
                                residual = (residual ^ contribution.astype(np.uint8))
                            last_commit = e_local[v_start:v_stop]
                            a_shifted_candidate_count += 1

                        total[shot, commit_cols] = last_commit

            a_candidate_values = None
            a_candidate_costs = None
            shifted_values = None
            shifted_costs = None
        else:
            shifted_values = []
            shifted_costs = []
            expanded_payload_to_a: list[int] = []
            expanded_payload_offsets: list[int] = []
            expanded_branch_bits: list[tuple[int, ...]] = []
            for payload_index, values, costs, flagged in run_a_candidate_payloads(
                candidate_payloads,
                parallel_workers,
                parallel_backend,
            ):
                bits_tuple = tuple(candidate_payloads[payload_index].get("fixed_value_bits", ()))
                for cand_index in range(values.shape[1]):
                    shifted_values.append(values[:, cand_index, :])
                    shifted_costs.append(costs[:, cand_index])
                    expanded_payload_to_a.append(payload_to_a[payload_index])
                    expanded_payload_offsets.append(payload_offsets[payload_index])
                    expanded_branch_bits.append(bits_tuple)
                a_flagged += flagged

            payload_to_a = expanded_payload_to_a
            payload_offsets = expanded_payload_offsets
            payload_branch_bits = expanded_branch_bits
            payloads_by_a: list[list[int]] = [[] for _ in a_tasks]
            for payload_index, a_index in enumerate(payload_to_a):
                payloads_by_a[a_index].append(payload_index)


            if a_shifted_chain_dp:
                M = len(a_tasks)
                b_left_chk: list[np.ndarray | None] = [None] * len(b_tasks)
                b_right_chk: list[np.ndarray | None] = [None] * len(b_tasks)
                for b_index, b_task in enumerate(b_tasks):
                    rows = b_task["rows"]
                    if b_index < M:
                        b_left_chk[b_index] = np.ascontiguousarray(
                            problem.chk[rows, a_tasks[b_index]["commit_cols"]],
                            dtype=np.uint8,
                        )
                    if b_index + 1 < M:
                        b_right_chk[b_index] = np.ascontiguousarray(
                            problem.chk[rows, a_tasks[b_index + 1]["commit_cols"]],
                            dtype=np.uint8,
                        )

                a_payload_lists = [payloads_by_a[a_index] for a_index in range(M)]

                for shot in range(shots):
                    K0 = len(a_payload_lists[0])
                    f_prev = np.empty(K0, dtype=np.float64)
                    for k in range(K0):
                        f_prev[k] = shifted_costs[a_payload_lists[0][k]][shot]
                    backpointers: list[np.ndarray] = []

                    for i in range(1, M):
                        K_curr = len(a_payload_lists[i])
                        K_prev = len(a_payload_lists[i - 1])
                        b_index = i - 1
                        if b_index < len(b_tasks) and b_left_chk[b_index] is not None and b_right_chk[b_index] is not None:
                            rows = b_tasks[b_index]["rows"]
                            s_b = np.asarray(det_data[shot, rows], dtype=np.uint8)
                            left_chk = b_left_chk[b_index]
                            right_chk = b_right_chk[b_index]
                            left_contribs = np.empty((K_prev, s_b.size), dtype=np.uint8)
                            for k_prev_idx, payload_prev in enumerate(a_payload_lists[i - 1]):
                                cand = np.asarray(shifted_values[payload_prev][shot], dtype=np.uint8)
                                left_contribs[k_prev_idx] = (left_chk @ cand) & 1
                            right_contribs = np.empty((K_curr, s_b.size), dtype=np.uint8)
                            for k_curr_idx, payload_curr in enumerate(a_payload_lists[i]):
                                cand = np.asarray(shifted_values[payload_curr][shot], dtype=np.uint8)
                                right_contribs[k_curr_idx] = (right_chk @ cand) & 1
                            # pair_weight[k_prev, k_curr] = popcount(s_b XOR left[k_prev] XOR right[k_curr])
                            combined = s_b[None, None, :] ^ left_contribs[:, None, :] ^ right_contribs[None, :, :]
                            pair_weight = combined.sum(axis=2, dtype=np.float64)
                        else:
                            pair_weight = np.zeros((K_prev, K_curr), dtype=np.float64)

                        local_costs_curr = np.fromiter(
                            (shifted_costs[payload_idx][shot] for payload_idx in a_payload_lists[i]),
                            dtype=np.float64,
                            count=K_curr,
                        )
                        transition = f_prev[:, None] + a_shifted_beta * pair_weight
                        best_prev = np.argmin(transition, axis=0)
                        f_curr = transition[best_prev, np.arange(K_curr)] + local_costs_curr
                        backpointers.append(best_prev.astype(np.int64))
                        f_prev = f_curr

                    chosen_k = [0] * M
                    chosen_k[M - 1] = int(np.argmin(f_prev))
                    for i in range(M - 1, 0, -1):
                        chosen_k[i - 1] = int(backpointers[i - 1][chosen_k[i]])

                    pair_weight_accumulated = 0
                    for i, k in enumerate(chosen_k):
                        payload_idx = a_payload_lists[i][k]
                        commit_cols = a_tasks[i]["commit_cols"]
                        total[shot, commit_cols] = shifted_values[payload_idx][shot]
                        chain_dp_selected_nonzero += int(payload_offsets[payload_idx] != 0)
                        a_shifted_selected_nonzero += int(payload_offsets[payload_idx] != 0)
                        a_shifted_total_cost += float(shifted_costs[payload_idx][shot])
                        a_shifted_candidate_count += len(a_payload_lists[i])

                    final_risk = 0
                    for b_index in range(min(len(b_tasks), M)):
                        rows = b_tasks[b_index]["rows"]
                        if b_left_chk[b_index] is None or b_right_chk[b_index] is None:
                            continue
                        s_b = np.asarray(det_data[shot, rows], dtype=np.uint8)
                        cand_left = np.asarray(
                            shifted_values[a_payload_lists[b_index][chosen_k[b_index]]][shot],
                            dtype=np.uint8,
                        )
                        cand_right = np.asarray(
                            shifted_values[a_payload_lists[b_index + 1][chosen_k[b_index + 1]]][shot],
                            dtype=np.uint8,
                        )
                        contrib = s_b ^ ((b_left_chk[b_index] @ cand_left) & 1) ^ ((b_right_chk[b_index] @ cand_right) & 1)
                        final_risk += int(contrib.sum())
                    a_shifted_total_risk += float(final_risk)
                    chain_dp_final_total_risk += float(final_risk)
                    chain_dp_shots += 1

                a_candidate_values = None
                a_candidate_costs = None
            elif a_shifted_joint_b_dp:
                M = len(a_tasks)
                if len(b_tasks) != max(0, M - 1):
                    raise ValueError("a_shifted_joint_b_dp expects one B window between each adjacent A pair")

                a_payload_lists = [payloads_by_a[a_index] for a_index in range(M)]
                b_specs = []
                for b_index, b_task in enumerate(b_tasks):
                    rows = b_task["rows"]
                    cols = b_task["cols"]
                    mat = problem.chk[rows, cols]
                    prior = problem.priors[cols]
                    physical_width = cols.stop - cols.start
                    if b_noisy_boundary:
                        mat, prior = add_noisy_boundary_columns(
                            problem,
                            mat,
                            prior,
                            rows,
                            cols,
                            boundary_width=problem.detector_block_size,
                        )
                    decoder = make_bp_osd(
                        mat,
                        prior,
                        max_iter,
                        osd_order,
                        window_shorten,
                        shorten_pre_max_iter,
                    )
                    b_specs.append(
                        {
                            "rows": rows,
                            "cols": cols,
                            "mat": mat,
                            "prior": prior,
                            "physical_mat": problem.chk[rows, cols],
                            "physical_width": physical_width,
                            "decoder": decoder,
                            "left_chk": np.ascontiguousarray(
                                problem.chk[rows, a_tasks[b_index]["commit_cols"]],
                                dtype=np.uint8,
                            ),
                            "right_chk": np.ascontiguousarray(
                                problem.chk[rows, a_tasks[b_index + 1]["commit_cols"]],
                                dtype=np.uint8,
                            ),
                        }
                    )

                joint_window_flagged = np.zeros((shots, len(b_specs)), dtype=bool)
                joint_residual_by_shot = np.zeros(shots, dtype=np.int32)
                joint_cost_by_shot = np.full(shots, np.inf, dtype=np.float64)
                joint_nonzero_by_shot = np.zeros(shots, dtype=np.int32)

                def apply_joint_b_dp(
                    local_a_payload_lists: list[list[int]],
                    local_shifted_values: list[np.ndarray],
                    local_shifted_costs: list[np.ndarray],
                    local_payload_offsets: list[int],
                    shot_indices: np.ndarray,
                    gated_pass: bool = False,
                ) -> tuple[list[tuple[int, float, int, int, list[bool]]], int]:
                    nonlocal joint_b_dp_b_decodes, tsae_interface_gated_b_decodes
                    if M == 0 or shot_indices.size == 0:
                        return [], 0

                    candidate_count = int(shot_indices.size) * sum(
                        len(items) for items in local_a_payload_lists
                    )
                    results = []

                    def best_state(costs: np.ndarray) -> int:
                        finite = np.isfinite(costs)
                        if finite.any():
                            return int(np.argmin(costs))
                        return 0

                    def trace_state(
                        parents: list[np.ndarray],
                        endpoint_state: int,
                        current_a: int,
                        target_a: int,
                    ) -> int:
                        state = int(endpoint_state)
                        for edge_index in range(current_a - 1, target_a - 1, -1):
                            state = int(parents[edge_index][state])
                        return state

                    def select_full_path(
                        a_local_costs: list[np.ndarray],
                        edge_costs: list[np.ndarray],
                    ) -> tuple[list[int], float]:
                        dp = a_local_costs[0].copy()
                        parents: list[np.ndarray] = []
                        for edge_index, costs in enumerate(edge_costs):
                            transition = dp[:, None] + costs
                            best_prev = np.argmin(transition, axis=0)
                            dp = (
                                transition[best_prev, np.arange(costs.shape[1])]
                                + a_local_costs[edge_index + 1]
                            )
                            parents.append(best_prev.astype(np.int64))

                        chosen = [0] * M
                        if parents:
                            chosen[M - 1] = best_state(dp)
                            for edge_index in range(len(parents) - 1, -1, -1):
                                chosen[edge_index] = int(parents[edge_index][chosen[edge_index + 1]])
                            final_cost = float(dp[chosen[M - 1]])
                        else:
                            chosen[0] = best_state(dp)
                            final_cost = float(dp[chosen[0]])
                        return chosen, final_cost

                    def select_frontier_path(
                        a_local_costs: list[np.ndarray],
                        edge_costs: list[np.ndarray],
                        lag: int,
                    ) -> tuple[list[int], float]:
                        if M == 1:
                            chosen0 = best_state(a_local_costs[0])
                            return [chosen0], float(a_local_costs[0][chosen0])

                        dp = a_local_costs[0].copy()
                        parents: list[np.ndarray] = []
                        chosen = [-1] * M
                        last_committed = -1
                        current_a = 0

                        def prune_to(target_a: int, target_state: int) -> None:
                            if target_a >= current_a:
                                return
                            for endpoint_state in range(dp.size):
                                traced = trace_state(parents, endpoint_state, current_a, target_a)
                                if traced != target_state:
                                    dp[endpoint_state] = np.inf

                        for edge_index, costs in enumerate(edge_costs):
                            current_a = edge_index + 1
                            transition = dp[:, None] + costs
                            best_prev = np.argmin(transition, axis=0)
                            dp = (
                                transition[best_prev, np.arange(costs.shape[1])]
                                + a_local_costs[current_a]
                            )
                            parents.append(best_prev.astype(np.int64))

                            target_a = current_a - lag
                            if target_a >= 0 and target_a > last_committed:
                                endpoint_state = best_state(dp)
                                target_state = trace_state(parents, endpoint_state, current_a, target_a)
                                chosen[target_a] = target_state
                                last_committed = target_a
                                prune_to(target_a, target_state)

                        for target_a in range(last_committed + 1, M):
                            endpoint_state = best_state(dp)
                            target_state = trace_state(parents, endpoint_state, current_a, target_a)
                            chosen[target_a] = target_state
                            last_committed = target_a
                            prune_to(target_a, target_state)

                        final_cost = 0.0
                        for a_index, state in enumerate(chosen):
                            final_cost += float(a_local_costs[a_index][state])
                        for edge_index, costs in enumerate(edge_costs):
                            final_cost += float(costs[chosen[edge_index], chosen[edge_index + 1]])
                        return [int(state) for state in chosen], final_cost

                    for local_pos, shot in enumerate(shot_indices):
                        edge_costs = []
                        edge_values = []
                        edge_residuals = []
                        decodes_this_shot = 0
                        for b_index, spec in enumerate(b_specs):
                            left_list = local_a_payload_lists[b_index]
                            right_list = local_a_payload_lists[b_index + 1]
                            costs = np.full((len(left_list), len(right_list)), np.inf, dtype=np.float64)
                            residual_weights = np.zeros((len(left_list), len(right_list)), dtype=np.int32)
                            values: list[list[np.ndarray | None]] = [
                                [None for _ in right_list] for _ in left_list
                            ]
                            s_base = np.asarray(det_data[shot, spec["rows"]], dtype=np.uint8)
                            for left_idx, left_payload in enumerate(left_list):
                                left_candidate = np.asarray(
                                    local_shifted_values[left_payload][local_pos],
                                    dtype=np.uint8,
                                )
                                left_contrib = (spec["left_chk"] @ left_candidate) & 1
                                for right_idx, right_payload in enumerate(right_list):
                                    right_candidate = np.asarray(
                                        local_shifted_values[right_payload][local_pos],
                                        dtype=np.uint8,
                                    )
                                    right_contrib = (spec["right_chk"] @ right_candidate) & 1
                                    local_syndrome = (
                                        s_base ^ left_contrib.astype(np.uint8) ^ right_contrib.astype(np.uint8)
                                    )
                                    local_e = spec["decoder"].decode(local_syndrome).astype(np.uint8)
                                    physical_e = local_e[: spec["physical_width"]].copy()
                                    physical_residual = (
                                        spec["physical_mat"] @ physical_e + local_syndrome
                                    ) % 2
                                    physical_weight = int(physical_residual.sum())
                                    decode_residual = (spec["mat"] @ local_e + local_syndrome) % 2
                                    closure_weight = max(physical_weight, int(decode_residual.sum()))
                                    costs[left_idx, right_idx] = (
                                        error_cost(local_e, spec["prior"])
                                        + a_shifted_joint_flag_penalty * closure_weight
                                    )
                                    residual_weights[left_idx, right_idx] = closure_weight
                                    values[left_idx][right_idx] = physical_e
                                    decodes_this_shot += 1
                            edge_costs.append(costs)
                            edge_values.append(values)
                            edge_residuals.append(residual_weights)

                        joint_b_dp_b_decodes += decodes_this_shot
                        if gated_pass:
                            tsae_interface_gated_b_decodes += decodes_this_shot

                        a_local_costs = [
                            np.fromiter(
                                (
                                    local_shifted_costs[payload_idx][local_pos]
                                    for payload_idx in local_a_payload_lists[a_index]
                                ),
                                dtype=np.float64,
                                count=len(local_a_payload_lists[a_index]),
                            )
                            for a_index in range(M)
                        ]
                        if a_shifted_joint_b_dp_lag:
                            frontier_lag = min(a_shifted_joint_b_dp_lag, max(1, M - 1))
                            chosen, final_cost = select_frontier_path(
                                a_local_costs,
                                edge_costs,
                                frontier_lag,
                            )
                        else:
                            chosen, final_cost = select_full_path(a_local_costs, edge_costs)

                        selected_residual = 0
                        selected_nonzero = 0
                        window_flags = []
                        for a_index, state in enumerate(chosen):
                            payload_idx = local_a_payload_lists[a_index][state]
                            commit_cols = a_tasks[a_index]["commit_cols"]
                            total[shot, commit_cols] = local_shifted_values[payload_idx][local_pos]
                            selected_nonzero += int(local_payload_offsets[payload_idx] != 0)
                        for b_index, spec in enumerate(b_specs):
                            left_state = chosen[b_index]
                            right_state = chosen[b_index + 1]
                            b_value = edge_values[b_index][left_state][right_state]
                            if b_value is None:
                                raise AssertionError("missing B candidate value")
                            total[shot, spec["cols"]] = b_value
                            residual_weight = int(edge_residuals[b_index][left_state, right_state])
                            selected_residual += residual_weight
                            window_flags.append(residual_weight > 0)

                        results.append((int(shot), final_cost, selected_residual, selected_nonzero, window_flags))
                    return results, candidate_count

                base_results, base_candidate_count = apply_joint_b_dp(
                    a_payload_lists,
                    shifted_values,
                    shifted_costs,
                    payload_offsets,
                    np.arange(shots, dtype=int),
                )
                joint_b_dp_decoded = bool(base_results)
                joint_b_dp_shots += len(base_results)
                a_shifted_candidate_count += base_candidate_count
                for shot, final_cost, selected_residual, selected_nonzero, window_flags in base_results:
                    joint_cost_by_shot[shot] = final_cost
                    joint_residual_by_shot[shot] = selected_residual
                    joint_nonzero_by_shot[shot] = selected_nonzero
                    joint_window_flagged[shot, : len(window_flags)] = window_flags

                if tsae_interface_gated and tsae_interface_branch:
                    gated_shots = np.flatnonzero(joint_residual_by_shot > 0)
                    tsae_interface_gated_shots = int(gated_shots.size)
                    if gated_shots.size:
                        base_total = total[gated_shots].copy()
                        base_cost = joint_cost_by_shot[gated_shots].copy()
                        base_residual = joint_residual_by_shot[gated_shots].copy()
                        base_nonzero = joint_nonzero_by_shot[gated_shots].copy()
                        base_window_flags = joint_window_flagged[gated_shots].copy()

                        gated_candidate_payloads = []
                        gated_payload_to_a: list[int] = []
                        gated_payload_offsets: list[int] = []
                        for a_index, task in enumerate(a_tasks):
                            interface_cols = interface_cols_by_a[a_index]
                            seen_offsets: set[int] = set()
                            for offset in a_shift_offsets:
                                if offset in seen_offsets:
                                    continue
                                seen_offsets.add(int(offset))
                                payload = shifted_a_payload(
                                    a_index,
                                    task,
                                    int(offset),
                                    len(gated_candidate_payloads),
                                )
                                if payload is None:
                                    continue
                                payload = dict(payload)
                                payload["syndromes"] = payload["syndromes"][gated_shots]
                                if interface_cols:
                                    branch_positions = [
                                        col - payload["cols_slice"].start
                                        for col in interface_cols
                                        if payload["cols_slice"].start <= col < payload["cols_slice"].stop
                                    ]
                                    if len(branch_positions) != len(interface_cols):
                                        continue
                                    for bits in itertools.product((0, 1), repeat=len(branch_positions)):
                                        branched_payload = dict(payload)
                                        branched_payload["task_index"] = len(gated_candidate_payloads)
                                        branched_payload["fixed_value_positions"] = branch_positions
                                        branched_payload["fixed_value_bits"] = list(bits)
                                        gated_candidate_payloads.append(branched_payload)
                                        gated_payload_to_a.append(a_index)
                                        gated_payload_offsets.append(int(offset))
                                        tsae_interface_branch_payloads += 1
                                        tsae_interface_gated_branch_payloads += 1
                                else:
                                    payload["task_index"] = len(gated_candidate_payloads)
                                    gated_candidate_payloads.append(payload)
                                    gated_payload_to_a.append(a_index)
                                    gated_payload_offsets.append(int(offset))
                            if not any(payload_a == a_index for payload_a in gated_payload_to_a):
                                payload = shifted_a_payload(
                                    a_index,
                                    task,
                                    0,
                                    len(gated_candidate_payloads),
                                )
                                if payload is None:
                                    raise ValueError(f"no valid gated shifted A payload for task {a_index}")
                                payload = dict(payload)
                                payload["syndromes"] = payload["syndromes"][gated_shots]
                                payload["task_index"] = len(gated_candidate_payloads)
                                gated_candidate_payloads.append(payload)
                                gated_payload_to_a.append(a_index)
                                gated_payload_offsets.append(0)

                        gated_shifted_values = []
                        gated_shifted_costs = []
                        gated_expanded_to_a: list[int] = []
                        gated_expanded_offsets: list[int] = []
                        for payload_index, values, costs, flagged in run_a_candidate_payloads(
                            gated_candidate_payloads,
                            parallel_workers,
                            parallel_backend,
                        ):
                            for cand_index in range(values.shape[1]):
                                gated_shifted_values.append(values[:, cand_index, :])
                                gated_shifted_costs.append(costs[:, cand_index])
                                gated_expanded_to_a.append(gated_payload_to_a[payload_index])
                                gated_expanded_offsets.append(gated_payload_offsets[payload_index])
                            a_flagged += flagged

                        gated_payloads_by_a: list[list[int]] = [[] for _ in a_tasks]
                        for payload_index, a_index in enumerate(gated_expanded_to_a):
                            gated_payloads_by_a[a_index].append(payload_index)

                        gated_results, gated_candidate_count = apply_joint_b_dp(
                            gated_payloads_by_a,
                            gated_shifted_values,
                            gated_shifted_costs,
                            gated_expanded_offsets,
                            gated_shots,
                            gated_pass=True,
                        )
                        a_shifted_candidate_count += gated_candidate_count
                        for local_pos, (shot, final_cost, selected_residual, selected_nonzero, window_flags) in enumerate(gated_results):
                            accept = (
                                selected_residual < int(base_residual[local_pos])
                                or (
                                    selected_residual == int(base_residual[local_pos])
                                    and final_cost < float(base_cost[local_pos])
                                )
                            )
                            if accept:
                                tsae_interface_gated_accepted += 1
                                if selected_residual == 0 and int(base_residual[local_pos]) > 0:
                                    tsae_interface_gated_improved_to_zero += 1
                                joint_cost_by_shot[shot] = final_cost
                                joint_residual_by_shot[shot] = selected_residual
                                joint_nonzero_by_shot[shot] = selected_nonzero
                                joint_window_flagged[shot, : len(window_flags)] = window_flags
                            else:
                                total[shot] = base_total[local_pos]
                                joint_cost_by_shot[shot] = base_cost[local_pos]
                                joint_residual_by_shot[shot] = base_residual[local_pos]
                                joint_nonzero_by_shot[shot] = base_nonzero[local_pos]
                                joint_window_flagged[shot] = base_window_flags[local_pos]

                b_flagged = int(joint_window_flagged.sum())
                joint_b_dp_physical_residual = int(joint_residual_by_shot.sum())
                finite_costs = joint_cost_by_shot[np.isfinite(joint_cost_by_shot)]
                joint_b_dp_selected_cost = float(finite_costs.sum())
                joint_b_dp_selected_nonzero = int(joint_nonzero_by_shot.sum())
                a_shifted_selected_nonzero += joint_b_dp_selected_nonzero
                a_shifted_total_risk += float(joint_b_dp_physical_residual)
                a_shifted_total_cost += joint_b_dp_selected_cost

                a_candidate_values = None
                a_candidate_costs = None
            elif tsae_interface_joint_score:
                joint_branch_groups_total = 0
                joint_branch_b_star_nontrivial = 0
                joint_branch_chose_center_offset = 0
                for a_index, task in enumerate(a_tasks):
                    commit_cols = task["commit_cols"]
                    payload_list = payloads_by_a[a_index]
                    bits_groups: dict[tuple, list[int]] = {}
                    for p_idx in payload_list:
                        bits = payload_branch_bits[p_idx] if p_idx < len(payload_branch_bits) else ()
                        bits_groups.setdefault(bits, []).append(p_idx)
                    joint_branch_groups_total += len(bits_groups)
                    nb_data = []
                    if a_index - 1 >= 0 and a_index - 1 < len(b_tasks):
                        rows = b_tasks[a_index - 1]["rows"]
                        chk_sub = np.ascontiguousarray(problem.chk[rows, commit_cols], dtype=np.uint8)
                        nb_data.append((rows, chk_sub))
                    if a_index < len(b_tasks):
                        rows = b_tasks[a_index]["rows"]
                        chk_sub = np.ascontiguousarray(problem.chk[rows, commit_cols], dtype=np.uint8)
                        nb_data.append((rows, chk_sub))
                    zero_bits = tuple(0 for _ in next(iter(bits_groups.keys())))
                    for shot in range(shots):
                        # Phase 1: pick b* using joint score over offsets
                        best_bits = None
                        best_joint_score = np.inf
                        for bits, group in bits_groups.items():
                            local_sum = 0.0
                            valid = True
                            for p in group:
                                c = float(shifted_costs[p][shot])
                                if not np.isfinite(c):
                                    valid = False
                                    break
                                local_sum += c
                            if not valid:
                                continue
                            b_risk_per_offset = 0
                            for p in group:
                                cand = np.asarray(shifted_values[p][shot], dtype=np.uint8)
                                for rows, chk_sub in nb_data:
                                    contrib = (chk_sub @ cand) & 1
                                    s_b = np.asarray(det_data[shot, rows], dtype=np.uint8)
                                    b_risk_per_offset += int((s_b ^ contrib.astype(np.uint8)).sum())
                            score = local_sum + tsae_interface_zb_weight * b_risk_per_offset
                            if score < best_joint_score:
                                best_joint_score = score
                                best_bits = bits

                        if best_bits is None:
                            best_bits = payload_branch_bits[payload_list[0]] if payload_list else ()

                        # Phase 2: within b* group, pick commit candidate by greedy (local + B_risk)
                        group = bits_groups.get(best_bits, payload_list)
                        best_commit_payload = group[0]
                        best_commit_score = np.inf
                        best_commit_b_risk = 0
                        for p in group:
                            cand = np.asarray(shifted_values[p][shot], dtype=np.uint8)
                            risk = 0
                            for rows, chk_sub in nb_data:
                                contrib = (chk_sub @ cand) & 1
                                s_b = np.asarray(det_data[shot, rows], dtype=np.uint8)
                                risk += int((s_b ^ contrib.astype(np.uint8)).sum())
                            score = float(shifted_costs[p][shot]) + a_shifted_beta * risk
                            if score < best_commit_score:
                                best_commit_score = score
                                best_commit_payload = p
                                best_commit_b_risk = risk

                        chosen_value = np.asarray(shifted_values[best_commit_payload][shot], dtype=np.uint8)
                        total[shot, commit_cols] = chosen_value
                        a_shifted_selected_nonzero += int(payload_offsets[best_commit_payload] != 0)
                        a_shifted_total_cost += float(best_commit_score)
                        a_shifted_total_risk += float(best_commit_b_risk)
                        a_shifted_candidate_count += len(payload_list)
                        if len(zero_bits) > 0 and best_bits != zero_bits:
                            joint_branch_b_star_nontrivial += 1
                        if payload_offsets[best_commit_payload] == tsae_interface_commit_offset:
                            joint_branch_chose_center_offset += 1
                a_candidate_values = None
                a_candidate_costs = None
            else:
                for a_index, task in enumerate(a_tasks):
                    commit_cols = task["commit_cols"]
                    neighbor_b = []
                    if a_index - 1 >= 0 and a_index - 1 < len(b_tasks):
                        neighbor_b.append(a_index - 1)
                    if a_index < len(b_tasks):
                        neighbor_b.append(a_index)
                    commit_region_start = int(np.searchsorted(problem.region_offsets, commit_cols.start))
                    commit_region_stop = int(np.searchsorted(problem.region_offsets, commit_cols.stop))
                    commit_region_slices = [
                        slice(
                            problem.region_offsets[region] - commit_cols.start,
                            problem.region_offsets[region + 1] - commit_cols.start,
                        )
                        for region in range(commit_region_start, commit_region_stop)
                    ]
                    can_stitch = (
                        tsae_stitch_mode != "none"
                        and len(commit_region_slices) == 3
                        and len(payloads_by_a[a_index]) >= 3
                    )
                    center_region = commit_region_start + 1 if len(commit_region_slices) == 3 else None
                    center_row_block = center_region // 2 if center_region is not None else None
                    a_center_rows = (
                        row_slice(problem, center_row_block, center_row_block + 1)
                        if center_row_block is not None and center_row_block < num_blocks
                        else None
                    )
                    left_boundary_rows = None
                    right_boundary_rows = None
                    if a_index - 1 >= 0 and a_index - 1 < len(b_tasks):
                        left_b = b_tasks[a_index - 1]["rows"]
                        left_stop = left_b.stop // problem.detector_block_size
                        left_boundary_rows = row_slice(problem, left_stop - 1, left_stop)
                    if a_index < len(b_tasks):
                        right_b = b_tasks[a_index]["rows"]
                        right_start = right_b.start // problem.detector_block_size
                        right_boundary_rows = row_slice(problem, right_start, right_start + 1)

                    for shot in range(shots):
                        best_payload = payloads_by_a[a_index][0]
                        best_score = np.inf
                        best_risk = 0

                        for payload_index in payloads_by_a[a_index]:
                            candidate = shifted_values[payload_index][shot]
                            risk = 0
                            for b_index in neighbor_b:
                                rows = b_tasks[b_index]["rows"]
                                contribution = candidate @ problem.chk[rows, commit_cols].T
                                risk += int(((det_data[shot, rows] + contribution) % 2).sum())
                            score = shifted_costs[payload_index][shot] + a_shifted_beta * risk
                            if score < best_score:
                                best_score = score
                                best_payload = payload_index
                                best_risk = risk
                        chosen_value = shifted_values[best_payload][shot]
                        chosen_offset_nonzero = int(payload_offsets[best_payload] != 0)
                        chosen_score = float(best_score)
                        chosen_risk = int(best_risk)

                        if can_stitch and a_center_rows is not None:
                            tsae_stitch_attempts += 1
                            by_offset = {payload_offsets[payload_index]: payload_index for payload_index in payloads_by_a[a_index]}
                            center_payload = by_offset.get(0, best_payload)
                            negative_payloads = [
                                payload_index
                                for payload_index in payloads_by_a[a_index]
                                if payload_offsets[payload_index] < 0
                            ]
                            positive_payloads = [
                                payload_index
                                for payload_index in payloads_by_a[a_index]
                                if payload_offsets[payload_index] > 0
                            ]
                            left_options = negative_payloads + [center_payload]
                            right_options = [center_payload] + positive_payloads

                            def stitch_value(left_payload: int, mid_payload: int, right_payload: int) -> np.ndarray:
                                stitched = shifted_values[mid_payload][shot].copy()
                                stitched[commit_region_slices[0]] = shifted_values[left_payload][shot][commit_region_slices[0]]
                                stitched[commit_region_slices[2]] = shifted_values[right_payload][shot][commit_region_slices[2]]
                                return stitched

                            def residual_weight(rows: slice, candidate: np.ndarray) -> int:
                                contribution = candidate @ problem.chk[rows, commit_cols].T
                                return int(((det_data[shot, rows] + contribution) % 2).sum())

                            def stitch_score(candidate: np.ndarray) -> tuple[int, int, int, float]:
                                a_weight = residual_weight(a_center_rows, candidate)
                                left_weight = residual_weight(left_boundary_rows, candidate) if left_boundary_rows is not None else 0
                                right_weight = residual_weight(right_boundary_rows, candidate) if right_boundary_rows is not None else 0
                                score = float(a_weight + a_shifted_beta * (left_weight + right_weight))
                                return a_weight, left_weight + right_weight, a_weight + left_weight + right_weight, score

                            stitched_value = None
                            stitched_score = None
                            stitched_risk = None
                            stitched_non_payload = 0
                            if tsae_stitch_mode == "component":
                                left_payload = min(
                                    left_options,
                                    key=lambda payload_index: (
                                        residual_weight(left_boundary_rows, shifted_values[payload_index][shot])
                                        if left_boundary_rows is not None
                                        else 0,
                                        shifted_costs[payload_index][shot],
                                    ),
                                )
                                right_payload = min(
                                    right_options,
                                    key=lambda payload_index: (
                                        residual_weight(right_boundary_rows, shifted_values[payload_index][shot])
                                        if right_boundary_rows is not None
                                        else 0,
                                        shifted_costs[payload_index][shot],
                                    ),
                                )
                                candidate = stitch_value(left_payload, center_payload, right_payload)
                                a_weight, boundary_weight, _, score = stitch_score(candidate)
                                if a_weight == 0:
                                    stitched_value = candidate
                                    stitched_score = score
                                    stitched_risk = boundary_weight
                                    stitched_non_payload = int(
                                        not (
                                            left_payload == center_payload
                                            and right_payload == center_payload
                                        )
                                    )
                            elif tsae_stitch_mode == "pairwise":
                                best_tuple = None
                                for left_payload in left_options:
                                    for right_payload in right_options:
                                        candidate = stitch_value(left_payload, center_payload, right_payload)
                                        a_weight, boundary_weight, _, score = stitch_score(candidate)
                                        tie_cost = (
                                            shifted_costs[left_payload][shot]
                                            + shifted_costs[center_payload][shot]
                                            + shifted_costs[right_payload][shot]
                                        )
                                        key = (score, a_weight, boundary_weight, tie_cost)
                                        if best_tuple is None or key < best_tuple[0]:
                                            best_tuple = (key, candidate, boundary_weight, left_payload, right_payload)
                                if best_tuple is not None:
                                    key, candidate, boundary_weight, left_payload, right_payload = best_tuple
                                    stitched_value = candidate
                                    stitched_score = float(key[0])
                                    stitched_risk = int(boundary_weight)
                                    stitched_non_payload = int(
                                        not (
                                            left_payload == center_payload
                                            and right_payload == center_payload
                                        )
                                    )

                            if stitched_value is not None:
                                chosen_value = stitched_value
                                chosen_score = float(stitched_score)
                                chosen_risk = int(stitched_risk)
                                tsae_stitch_accepted += 1
                                tsae_stitch_non_payload += stitched_non_payload
                            else:
                                tsae_stitch_fallback += 1

                        total[shot, commit_cols] = chosen_value
                        a_shifted_selected_nonzero += chosen_offset_nonzero
                        a_shifted_total_risk += float(chosen_risk)
                        a_shifted_total_cost += float(chosen_score)
                        if can_stitch:
                            tsae_stitch_total_score += float(chosen_score)
                        a_shifted_candidate_count += len(payloads_by_a[a_index])

            if oracle_diag is not None:
                tsae_diag_in_pool = np.zeros((shots, len(a_tasks)), dtype=bool)
                tsae_diag_closest_hamming = np.full((shots, len(a_tasks)), -1, dtype=np.int32)
                tsae_diag_pool_size = np.zeros(len(a_tasks), dtype=np.int32)
                tsae_diag_picked_offset = np.zeros((shots, len(a_tasks)), dtype=np.int32)
                tsae_diag_oracle_offset = np.full((shots, len(a_tasks)), 99, dtype=np.int32)
                for a_index, task in enumerate(a_tasks):
                    commit_cols = task["commit_cols"]
                    payload_list = payloads_by_a[a_index]
                    tsae_diag_pool_size[a_index] = len(payload_list)
                    for shot in range(shots):
                        oracle_commit = np.asarray(oracle_diag[shot, commit_cols], dtype=np.uint8)
                        best_hamming = oracle_commit.size + 1
                        matched_offset = 99
                        for payload_index in payload_list:
                            cand = np.asarray(shifted_values[payload_index][shot], dtype=np.uint8)
                            hamming = int((cand ^ oracle_commit).sum())
                            if hamming < best_hamming:
                                best_hamming = hamming
                            if hamming == 0 and matched_offset == 99:
                                matched_offset = payload_offsets[payload_index]
                        tsae_diag_in_pool[shot, a_index] = (matched_offset != 99)
                        tsae_diag_closest_hamming[shot, a_index] = best_hamming
                        tsae_diag_oracle_offset[shot, a_index] = matched_offset
                        committed_cand = np.asarray(total[shot, commit_cols], dtype=np.uint8)
                        for payload_index in payload_list:
                            cand = np.asarray(shifted_values[payload_index][shot], dtype=np.uint8)
                            if np.array_equal(cand, committed_cand):
                                tsae_diag_picked_offset[shot, a_index] = payload_offsets[payload_index]
                                break

            a_candidate_values = None
            a_candidate_costs = None
    elif soft_boundary_message or top_k_boundary <= 1 or not use_buffer_aligned:
        for task_index, commit_values, flagged in run_window_payloads(
            a_payloads,
            parallel_workers,
            parallel_backend,
        ):
            total[:, a_tasks[task_index]["commit_cols"]] = commit_values
            a_flagged += flagged
        a_candidate_values = None
        a_candidate_costs = None
    else:
        candidate_payloads = []
        for payload in a_payloads:
            candidate_payload = dict(payload)
            candidate_payload["top_k"] = top_k_boundary
            candidate_payloads.append(candidate_payload)
        a_candidate_values = [None] * len(a_tasks)
        a_candidate_costs = [None] * len(a_tasks)
        for task_index, values, costs, flagged in run_a_candidate_payloads(
            candidate_payloads,
            parallel_workers,
            parallel_backend,
        ):
            a_candidate_values[task_index] = values
            a_candidate_costs[task_index] = costs
            a_flagged += flagged

    use_soft_boundary = (
        soft_boundary_message
        and oracle_errors is None
        and use_buffer_aligned
        and top_k_boundary <= 1
        and not joint_b_dp_decoded
    )
    a_only_total = total.copy()
    if joint_b_dp_decoded:
        for task in b_tasks:
            a_only_total[:, task["cols"]] = 0
    r_after_a = (det_data + a_only_total @ problem.chk.T) % 2
    b_local_residuals = None
    b_noisy_boundary_values = None
    z_repair_attempts = 0
    z_repair_accepted = 0
    z_repair_success_to_zero = 0
    z_repair_no_candidate = 0
    z_repair_decode_flagged = 0
    z_repair_weight_before = 0
    z_repair_weight_after = 0
    z_repair_delta_weight = 0
    z_repair_candidate_vars = 0
    z_repair_flipped_vars = 0
    z_repair_left_count = 0
    z_repair_right_count = 0
    z_repair_both_sides_count = 0
    z_joint_attempts = 0
    z_joint_accepted = 0
    z_joint_success_to_zero = 0
    z_joint_local_flagged = 0
    z_joint_no_candidate = 0
    z_joint_weight_before = 0
    z_joint_weight_after = 0
    z_joint_delta_weight = 0
    z_joint_candidate_vars = 0
    z_joint_flipped_vars = 0
    z_joint_left_count = 0
    z_joint_right_count = 0
    z_joint_both_sides_count = 0

    if use_soft_boundary:
        b_flagged = 0
        topk_selected_cost = 0.0
        soft_one = float(np.clip(soft_boundary_one_prior, 1e-12, 0.49))
        soft_zero = float(np.clip(soft_boundary_zero_prior, 1e-12, 0.49))
        for task in b_tasks:
            rows = task["rows"]
            row_start = rows.start // problem.detector_block_size
            row_stop = rows.stop // problem.detector_block_size
            left_boundary_region = 2 * row_start - 1
            right_boundary_region = 2 * (row_stop - 1) + 1
            col_start = max(0, left_boundary_region)
            col_stop = min(right_boundary_region + 1, num_regions)
            cols = region_col_slice(problem, col_start, col_stop)
            mat = problem.chk[rows, cols]
            base_prior = problem.priors[cols]
            left_cols = (
                region_col_slice(problem, left_boundary_region, left_boundary_region + 1)
                if 0 <= left_boundary_region < num_regions
                else None
            )
            right_cols = (
                region_col_slice(problem, right_boundary_region, right_boundary_region + 1)
                if 0 <= right_boundary_region < num_regions
                else None
            )
            left_local = (
                slice(left_cols.start - cols.start, left_cols.stop - cols.start)
                if left_cols is not None
                else None
            )
            right_local = (
                slice(right_cols.start - cols.start, right_cols.stop - cols.start)
                if right_cols is not None
                else None
            )

            for shot in range(shots):
                local_prior = base_prior.copy()
                if left_cols is not None:
                    local_prior[left_local] = np.where(total[shot, left_cols] > 0, soft_one, soft_zero)
                if right_cols is not None:
                    local_prior[right_local] = np.where(total[shot, right_cols] > 0, soft_one, soft_zero)
                decoder = make_bp_osd(mat, local_prior, max_iter, osd_order, window_shorten, shorten_pre_max_iter)
                local_e = decoder.decode(det_data[shot, rows]).astype(np.uint8)
                if ((mat @ local_e + det_data[shot, rows]) % 2).any():
                    b_flagged += 1
                total[shot, cols] = local_e
    elif joint_b_dp_decoded:
        topk_selected_cost = joint_b_dp_selected_cost
        if seam_diagnostics and use_buffer_aligned:
            b_local_residuals = []
            residual = r_after_a
            for task in b_tasks:
                rows = task["rows"]
                cols = task["cols"]
                values = total[:, cols]
                local_residual = (
                    values @ problem.chk[rows, cols].T
                    + residual[:, rows]
                ) % 2
                b_local_residuals.append(local_residual.astype(np.uint8))
    elif oracle_errors is not None or top_k_boundary <= 1 or not use_buffer_aligned:
        residual = (det_data + total @ problem.chk.T) % 2

        b_flagged = 0
        repair_attempts = 0
        repair_accepted = 0
        repair_success = 0
        twoflip_attempts = 0
        twoflip_accepted = 0
        twoflip_success = 0
        aba_initial_b_flagged = 0
        aba_feedback_attempts = 0
        aba_feedback_proposed = 0
        aba_feedback_accepted = 0
        aba_rerun_windows = 0
        if aba_boundary_feedback and use_buffer_aligned:
            b_decoders = []
            b_mats = []
            b_value_limits = []
            for task in b_tasks:
                rows = task["rows"]
                cols = task["cols"]
                mat = problem.chk[rows, cols]
                prior = problem.priors[cols]
                value_limit = cols.stop - cols.start
                if b_noisy_boundary:
                    mat, prior = add_noisy_boundary_columns(
                        problem,
                        mat,
                        prior,
                        rows,
                        cols,
                        boundary_width=problem.detector_block_size,
                    )
                b_mats.append(mat)
                b_value_limits.append(value_limit)
                b_decoders.append(
                    make_bp_osd(
                        mat,
                        prior,
                        max_iter,
                        osd_order,
                        window_shorten,
                        shorten_pre_max_iter,
                    )
                )

            b_values = [
                np.zeros((shots, b_value_limits[index]), dtype=np.uint8)
                for index in range(len(b_tasks))
            ]
            proposals_by_shot: list[list[tuple[int, int, tuple[int, ...], int, int]]] = [
                [] for _ in range(shots)
            ]

            for b_index, task in enumerate(b_tasks):
                rows = task["rows"]
                decoder = b_decoders[b_index]
                for shot in range(shots):
                    local_syndrome = residual[shot, rows].copy()
                    local_e = decoder.decode(local_syndrome).astype(np.uint8)
                    b_values[b_index][shot] = local_e[: b_value_limits[b_index]]

            for b_index, task in enumerate(b_tasks):
                total[:, task["cols"]] = b_values[b_index]

            residual = (det_data + total @ problem.chk.T) % 2
            for b_index, task in enumerate(b_tasks):
                rows = task["rows"]
                boundary_cols = np.concatenate(
                    (
                        np.arange(a_tasks[b_index]["commit_cols"].start, a_tasks[b_index]["commit_cols"].stop),
                        np.arange(a_tasks[b_index + 1]["commit_cols"].start, a_tasks[b_index + 1]["commit_cols"].stop),
                    )
                )
                affects = np.asarray(problem.chk[rows, boundary_cols].sum(axis=0)).reshape(-1) > 0
                boundary_cols = boundary_cols[affects]

                for shot in range(shots):
                    stitched_residual = residual[shot, rows].copy()
                    residual_weight = int(stitched_residual.sum())
                    if residual_weight == 0:
                        continue

                    aba_initial_b_flagged += 1
                    aba_feedback_attempts += 1
                    flip_cols, repaired_weight = choose_boundary_feedback_flips(
                        problem.chk,
                        rows,
                        boundary_cols,
                        stitched_residual,
                        aba_max_flips,
                        aba_candidate_cols,
                    )
                    if flip_cols and repaired_weight < residual_weight:
                        aba_feedback_proposed += 1
                        proposals_by_shot[shot].append(
                            (
                                residual_weight - repaired_weight,
                                b_index,
                                tuple(sorted(flip_cols)),
                                residual_weight,
                                repaired_weight,
                            )
                        )

            affected = set()
            for shot, proposals in enumerate(proposals_by_shot):
                used_cols: set[int] = set()
                for _, b_index, flip_cols, _, _ in sorted(
                    proposals,
                    key=lambda item: (-item[0], len(item[2]), item[1]),
                ):
                    if any(col in used_cols for col in flip_cols):
                        continue
                    for col in flip_cols:
                        total[shot, col] ^= 1
                        used_cols.add(col)
                    aba_feedback_accepted += 1
                    for neighbor in (b_index - 1, b_index, b_index + 1):
                        if 0 <= neighbor < len(b_tasks):
                            affected.add((shot, neighbor))

            residual = (det_data + total @ problem.chk.T) % 2
            for shot, b_index in sorted(affected):
                task = b_tasks[b_index]
                rows = task["rows"]
                cols = task["cols"]
                decoder = b_decoders[b_index]
                local_e = decoder.decode(residual[shot, rows]).astype(np.uint8)
                total[shot, cols] = local_e[: b_value_limits[b_index]]
                residual[shot] = (det_data[shot] + total[shot] @ problem.chk.T) % 2
                aba_rerun_windows += 1

            residual = (det_data + total @ problem.chk.T) % 2
            for task in b_tasks:
                rows = task["rows"]
                for shot in range(shots):
                    if int(residual[shot, rows].sum()) > 0:
                        b_flagged += 1
        elif flag_triggered_single_flip and use_buffer_aligned and a_commit_reliability is not None:
            b_decoders = [
                make_bp_osd(
                    problem.chk[task["rows"], task["cols"]],
                    problem.priors[task["cols"]],
                    max_iter,
                    osd_order,
                    window_shorten,
                    shorten_pre_max_iter,
                )
                for task in b_tasks
            ]
            for b_index, task in enumerate(b_tasks):
                rows = task["rows"]
                cols = task["cols"]
                mat = problem.chk[rows, cols]
                decoder = b_decoders[b_index]
                boundary_cols = np.concatenate(
                    (
                        np.arange(a_tasks[b_index]["commit_cols"].start, a_tasks[b_index]["commit_cols"].stop),
                        np.arange(a_tasks[b_index + 1]["commit_cols"].start, a_tasks[b_index + 1]["commit_cols"].stop),
                    )
                )
                affects = np.asarray(problem.chk[rows, boundary_cols].sum(axis=0)).reshape(-1) > 0
                boundary_cols = boundary_cols[affects]

                for shot in range(shots):
                    local_syndrome = residual[shot, rows].copy()
                    local_e = decoder.decode(local_syndrome).astype(np.uint8)
                    local_residual = (mat @ local_e + local_syndrome) % 2
                    residual_weight = int(local_residual.sum())
                    if residual_weight == 0:
                        total[shot, cols] = local_e
                        residual[shot] = (det_data[shot] + total[shot] @ problem.chk.T) % 2
                        continue

                    b_flagged += 1
                    if boundary_cols.size == 0:
                        total[shot, cols] = local_e
                        residual[shot] = (det_data[shot] + total[shot] @ problem.chk.T) % 2
                        continue

                    repair_attempts += 1
                    reliability = a_commit_reliability[shot, boundary_cols]
                    ordered = np.argsort(reliability)
                    single_index = int(ordered[0])
                    flip_col = int(boundary_cols[single_index])
                    repaired_syndrome = (
                        local_syndrome + problem.chk[rows, flip_col].astype(np.uint8)
                    ) % 2
                    repaired_e = decoder.decode(repaired_syndrome).astype(np.uint8)
                    repaired_residual = (mat @ repaired_e + repaired_syndrome) % 2
                    repaired_weight = int(repaired_residual.sum())
                    best_flip_cols = [flip_col]
                    best_e = repaired_e
                    best_weight = repaired_weight

                    if bounded_two_flip and best_weight > 0 and boundary_cols.size >= 2:
                        twoflip_attempts += 1
                        candidate_count = min(two_flip_candidates, boundary_cols.size)
                        candidate_cols = boundary_cols[ordered[:candidate_count]]
                        for first in range(candidate_count):
                            for second in range(first + 1, candidate_count):
                                col1 = int(candidate_cols[first])
                                col2 = int(candidate_cols[second])
                                two_syndrome = (
                                    local_syndrome
                                    + problem.chk[rows, col1].astype(np.uint8)
                                    + problem.chk[rows, col2].astype(np.uint8)
                                ) % 2
                                two_e = decoder.decode(two_syndrome).astype(np.uint8)
                                two_residual = (mat @ two_e + two_syndrome) % 2
                                two_weight = int(two_residual.sum())
                                if two_weight < best_weight:
                                    best_weight = two_weight
                                    best_flip_cols = [col1, col2]
                                    best_e = two_e

                    if best_weight < residual_weight:
                        for flip in best_flip_cols:
                            total[shot, flip] ^= 1
                        total[shot, cols] = best_e
                        repair_accepted += 1
                        if len(best_flip_cols) == 2:
                            twoflip_accepted += 1
                        if best_weight == 0:
                            repair_success += 1
                            if len(best_flip_cols) == 2:
                                twoflip_success += 1
                    else:
                        total[shot, cols] = local_e
                    residual[shot] = (det_data[shot] + total[shot] @ problem.chk.T) % 2
        else:
            b_payloads = []
            for task_index, task in enumerate(b_tasks):
                rows = task["rows"]
                cols = task["cols"]
                mat = problem.chk[rows, cols]
                prior = problem.priors[cols]
                value_limit = cols.stop - cols.start
                if b_noisy_boundary:
                    mat, prior = add_noisy_boundary_columns(
                        problem,
                        mat,
                        prior,
                        rows,
                        cols,
                        boundary_width=problem.detector_block_size,
                    )
                    if seam_diagnostics or z_boundary_repair or z_joint_retry:
                        value_limit = mat.shape[1]
                b_payloads.append(
                    {
                        "task_index": task_index,
                        "mat": mat,
                        "prior": prior,
                        "syndromes": residual[:, rows],
                        "max_iter": max_iter,
                        "osd_order": osd_order,
                        "shorten": window_shorten,
                        "shorten_pre_max_iter": shorten_pre_max_iter,
                        "value_start": None,
                        "value_stop": None,
                        "value_limit": value_limit,
                    }
                )

            for task_index, values, flagged in run_window_payloads(
                b_payloads,
                parallel_workers,
                parallel_backend,
            ):
                physical_width = b_tasks[task_index]["cols"].stop - b_tasks[task_index]["cols"].start
                total[:, b_tasks[task_index]["cols"]] = values[:, :physical_width]
                if (
                    (seam_diagnostics or z_boundary_repair or z_joint_retry)
                    and b_noisy_boundary
                    and values.shape[1] > physical_width
                ):
                    if b_noisy_boundary_values is None:
                        b_noisy_boundary_values = [None] * len(b_tasks)
                    b_noisy_boundary_values[task_index] = values[:, physical_width:].copy()
                b_flagged += flagged

            if z_boundary_repair and b_noisy_boundary_values is not None:
                stitched_residual = (det_data + total @ problem.chk.T) % 2
                boundary_width = problem.detector_block_size

                def region_cols_array(region_start: int, region_stop: int) -> np.ndarray:
                    region_start = max(0, region_start)
                    region_stop = min(num_regions, region_stop)
                    if region_start >= region_stop:
                        return np.array([], dtype=int)
                    col_slice = region_col_slice(problem, region_start, region_stop)
                    return np.arange(col_slice.start, col_slice.stop, dtype=int)

                for task_index, z_values in enumerate(b_noisy_boundary_values):
                    if z_values is None:
                        continue
                    task = b_tasks[task_index]
                    rows = task["rows"]
                    row_start = rows.start // problem.detector_block_size
                    row_stop = rows.stop // problem.detector_block_size
                    left_boundary_region = 2 * row_start - 1
                    right_boundary_region = 2 * (row_stop - 1) + 1
                    b_region_start = left_boundary_region + 1
                    b_region_stop = right_boundary_region

                    for shot in range(shots):
                        z_value = z_values[shot]
                        left_z = z_value[:boundary_width]
                        right_z = z_value[boundary_width:]
                        left_used = bool(left_z.any())
                        right_used = bool(right_z.any())
                        if not left_used and not right_used:
                            continue

                        before_weight = int(stitched_residual[shot].sum())
                        if before_weight == 0:
                            continue

                        z_repair_attempts += 1
                        z_repair_weight_before += before_weight
                        if left_used and right_used:
                            z_repair_both_sides_count += 1
                        elif left_used:
                            z_repair_left_count += 1
                        else:
                            z_repair_right_count += 1

                        z_row_parts = []
                        candidate_parts = []
                        if left_used:
                            z_row_parts.append(rows.start + np.flatnonzero(left_z))
                            candidate_parts.append(
                                region_cols_array(left_boundary_region, left_boundary_region + 1)
                            )
                            candidate_parts.append(
                                region_cols_array(
                                    b_region_start,
                                    min(b_region_start + z_repair_edge_width, b_region_stop),
                                )
                            )
                        if right_used:
                            z_row_parts.append(rows.stop - boundary_width + np.flatnonzero(right_z))
                            candidate_parts.append(
                                region_cols_array(
                                    max(b_region_start, b_region_stop - z_repair_edge_width),
                                    b_region_stop,
                                )
                            )
                            candidate_parts.append(
                                region_cols_array(right_boundary_region, right_boundary_region + 1)
                            )

                        z_rows = unique_concat(z_row_parts)
                        candidate_cols = unique_concat(candidate_parts)
                        if z_rows.size == 0 or candidate_cols.size == 0:
                            z_repair_no_candidate += 1
                            z_repair_weight_after += before_weight
                            continue

                        z_affects = (
                            np.asarray(problem.chk[np.ix_(z_rows, candidate_cols)].sum(axis=0))
                            .reshape(-1)
                            .astype(bool)
                        )
                        candidate_cols = candidate_cols[z_affects]
                        if candidate_cols.size == 0:
                            z_repair_no_candidate += 1
                            z_repair_weight_after += before_weight
                            continue

                        affected_rows = (
                            np.asarray(problem.chk[:, candidate_cols].sum(axis=1))
                            .reshape(-1)
                            .astype(bool)
                        )
                        repair_rows = np.flatnonzero(affected_rows)
                        if repair_rows.size == 0:
                            z_repair_no_candidate += 1
                            z_repair_weight_after += before_weight
                            continue

                        repair_mat = np.asarray(
                            problem.chk[np.ix_(repair_rows, candidate_cols)],
                            dtype=np.uint8,
                        )
                        repair_syndrome = stitched_residual[shot, repair_rows].astype(np.uint8)
                        repair_prior = problem.priors[candidate_cols]
                        full_candidate_cols = np.asarray(
                            problem.chk[:, candidate_cols],
                            dtype=np.uint8,
                        )
                        overlap = stitched_residual[shot].astype(np.int16) @ full_candidate_cols.astype(np.int16)
                        support = full_candidate_cols.sum(axis=0).astype(np.int16)
                        gain = 2 * overlap - support
                        gain_order = np.argsort(-gain)
                        positive = gain_order[gain[gain_order] > 0]
                        if positive.size == 0:
                            enum_local = gain_order[: min(16, gain_order.size)]
                        else:
                            enum_local = positive[: min(16, positive.size)]
                        best_enum_delta = None
                        best_enum_after = before_weight

                        for local_index in enum_local:
                            contribution = full_candidate_cols[:, local_index]
                            candidate_residual = (stitched_residual[shot] + contribution) % 2
                            after_weight = int(candidate_residual.sum())
                            if after_weight < best_enum_after:
                                best_enum_after = after_weight
                                best_enum_delta = np.zeros(candidate_cols.size, dtype=np.uint8)
                                best_enum_delta[local_index] = 1
                                if best_enum_after == 0:
                                    break

                        if best_enum_after != 0 and enum_local.size >= 2:
                            for first_pos in range(enum_local.size):
                                first = int(enum_local[first_pos])
                                for second in enum_local[first_pos + 1 :]:
                                    contribution = (
                                        full_candidate_cols[:, first]
                                        + full_candidate_cols[:, int(second)]
                                    ) % 2
                                    candidate_residual = (stitched_residual[shot] + contribution) % 2
                                    after_weight = int(candidate_residual.sum())
                                    if after_weight < best_enum_after:
                                        best_enum_after = after_weight
                                        best_enum_delta = np.zeros(candidate_cols.size, dtype=np.uint8)
                                        best_enum_delta[first] = 1
                                        best_enum_delta[int(second)] = 1
                                        if best_enum_after == 0:
                                            break
                                if best_enum_after == 0:
                                    break

                        if best_enum_delta is not None and best_enum_after == 0:
                            total[shot, candidate_cols] ^= best_enum_delta
                            stitched_residual[shot] = (
                                stitched_residual[shot]
                                + best_enum_delta @ full_candidate_cols.T
                            ) % 2
                            z_repair_candidate_vars += int(candidate_cols.size)
                            z_repair_flipped_vars += int(best_enum_delta.sum())
                            z_repair_weight_after += best_enum_after
                            z_repair_delta_weight += before_weight - best_enum_after
                            z_repair_accepted += 1
                            z_repair_success_to_zero += 1
                            continue

                        repair_order = min(osd_order, repair_mat.shape[1])
                        while True:
                            try:
                                repair_decoder = make_bp_osd(
                                    repair_mat,
                                    repair_prior,
                                    max_iter,
                                    repair_order,
                                    window_shorten,
                                    shorten_pre_max_iter,
                                )
                                break
                            except ValueError:
                                if repair_order <= 0:
                                    raise
                                repair_order -= 1
                        delta = repair_decoder.decode(repair_syndrome).astype(np.uint8)
                        local_residual = (repair_mat @ delta + repair_syndrome) % 2
                        if local_residual.any():
                            z_repair_decode_flagged += 1

                        z_repair_candidate_vars += int(candidate_cols.size)
                        if best_enum_delta is not None and best_enum_after < before_weight:
                            bp_candidate_delta = delta
                            bp_contribution = (bp_candidate_delta @ full_candidate_cols.T) % 2
                            bp_residual = (stitched_residual[shot] + bp_contribution) % 2
                            bp_after = int(bp_residual.sum())
                            if best_enum_after < bp_after:
                                delta = best_enum_delta
                        z_repair_flipped_vars += int(delta.sum())
                        if not delta.any():
                            z_repair_weight_after += before_weight
                            continue

                        contribution = (delta @ full_candidate_cols.T) % 2
                        candidate_residual = (stitched_residual[shot] + contribution) % 2
                        after_weight = int(candidate_residual.sum())
                        z_repair_weight_after += after_weight
                        z_repair_delta_weight += before_weight - after_weight

                        if after_weight < before_weight:
                            total[shot, candidate_cols] ^= delta
                            stitched_residual[shot] = candidate_residual.astype(np.uint8)
                            z_repair_accepted += 1
                            if after_weight == 0:
                                z_repair_success_to_zero += 1

            if z_joint_retry and b_noisy_boundary_values is not None:
                stitched_residual = (det_data + total @ problem.chk.T) % 2
                boundary_width = problem.detector_block_size
                joint_specs = []
                for task_index, task in enumerate(b_tasks):
                    rows = task["rows"]
                    cols = task["cols"]
                    boundary_cols = np.array([], dtype=int)
                    if task_index < len(a_tasks):
                        left_commit = a_tasks[task_index]["commit_cols"]
                        boundary_cols = np.concatenate(
                            (
                                boundary_cols,
                                np.arange(left_commit.start, left_commit.stop, dtype=int),
                            )
                        )
                    if task_index + 1 < len(a_tasks):
                        right_commit = a_tasks[task_index + 1]["commit_cols"]
                        boundary_cols = np.concatenate(
                            (
                                boundary_cols,
                                np.arange(right_commit.start, right_commit.stop, dtype=int),
                            )
                        )
                    if boundary_cols.size:
                        affects = (
                            np.asarray(problem.chk[rows, boundary_cols].sum(axis=0))
                            .reshape(-1)
                            .astype(bool)
                        )
                        boundary_cols = boundary_cols[affects]
                    b_cols = np.arange(cols.start, cols.stop, dtype=int)
                    candidate_cols = unique_concat([b_cols, boundary_cols])
                    if candidate_cols.size == 0:
                        joint_specs.append(None)
                        continue
                    joint_mat = np.asarray(problem.chk[np.ix_(np.arange(rows.start, rows.stop), candidate_cols)], dtype=np.uint8)
                    joint_prior = problem.priors[candidate_cols]
                    joint_decoder = make_bp_osd(
                        joint_mat,
                        joint_prior,
                        max_iter,
                        osd_order,
                        window_shorten,
                        shorten_pre_max_iter,
                    )
                    joint_specs.append((candidate_cols, joint_mat, joint_decoder))

                for task_index, z_values in enumerate(b_noisy_boundary_values):
                    if z_values is None or joint_specs[task_index] is None:
                        continue
                    task = b_tasks[task_index]
                    rows = task["rows"]
                    row_indices = np.arange(rows.start, rows.stop)
                    candidate_cols, joint_mat, joint_decoder = joint_specs[task_index]
                    for shot in range(shots):
                        z_value = z_values[shot]
                        left_used = bool(z_value[:boundary_width].any())
                        right_used = bool(z_value[boundary_width:].any())
                        if not left_used and not right_used:
                            continue

                        before_weight = int(stitched_residual[shot].sum())
                        if before_weight == 0:
                            continue

                        z_joint_attempts += 1
                        z_joint_weight_before += before_weight
                        if left_used and right_used:
                            z_joint_both_sides_count += 1
                        elif left_used:
                            z_joint_left_count += 1
                        else:
                            z_joint_right_count += 1

                        current_values = total[shot, candidate_cols]
                        local_syndrome = (
                            stitched_residual[shot, rows]
                            + current_values @ problem.chk[np.ix_(row_indices, candidate_cols)].T
                        ) % 2
                        decoded_values = joint_decoder.decode(local_syndrome).astype(np.uint8)
                        local_residual = (joint_mat @ decoded_values + local_syndrome) % 2
                        if local_residual.any():
                            z_joint_local_flagged += 1

                        delta = (current_values + decoded_values) % 2
                        z_joint_candidate_vars += int(candidate_cols.size)
                        z_joint_flipped_vars += int(delta.sum())
                        if not delta.any():
                            z_joint_weight_after += before_weight
                            continue

                        contribution = (delta @ problem.chk[:, candidate_cols].T) % 2
                        candidate_residual = (stitched_residual[shot] + contribution) % 2
                        after_weight = int(candidate_residual.sum())
                        z_joint_weight_after += after_weight
                        z_joint_delta_weight += before_weight - after_weight

                        if after_weight < before_weight:
                            total[shot, candidate_cols] = decoded_values
                            stitched_residual[shot] = candidate_residual.astype(np.uint8)
                            z_joint_accepted += 1
                            if after_weight == 0:
                                z_joint_success_to_zero += 1

            if seam_diagnostics:
                b_local_residuals = []
                for task_index, task in enumerate(b_tasks):
                    rows = task["rows"]
                    cols = task["cols"]
                    values = total[:, task["cols"]]
                    local_residual = (
                        values @ problem.chk[rows, cols].T
                        + residual[:, rows]
                    ) % 2
                    b_local_residuals.append(local_residual.astype(np.uint8))
        topk_selected_cost = 0.0
    else:
        b_flagged = 0
        topk_selected_cost = 0.0
        b_decoders = [
            make_bp_osd(
                problem.chk[task["rows"], task["cols"]],
                problem.priors[task["cols"]],
                max_iter,
                osd_order,
                window_shorten,
                shorten_pre_max_iter,
            )
            for task in b_tasks
        ]
        flag_penalty = 1000.0

        for shot in range(shots):
            b_options = []
            for b_index, task in enumerate(b_tasks):
                rows = task["rows"]
                cols = task["cols"]
                decoder = b_decoders[b_index]
                mat = problem.chk[rows, cols]
                prior = problem.priors[cols]
                left_a = b_index
                right_a = b_index + 1
                left_values = a_candidate_values[left_a][shot]
                right_values = a_candidate_values[right_a][shot]
                options = {}
                for left_k in range(top_k_boundary):
                    for right_k in range(top_k_boundary):
                        local_syndrome = det_data[shot, rows].copy()
                        left_cols = a_tasks[left_a]["commit_cols"]
                        right_cols = a_tasks[right_a]["commit_cols"]
                        local_syndrome = (
                            local_syndrome
                            + left_values[left_k] @ problem.chk[rows, left_cols].T
                            + right_values[right_k] @ problem.chk[rows, right_cols].T
                        ) % 2
                        local_e = decoder.decode(local_syndrome).astype(np.uint8)
                        local_residual = (mat @ local_e + local_syndrome) % 2
                        residual_weight = int(local_residual.sum())
                        cost = error_cost(local_e, prior) + flag_penalty * residual_weight
                        options[(left_k, right_k)] = (cost, local_e, residual_weight)
                b_options.append(options)

            dp = a_candidate_costs[0][shot].copy()
            parents = []
            for b_index, options in enumerate(b_options):
                next_cost = np.full(top_k_boundary, np.inf)
                parent = np.zeros(top_k_boundary, dtype=int)
                for left_k in range(top_k_boundary):
                    for right_k in range(top_k_boundary):
                        cost = (
                            dp[left_k]
                            + a_candidate_costs[b_index + 1][shot, right_k]
                            + options[(left_k, right_k)][0]
                        )
                        if cost < next_cost[right_k]:
                            next_cost[right_k] = cost
                            parent[right_k] = left_k
                parents.append(parent)
                dp = next_cost

            chosen = [0] * len(a_tasks)
            if b_options:
                chosen[-1] = int(np.argmin(dp))
                for b_index in range(len(b_options) - 1, -1, -1):
                    chosen[b_index] = int(parents[b_index][chosen[b_index + 1]])
                topk_selected_cost += float(dp[chosen[-1]])
            else:
                for a_index in range(len(a_tasks)):
                    chosen[a_index] = int(np.argmin(a_candidate_costs[a_index][shot]))
                topk_selected_cost += float(sum(a_candidate_costs[i][shot, chosen[i]] for i in range(len(a_tasks))))

            for a_index, task in enumerate(a_tasks):
                total[shot, task["commit_cols"]] = a_candidate_values[a_index][shot, chosen[a_index]]

            for b_index, task in enumerate(b_tasks):
                local_e = b_options[b_index][(chosen[b_index], chosen[b_index + 1])][1]
                residual_weight = b_options[b_index][(chosen[b_index], chosen[b_index + 1])][2]
                total[shot, task["cols"]] = local_e
                if residual_weight:
                    b_flagged += 1

    diagnostics = {
        "schedule": "buffer_aligned" if use_buffer_aligned else "staggered",
        "a_windows": len(a_tasks),
        "b_windows": len(b_tasks),
        "a_window_rows": [task["row_label"] for task in a_tasks],
        "b_window_rows": [task["row_label"] for task in b_tasks],
        "a_commit_cols": [task["commit_label"] for task in a_tasks],
        "b_commit_cols": [task["commit_label"] for task in b_tasks],
        "step": step,
        "a_source": "oracle" if oracle_errors is not None else "decoded",
        "a_noisy_boundary": a_noisy_boundary,
        "b_noisy_boundary": b_noisy_boundary,
        "window_shorten": window_shorten,
        "shorten_pre_max_iter": shorten_pre_max_iter,
        "parallel_workers": parallel_workers,
        "parallel_backend": parallel_backend,
        "n_a": n_a,
        "n_a_solve": n_a_solve,
        "n_b": n_b,
        "top_k_boundary": top_k_boundary,
        "topk_selected_cost": topk_selected_cost,
        "flag_triggered_single_flip": flag_triggered_single_flip,
        "bounded_two_flip": bounded_two_flip,
        "two_flip_candidates": two_flip_candidates,
        "a_boundary_weight_scale": a_boundary_weight_scale,
        "aba_boundary_feedback": aba_boundary_feedback,
        "aba_max_flips": aba_max_flips,
        "aba_candidate_cols": aba_candidate_cols,
        "a_neighbor_rerank": a_neighbor_rerank,
        "a_rerank_top_k": a_rerank_top_k,
        "a_neighbor_beta": a_neighbor_beta,
        "a_neighbor_rerank_selected_nonzero": a_neighbor_rerank_selected_nonzero,
        "a_neighbor_rerank_total_risk": a_neighbor_rerank_total_risk,
        "a_neighbor_rerank_total_cost": a_neighbor_rerank_total_cost,
        "a_shifted_ensemble": a_shifted_ensemble,
        "a_shift_offsets": list(a_shift_offsets),
        "a_shifted_beta": a_shifted_beta,
        "tsae_top_k": tsae_top_k,
        "tsae_boundary_top_k": tsae_boundary_top_k,
        "tsae_stitch_mode": tsae_stitch_mode,
        "tsae_interface_branch": tsae_interface_branch,
        "tsae_interface_gated": tsae_interface_gated,
        "tsae_interface_cols_per_side": tsae_interface_cols_per_side,
        "tsae_interface_branch_tasks": tsae_interface_branch_tasks,
        "tsae_interface_branch_payloads": tsae_interface_branch_payloads,
        "tsae_interface_branch_cols": tsae_interface_branch_cols,
        "tsae_interface_gated_shots": tsae_interface_gated_shots,
        "tsae_interface_gated_accepted": tsae_interface_gated_accepted,
        "tsae_interface_gated_improved_to_zero": tsae_interface_gated_improved_to_zero,
        "tsae_interface_gated_branch_payloads": tsae_interface_gated_branch_payloads,
        "tsae_interface_gated_b_decodes": tsae_interface_gated_b_decodes,
        "tsae_interface_joint_score": tsae_interface_joint_score,
        "tsae_interface_zb_weight": tsae_interface_zb_weight,
        "tsae_interface_commit_offset": tsae_interface_commit_offset,
        "joint_branch_groups_total": joint_branch_groups_total,
        "joint_branch_b_star_nontrivial": joint_branch_b_star_nontrivial,
        "joint_branch_chose_center_offset": joint_branch_chose_center_offset,
        "a_shifted_valid_offsets": {
            str(index): offsets for index, offsets in a_shifted_valid_offsets.items()
        },
        "a_shifted_selected_nonzero": a_shifted_selected_nonzero,
        "a_shifted_total_risk": a_shifted_total_risk,
        "a_shifted_total_cost": a_shifted_total_cost,
        "a_shifted_candidate_count": a_shifted_candidate_count,
        "tsae_stitch_attempts": tsae_stitch_attempts,
        "tsae_stitch_accepted": tsae_stitch_accepted,
        "tsae_stitch_fallback": tsae_stitch_fallback,
        "tsae_stitch_non_payload": tsae_stitch_non_payload,
        "tsae_stitch_total_score": tsae_stitch_total_score,
        "a_shifted_chain_dp": a_shifted_chain_dp,
        "chain_dp_shots": chain_dp_shots,
        "chain_dp_selected_nonzero": chain_dp_selected_nonzero,
        "chain_dp_final_total_risk": chain_dp_final_total_risk,
        "a_shifted_joint_b_dp": a_shifted_joint_b_dp,
        "a_shifted_joint_b_dp_lag": a_shifted_joint_b_dp_lag,
        "joint_b_dp_frontier_streaming": bool(a_shifted_joint_b_dp and a_shifted_joint_b_dp_lag),
        "a_shifted_joint_flag_penalty": a_shifted_joint_flag_penalty,
        "joint_b_dp_shots": joint_b_dp_shots,
        "joint_b_dp_b_decodes": joint_b_dp_b_decodes,
        "joint_b_dp_physical_residual": joint_b_dp_physical_residual,
        "joint_b_dp_selected_cost": joint_b_dp_selected_cost,
        "joint_b_dp_selected_nonzero": joint_b_dp_selected_nonzero,
        "tsae_diag_in_pool": tsae_diag_in_pool.tolist() if tsae_diag_in_pool is not None else None,
        "tsae_diag_closest_hamming": tsae_diag_closest_hamming.tolist() if tsae_diag_closest_hamming is not None else None,
        "tsae_diag_pool_size": tsae_diag_pool_size.tolist() if tsae_diag_pool_size is not None else None,
        "tsae_diag_picked_offset": tsae_diag_picked_offset.tolist() if tsae_diag_picked_offset is not None else None,
        "tsae_diag_oracle_offset": tsae_diag_oracle_offset.tolist() if tsae_diag_oracle_offset is not None else None,
        "a_two_color_order": a_two_color_order,
        "a_two_color_first_flagged": a_two_color_first_flagged,
        "a_two_color_second_flagged": a_two_color_second_flagged,
        "z_boundary_repair": z_boundary_repair,
        "z_repair_edge_width": z_repair_edge_width,
        "z_repair_attempts": z_repair_attempts,
        "z_repair_accepted": z_repair_accepted,
        "z_repair_success_to_zero": z_repair_success_to_zero,
        "z_repair_no_candidate": z_repair_no_candidate,
        "z_repair_decode_flagged": z_repair_decode_flagged,
        "z_repair_weight_before": z_repair_weight_before,
        "z_repair_weight_after": z_repair_weight_after,
        "z_repair_delta_weight": z_repair_delta_weight,
        "z_repair_candidate_vars": z_repair_candidate_vars,
        "z_repair_flipped_vars": z_repair_flipped_vars,
        "z_repair_left_count": z_repair_left_count,
        "z_repair_right_count": z_repair_right_count,
        "z_repair_both_sides_count": z_repair_both_sides_count,
        "z_joint_retry": z_joint_retry,
        "z_joint_attempts": z_joint_attempts,
        "z_joint_accepted": z_joint_accepted,
        "z_joint_success_to_zero": z_joint_success_to_zero,
        "z_joint_local_flagged": z_joint_local_flagged,
        "z_joint_no_candidate": z_joint_no_candidate,
        "z_joint_weight_before": z_joint_weight_before,
        "z_joint_weight_after": z_joint_weight_after,
        "z_joint_delta_weight": z_joint_delta_weight,
        "z_joint_candidate_vars": z_joint_candidate_vars,
        "z_joint_flipped_vars": z_joint_flipped_vars,
        "z_joint_left_count": z_joint_left_count,
        "z_joint_right_count": z_joint_right_count,
        "z_joint_both_sides_count": z_joint_both_sides_count,
        "aba_initial_b_flagged": locals().get("aba_initial_b_flagged", 0),
        "aba_feedback_attempts": locals().get("aba_feedback_attempts", 0),
        "aba_feedback_proposed": locals().get("aba_feedback_proposed", 0),
        "aba_feedback_accepted": locals().get("aba_feedback_accepted", 0),
        "aba_rerun_windows": locals().get("aba_rerun_windows", 0),
        "a_boundary_penalty_col_count": len(a_boundary_penalty_cols)
        if "a_boundary_penalty_cols" in locals()
        else 0,
        "repair_attempts": locals().get("repair_attempts", 0),
        "repair_accepted": locals().get("repair_accepted", 0),
        "repair_success": locals().get("repair_success", 0),
        "twoflip_attempts": locals().get("twoflip_attempts", 0),
        "twoflip_accepted": locals().get("twoflip_accepted", 0),
        "twoflip_success": locals().get("twoflip_success", 0),
        "soft_boundary_message": soft_boundary_message,
        "soft_boundary_one_prior": soft_boundary_one_prior,
        "soft_boundary_zero_prior": soft_boundary_zero_prior,
        "a_flagged_local": a_flagged,
        "b_flagged_local": b_flagged,
    }
    if seam_diagnostics and use_buffer_aligned:
        final_residual = (det_data + total @ problem.chk.T) % 2
        a_owned_blocks = set()
        for task in a_tasks:
            if task["commit_cols"].start >= task["commit_cols"].stop:
                continue
            commit_region_start = int(np.searchsorted(problem.region_offsets, task["commit_cols"].start))
            commit_region_stop = int(np.searchsorted(problem.region_offsets, task["commit_cols"].stop))
            if commit_region_start >= commit_region_stop:
                continue
            center_region = commit_region_start + (commit_region_stop - commit_region_start) // 2
            center_block = center_region // 2
            if 0 <= center_block < num_blocks:
                a_owned_blocks.add(center_block)

        b_owned_blocks = set()
        for task in b_tasks:
            b_owned_blocks.update(
                range(
                    task["rows"].start // problem.detector_block_size,
                    task["rows"].stop // problem.detector_block_size,
                )
            )
        boundary_blocks = {0, num_blocks - 1}
        all_blocks = set(range(num_blocks))
        unowned_blocks = all_blocks - a_owned_blocks - b_owned_blocks - boundary_blocks

        def rows_for_blocks(blocks: set[int]) -> np.ndarray:
            if not blocks:
                return np.array([], dtype=int)
            return np.concatenate(
                [
                    np.arange(
                        row_slice(problem, block, block + 1).start,
                        row_slice(problem, block, block + 1).stop,
                    )
                    for block in sorted(blocks)
                ]
            )

        def residual_weight_by_blocks(residual_matrix: np.ndarray, blocks: set[int]) -> int:
            rows = rows_for_blocks(blocks)
            if rows.size == 0:
                return 0
            return int(residual_matrix[:, rows].sum())

        def flagged_by_blocks(residual_matrix: np.ndarray, blocks: set[int]) -> int:
            rows = rows_for_blocks(blocks)
            if rows.size == 0:
                return 0
            return int(np.count_nonzero(residual_matrix[:, rows].sum(axis=1)))

        final_block_hist = {}
        r_after_a_block_hist = {}
        for block in range(num_blocks):
            rows = row_slice(problem, block, block + 1)
            final_weight = int(final_residual[:, rows].sum())
            a_weight = int(r_after_a[:, rows].sum())
            if final_weight:
                final_block_hist[f"s{block + 1}"] = final_weight
            if a_weight:
                r_after_a_block_hist[f"s{block + 1}"] = a_weight

        b_local_global_mismatch = 0
        b_local_global_mismatch_by_window = []
        if b_local_residuals is not None:
            for task_index, task in enumerate(b_tasks):
                rows = task["rows"]
                global_part = final_residual[:, rows]
                local_part = b_local_residuals[task_index]
                mismatch = int(np.count_nonzero(np.any(local_part != global_part, axis=1)))
                b_local_global_mismatch += mismatch
                b_local_global_mismatch_by_window.append(
                    {
                        "window": task["row_label"],
                        "mismatch_shots": mismatch,
                        "local_weight": int(local_part.sum()),
                        "global_weight": int(global_part.sum()),
                    }
                )

        b_noisy_boundary_stats = {
            "enabled": bool(b_noisy_boundary),
            "collected": b_noisy_boundary_values is not None,
        }
        if b_noisy_boundary_values is not None:
            boundary_width = problem.detector_block_size
            final_flagged_shots = final_residual.sum(axis=1) > 0
            noisy_used_by_shot = np.zeros(shots, dtype=bool)
            noisy_used_window_shots = 0
            noisy_total_weight = 0
            noisy_left_weight = 0
            noisy_right_weight = 0
            noisy_by_window = []
            for task_index, values in enumerate(b_noisy_boundary_values):
                if values is None:
                    continue
                used = values.sum(axis=1) > 0
                noisy_used_by_shot |= used
                noisy_used_window_shots += int(np.count_nonzero(used))
                noisy_total_weight += int(values.sum())
                left = values[:, :boundary_width]
                right = values[:, boundary_width:]
                left_weight = int(left.sum())
                right_weight = int(right.sum())
                noisy_left_weight += left_weight
                noisy_right_weight += right_weight
                noisy_by_window.append(
                    {
                        "window": b_tasks[task_index]["row_label"],
                        "used_shots": int(np.count_nonzero(used)),
                        "weight": int(values.sum()),
                        "left_weight": left_weight,
                        "right_weight": right_weight,
                    }
                )

            b_noisy_boundary_stats = {
                "enabled": True,
                "collected": True,
                "used_shots": int(np.count_nonzero(noisy_used_by_shot)),
                "used_rate": float(np.count_nonzero(noisy_used_by_shot) / shots) if shots else 0.0,
                "used_window_shots": noisy_used_window_shots,
                "total_weight": noisy_total_weight,
                "left_weight": noisy_left_weight,
                "right_weight": noisy_right_weight,
                "final_flagged_overlap_shots": int(
                    np.count_nonzero(noisy_used_by_shot & final_flagged_shots)
                ),
                "final_flagged_without_noisy_shots": int(
                    np.count_nonzero(final_flagged_shots & ~noisy_used_by_shot)
                ),
                "noisy_without_final_flagged_shots": int(
                    np.count_nonzero(noisy_used_by_shot & ~final_flagged_shots)
                ),
                "by_window": noisy_by_window,
            }

        diagnostics["seam_diagnostics"] = {
            "a_owned_rows": [f"s{block + 1}" for block in sorted(a_owned_blocks)],
            "b_owned_rows": [f"s{block + 1}" for block in sorted(b_owned_blocks)],
            "boundary_rows": [f"s{block + 1}" for block in sorted(boundary_blocks)],
            "unowned_rows": [f"s{block + 1}" for block in sorted(unowned_blocks)],
            "r_after_a_weight": {
                "A_owned_rows": residual_weight_by_blocks(r_after_a, a_owned_blocks),
                "B_owned_rows": residual_weight_by_blocks(r_after_a, b_owned_blocks),
                "boundary_rows": residual_weight_by_blocks(r_after_a, boundary_blocks),
                "unowned_rows": residual_weight_by_blocks(r_after_a, unowned_blocks),
            },
            "r_after_a_flagged": {
                "A_owned_rows": flagged_by_blocks(r_after_a, a_owned_blocks),
                "B_owned_rows": flagged_by_blocks(r_after_a, b_owned_blocks),
                "boundary_rows": flagged_by_blocks(r_after_a, boundary_blocks),
                "unowned_rows": flagged_by_blocks(r_after_a, unowned_blocks),
            },
            "final_weight": {
                "A_owned_rows": residual_weight_by_blocks(final_residual, a_owned_blocks),
                "B_owned_rows": residual_weight_by_blocks(final_residual, b_owned_blocks),
                "boundary_rows": residual_weight_by_blocks(final_residual, boundary_blocks),
                "unowned_rows": residual_weight_by_blocks(final_residual, unowned_blocks),
            },
            "final_flagged": {
                "A_owned_rows": flagged_by_blocks(final_residual, a_owned_blocks),
                "B_owned_rows": flagged_by_blocks(final_residual, b_owned_blocks),
                "boundary_rows": flagged_by_blocks(final_residual, boundary_blocks),
                "unowned_rows": flagged_by_blocks(final_residual, unowned_blocks),
            },
            "final_residual_weight_mean": float(final_residual.sum(axis=1).mean()),
            "final_residual_weight_max": int(final_residual.sum(axis=1).max()) if shots else 0,
            "nonzero_final_residual_row_count": int(np.count_nonzero(final_residual.sum(axis=0))),
            "r_after_a_block_hist": r_after_a_block_hist,
            "final_block_hist": final_block_hist,
            "b_local_global_mismatch_shots": b_local_global_mismatch,
            "b_local_global_mismatch_by_window": b_local_global_mismatch_by_window,
            "b_noisy_boundary": b_noisy_boundary_stats,
        }
    return total, diagnostics
