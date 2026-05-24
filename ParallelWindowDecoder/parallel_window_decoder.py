import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDING_ROOT = ROOT / "SlidingWindowDecoder"
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
sys.path.insert(0, str(SLIDING_ROOT))

import numpy as np
from ldpc import BpOsdDecoder
import stim

from src.build_circuit import build_circuit, dem_to_check_matrices  # noqa: E402
from src.codes_q import create_bivariate_bicycle_codes  # noqa: E402

try:
    from src import osd_window as ShortenedOsdWindow  # noqa: E402
except ImportError:
    ShortenedOsdWindow = None

try:
    from osd_list import osd_list_candidates as cython_osd_list_candidates
except ImportError:
    cython_osd_list_candidates = None


@dataclass
class PreparedProblem:
    chk: np.ndarray
    obs: np.ndarray
    priors: np.ndarray
    ordered_cols: list[int]
    raw_error_to_col: list[int]
    region_offsets: list[int]
    num_detector_blocks: int
    detector_block_size: int
    z_basis: bool
    dem: object


def make_code(N: int):
    if N == 72:
        return create_bivariate_bicycle_codes(6, 6, [3], [1, 2], [1, 2], [3])
    if N == 90:
        return create_bivariate_bicycle_codes(15, 3, [9], [1, 2], [2, 7], [0])
    if N == 108:
        return create_bivariate_bicycle_codes(9, 6, [3], [1, 2], [1, 2], [3])
    if N == 144:
        return create_bivariate_bicycle_codes(12, 6, [3], [1, 2], [1, 2], [3])
    if N == 288:
        return create_bivariate_bicycle_codes(12, 12, [3], [2, 7], [1, 2], [3])
    if N == 360:
        return create_bivariate_bicycle_codes(30, 6, [9], [1, 2], [25, 26], [3])
    if N == 756:
        return create_bivariate_bicycle_codes(21, 18, [3], [10, 17], [3, 19], [5])
    raise ValueError(f"unsupported N={N}")


def error_key(targets: list[stim.DemTarget]) -> str:
    detectors = []
    observables = []
    for target in targets:
        if target.is_relative_detector_id():
            detectors.append(target.val)
        elif target.is_logical_observable_id():
            observables.append(target.val)
    return " ".join(
        [f"D{det}" for det in sorted(detectors)]
        + [f"L{obs}" for obs in sorted(observables)]
    )


def raw_error_to_collapsed_cols(dem: stim.DetectorErrorModel, col_dict: dict[str, int]) -> list[int]:
    mapping = []
    for instruction in dem.flattened():
        if instruction.type == "error":
            mapping.append(col_dict[error_key(instruction.targets_copy())])
    return mapping


def prepare_problem(N: int, p: float, num_repeat: int, z_basis: bool) -> PreparedProblem:
    code, A_list, B_list = make_code(N)
    circuit = build_circuit(code, A_list, B_list, p, num_repeat, z_basis=z_basis)
    dem = circuit.detector_error_model()
    chk_sparse, obs_sparse, priors_raw, col_dict = dem_to_check_matrices(dem, return_col_dict=True)
    raw_error_to_col = raw_error_to_collapsed_cols(dem, col_dict)

    num_row, num_col = chk_sparse.shape
    detector_block_size = code.N // 2
    num_detector_blocks = num_row // detector_block_size

    region_cols = [[] for _ in range(2 * num_detector_blocks - 1)]
    for col in range(num_col):
        nz = np.nonzero(chk_sparse[:, col])[0]
        first_block = int(nz.min() // detector_block_size)
        last_block = int(nz.max() // detector_block_size)
        if first_block == last_block:
            region = 2 * first_block
        elif last_block == first_block + 1:
            region = 2 * first_block + 1
        else:
            raise ValueError(
                f"column {col} spans non-local detector blocks "
                f"{first_block}..{last_block}"
            )
        region_cols[region].append(col)

    ordered_cols = [col for cols in region_cols for col in cols]
    chk = chk_sparse[:, ordered_cols].toarray().astype(np.uint8)
    obs = obs_sparse[:, ordered_cols].toarray().astype(np.uint8)
    priors = np.asarray(priors_raw[ordered_cols], dtype=float)

    region_offsets = [0]
    for cols in region_cols:
        region_offsets.append(region_offsets[-1] + len(cols))

    return PreparedProblem(
        chk=chk,
        obs=obs,
        priors=priors,
        ordered_cols=ordered_cols,
        raw_error_to_col=raw_error_to_col,
        region_offsets=region_offsets,
        num_detector_blocks=num_detector_blocks,
        detector_block_size=detector_block_size,
        z_basis=z_basis,
        dem=dem,
    )


def region_col_slice(problem: PreparedProblem, region_start: int, region_stop: int) -> slice:
    return slice(problem.region_offsets[region_start], problem.region_offsets[region_stop])


def row_slice(problem: PreparedProblem, block_start: int, block_stop: int) -> slice:
    n = problem.detector_block_size
    return slice(block_start * n, block_stop * n)


def make_bp_osd(
    mat: np.ndarray,
    priors: np.ndarray,
    max_iter: int,
    osd_order: int,
    shorten: bool = False,
    shorten_pre_max_iter: int = 8,
):
    if shorten:
        if ShortenedOsdWindow is None:
            raise RuntimeError("src.osd_window is not available; rebuild SlidingWindowDecoder extensions first")
        return ShortenedOsdWindow(
            mat,
            channel_probs=priors,
            pre_max_iter=shorten_pre_max_iter,
            post_max_iter=max_iter,
            ms_scaling_factor=1.0,
            new_n=None,
            osd_method="osd_cs",
            osd_order=max(0, osd_order),
        )
    return BpOsdDecoder(
        mat,
        channel_probs=list(priors),
        max_iter=max_iter,
        bp_method="minimum_sum",
        ms_scaling_factor=1.0,
        osd_method="OSD_CS",
        osd_order=osd_order,
    )


def error_cost(bits: np.ndarray, priors: np.ndarray) -> float:
    priors = np.clip(priors, 1e-12, 1 - 1e-12)
    return float(np.dot(bits, np.log((1 - priors) / priors)))


def gf2_osd_list_candidates(
    mat: np.ndarray,
    syndrome: np.ndarray,
    prior: np.ndarray,
    seed: np.ndarray,
    reliabilities: np.ndarray,
    top_k: int,
) -> list[np.ndarray]:
    if top_k <= 1:
        return [seed.astype(np.uint8, copy=True)]

    order = np.argsort(-np.asarray(reliabilities, dtype=float))
    h = mat[:, order].astype(np.uint8, copy=True)
    rhs = syndrome.astype(np.uint8, copy=True)
    rows, cols = h.shape
    pivot_cols = []
    pivot_rows = []
    row = 0

    for col in range(cols):
        candidates = np.flatnonzero(h[row:, col])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            h[[row, pivot]] = h[[pivot, row]]
            rhs[[row, pivot]] = rhs[[pivot, row]]
        other_rows = np.flatnonzero(h[:, col])
        for other in other_rows:
            if other != row:
                h[other] ^= h[row]
                rhs[other] ^= rhs[row]
        pivot_cols.append(col)
        pivot_rows.append(row)
        row += 1
        if row == rows:
            break

    if row < rows:
        inconsistent = np.flatnonzero((h[row:].sum(axis=1) == 0) & (rhs[row:] == 1))
        if inconsistent.size:
            return [seed.astype(np.uint8, copy=True)]

    pivot_set = set(pivot_cols)
    free_cols = [col for col in range(cols) if col not in pivot_set]
    seed_perm = seed[order].astype(np.uint8, copy=True)
    base_free = seed_perm[free_cols].copy()
    free_reliability = np.asarray(reliabilities, dtype=float)[order][free_cols]
    low_free_order = np.argsort(free_reliability)
    search_free_count = min(len(low_free_order), max(4, top_k + 2))

    flip_sets: list[tuple[int, ...]] = [()]
    for idx in range(search_free_count):
        flip_sets.append((int(low_free_order[idx]),))
    for i in range(search_free_count):
        for j in range(i + 1, search_free_count):
            flip_sets.append((int(low_free_order[i]), int(low_free_order[j])))
            if len(flip_sets) >= max(4 * top_k, top_k + 1):
                break
        if len(flip_sets) >= max(4 * top_k, top_k + 1):
            break

    scored: list[tuple[float, bytes, np.ndarray]] = []
    for flips in flip_sets:
        x_perm = np.zeros(cols, dtype=np.uint8)
        free_values = base_free.copy()
        for flip in flips:
            free_values[flip] ^= 1
        x_perm[free_cols] = free_values
        for pivot_col, pivot_row in reversed(list(zip(pivot_cols, pivot_rows))):
            parity = int(np.bitwise_and(h[pivot_row, free_cols], free_values).sum() % 2)
            x_perm[pivot_col] = rhs[pivot_row] ^ parity
        x = np.zeros(cols, dtype=np.uint8)
        x[order] = x_perm
        if ((mat @ x + syndrome) % 2).any():
            continue
        scored.append((error_cost(x, prior), x.tobytes(), x))

    if not scored:
        return [seed.astype(np.uint8, copy=True)]

    scored.sort(key=lambda item: item[0])
    unique = []
    seen = set()
    for _, key, candidate in scored:
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) == top_k:
            break
    while len(unique) < top_k:
        unique.append(unique[-1].copy())
    return unique


def decode_window_payload(payload: dict):
    mat = payload["mat"]
    prior = payload["prior"]
    syndromes = payload["syndromes"]
    max_iter = payload["max_iter"]
    osd_order = payload["osd_order"]
    shorten = payload.get("shorten", False)
    shorten_pre_max_iter = payload.get("shorten_pre_max_iter", 8)
    value_start = payload.get("value_start")
    value_stop = payload.get("value_stop")
    value_limit = payload.get("value_limit", mat.shape[1])

    decoder = make_bp_osd(mat, prior, max_iter, osd_order, shorten, shorten_pre_max_iter)
    shots = syndromes.shape[0]
    if value_start is None:
        values = np.zeros((shots, value_limit), dtype=np.uint8)
    else:
        values = np.zeros((shots, value_stop - value_start), dtype=np.uint8)

    flagged = 0
    for shot in range(shots):
        local_e = decoder.decode(syndromes[shot]).astype(np.uint8)
        if ((mat @ local_e + syndromes[shot]) % 2).any():
            flagged += 1
        if value_start is None:
            values[shot] = local_e[:value_limit]
        else:
            values[shot] = local_e[value_start:value_stop]
    return payload["task_index"], values, flagged


def decode_a_candidate_payload(payload: dict):
    mat = payload["mat"]
    prior = payload["prior"]
    syndromes = payload["syndromes"]
    max_iter = payload["max_iter"]
    osd_order = payload["osd_order"]
    shorten = payload.get("shorten", False)
    shorten_pre_max_iter = payload.get("shorten_pre_max_iter", 8)
    value_start = payload["value_start"]
    value_stop = payload["value_stop"]
    top_k = payload["top_k"]

    decoder = make_bp_osd(mat, prior, max_iter, osd_order, shorten, shorten_pre_max_iter)
    shots = syndromes.shape[0]
    width = value_stop - value_start
    values = np.zeros((shots, top_k, width), dtype=np.uint8)
    costs = np.full((shots, top_k), np.inf, dtype=float)
    flagged = 0

    for shot in range(shots):
        if hasattr(decoder, "decode_list"):
            candidates = decoder.decode_list(
                syndromes[shot],
                top_k=top_k,
                force_osd=True,
            ).astype(np.uint8)
            local_e = candidates[0]
        else:
            local_e = decoder.decode(syndromes[shot]).astype(np.uint8)
            reliabilities = np.abs(np.asarray(decoder.log_prob_ratios, dtype=float))
            if cython_osd_list_candidates is not None:
                candidates = cython_osd_list_candidates(
                    np.ascontiguousarray(mat, dtype=np.uint8),
                    np.ascontiguousarray(syndromes[shot], dtype=np.uint8),
                    np.ascontiguousarray(prior, dtype=np.float64),
                    np.ascontiguousarray(local_e, dtype=np.uint8),
                    np.ascontiguousarray(reliabilities, dtype=np.float64),
                    top_k,
                )
            else:
                candidates = gf2_osd_list_candidates(
                    mat,
                    syndromes[shot],
                    prior,
                    local_e,
                    reliabilities,
                    top_k,
                )

        residual = (mat @ local_e + syndromes[shot]) % 2
        residual_weight = int(residual.sum())
        if residual_weight:
            flagged += 1

        for cand, candidate_full in enumerate(candidates[:top_k]):
            candidate_commit = candidate_full[value_start:value_stop].copy()
            values[shot, cand] = candidate_commit
            costs[shot, cand] = error_cost(candidate_full, prior)

    return payload["task_index"], values, costs, flagged


def run_window_payloads(payloads: list[dict], parallel_workers: int, parallel_backend: str):
    if parallel_workers <= 1 or len(payloads) <= 1:
        return [decode_window_payload(payload) for payload in payloads]
    executor_cls = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=parallel_workers) as executor:
        return list(executor.map(decode_window_payload, payloads))


def run_a_candidate_payloads(payloads: list[dict], parallel_workers: int, parallel_backend: str):
    if parallel_workers <= 1 or len(payloads) <= 1:
        return [decode_a_candidate_payload(payload) for payload in payloads]
    executor_cls = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=parallel_workers) as executor:
        return list(executor.map(decode_a_candidate_payload, payloads))


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


def sample_dem(problem: PreparedProblem, shots: int, seed: int | None):
    sampler = problem.dem.compile_sampler(seed=seed)
    det_data, obs_data, err_data = sampler.sample(
        shots=shots,
        return_errors=True,
        bit_packed=False,
    )
    collapsed = np.zeros((shots, problem.chk.shape[1]), dtype=np.uint8)
    for raw_col, collapsed_col in enumerate(problem.raw_error_to_col):
        collapsed[:, collapsed_col] ^= err_data[:, raw_col].astype(np.uint8)
    collapsed = collapsed[:, problem.ordered_cols]
    return det_data.astype(np.uint8), obs_data.astype(np.uint8), collapsed


def score(problem: PreparedProblem, det_data: np.ndarray, obs_data: np.ndarray, e_hat: np.ndarray):
    residual = (det_data + e_hat @ problem.chk.T) % 2
    flagged = residual.any(axis=1)
    logical = ((obs_data + e_hat @ problem.obs.T) % 2).any(axis=1)
    failed = np.logical_or(flagged, logical)
    return {
        "flagged": int(flagged.sum()),
        "logical_or_flagged": int(failed.sum()),
        "shots": int(det_data.shape[0]),
        "ler": float(failed.mean()),
    }


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
    """Sliding-window BP+OSD baseline matching SlidingWindowDecoder/osd.py.

    This mirrors the original notebook/script structure: build overlapping
    windows from anchors, add method=1 noisy-syndrome boundary columns on
    non-final windows, commit only the left F=1 block, and update the residual.
    """
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
):
    shots = det_data.shape[0]
    total = np.zeros((shots, problem.chk.shape[1]), dtype=np.uint8)
    num_blocks = problem.num_detector_blocks
    num_regions = len(problem.region_offsets) - 1
    if n_a_solve < n_a:
        raise ValueError("n_a_solve must be >= n_a")
    if top_k_boundary > 1 and b_noisy_boundary:
        raise ValueError("top_k_boundary > 1 is not supported with b_noisy_boundary")
    use_buffer_aligned = n_a == 3 and n_a_solve == n_b and n_a_solve % 2 == 1
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

    a_payloads = []
    for task_index, task in enumerate(a_tasks):
        rows = task["rows"]
        cols = task["cols"]
        commit_cols = task["commit_cols"]
        mat = problem.chk[rows, cols]
        prior = problem.priors[cols]
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
        a_payloads.append(
            {
                "task_index": task_index,
                "mat": mat,
                "prior": prior,
                "syndromes": det_data[:, rows],
                "max_iter": max_iter,
                "osd_order": osd_order,
                "shorten": window_shorten,
                "shorten_pre_max_iter": shorten_pre_max_iter,
                "value_start": local_commit_start,
                "value_stop": local_commit_stop,
            }
        )

    a_flagged = 0
    if oracle_errors is not None:
        for task in a_tasks:
            total[:, task["commit_cols"]] = oracle_errors[:, task["commit_cols"]]
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
    )

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
    elif oracle_errors is not None or top_k_boundary <= 1 or not use_buffer_aligned:
        residual = (det_data + total @ problem.chk.T) % 2

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

        b_flagged = 0
        for task_index, values, flagged in run_window_payloads(
            b_payloads,
            parallel_workers,
            parallel_backend,
        ):
            total[:, b_tasks[task_index]["cols"]] = values
            b_flagged += flagged
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
        "soft_boundary_message": soft_boundary_message,
        "soft_boundary_one_prior": soft_boundary_one_prior,
        "soft_boundary_zero_prior": soft_boundary_zero_prior,
        "a_flagged_local": a_flagged,
        "b_flagged_local": b_flagged,
    }
    return total, diagnostics


def print_result(name: str, metrics: dict, elapsed: float, extra: dict | None = None):
    print(f"\n{name}")
    print(f"  elapsed_sec: {elapsed:.3f}")
    print(f"  flagged: {metrics['flagged']}/{metrics['shots']}")
    print(f"  logical_or_flagged: {metrics['logical_or_flagged']}/{metrics['shots']}")
    print(f"  LER: {metrics['ler']:.6g}")
    if extra:
        for key, value in extra.items():
            print(f"  {key}: {value}")


def s_range(start_block: int, stop_block: int) -> str:
    start = start_block + 1
    stop = stop_block
    if start == stop:
        return f"s{start}"
    return f"s{start}-s{stop}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=72)
    parser.add_argument("--p", type=float, default=0.003)
    parser.add_argument("--num-repeat", type=int, default=4)
    parser.add_argument("--num-shots", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--osd-order", type=int, default=0)
    parser.add_argument("--sliding-width", type=int, default=3)
    parser.add_argument("--b-width", type=int, default=2)
    parser.add_argument("--a-radius", type=int, default=1)
    parser.add_argument("--a-size", type=int, default=None)
    parser.add_argument("--a-solve-size", type=int, default=None)
    parser.add_argument("--x-basis", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-global", action="store_true")
    parser.add_argument("--run-sliding", action="store_true")
    parser.add_argument("--oracle-boundaries", action="store_true")
    parser.add_argument("--a-noisy-boundary", action="store_true")
    parser.add_argument("--b-noisy-boundary", action="store_true")
    parser.add_argument("--window-shorten", action="store_true")
    parser.add_argument("--shorten-pre-max-iter", type=int, default=8)
    parser.add_argument("--sliding-method", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["thread", "process"], default="thread")
    parser.add_argument("--top-k-boundary", type=int, default=1)
    parser.add_argument("--soft-boundary-message", action="store_true")
    parser.add_argument("--soft-boundary-one-prior", type=float, default=0.2)
    parser.add_argument("--soft-boundary-zero-prior", type=float, default=1e-4)
    args = parser.parse_args()

    problem = prepare_problem(args.N, args.p, args.num_repeat, z_basis=not args.x_basis)
    print(
        "problem:",
        f"chk={problem.chk.shape}",
        f"obs={problem.obs.shape}",
        f"detector_blocks={problem.num_detector_blocks}",
        f"regions={len(problem.region_offsets) - 1}",
    )

    t0 = time.perf_counter()
    det_data, obs_data, err_data = sample_dem(problem, args.num_shots, args.seed)
    print(f"sampled {args.num_shots} shots in {time.perf_counter() - t0:.3f}s")

    if args.run_global:
        t0 = time.perf_counter()
        e_hat = decode_global(problem, det_data, args.max_iter, args.osd_order)
        print_result("global BP+OSD", score(problem, det_data, obs_data, e_hat), time.perf_counter() - t0)

    if args.run_sliding:
        t0 = time.perf_counter()
        e_hat = decode_sliding(
            problem,
            det_data,
            args.sliding_width,
            args.max_iter,
            args.osd_order,
            method=args.sliding_method,
            shorten=args.window_shorten,
            shorten_pre_max_iter=args.shorten_pre_max_iter,
        )
        print_result("sequential sliding BP+OSD", score(problem, det_data, obs_data, e_hat), time.perf_counter() - t0)

    t0 = time.perf_counter()
    e_hat, diagnostics = decode_parallel_ab(
        problem,
        det_data,
        args.b_width,
        args.a_radius,
        args.a_size,
        args.a_solve_size,
        args.max_iter,
        args.osd_order,
        oracle_errors=err_data if args.oracle_boundaries else None,
        a_noisy_boundary=args.a_noisy_boundary,
        parallel_workers=args.parallel_workers,
        parallel_backend=args.parallel_backend,
        top_k_boundary=args.top_k_boundary,
        soft_boundary_message=args.soft_boundary_message,
        soft_boundary_one_prior=args.soft_boundary_one_prior,
        soft_boundary_zero_prior=args.soft_boundary_zero_prior,
        window_shorten=args.window_shorten,
        shorten_pre_max_iter=args.shorten_pre_max_iter,
        b_noisy_boundary=args.b_noisy_boundary,
    )
    print_result("parallel A/B BP+OSD", score(problem, det_data, obs_data, e_hat), time.perf_counter() - t0, diagnostics)


if __name__ == "__main__":
    main()
