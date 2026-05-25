import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
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


def scale_priors_by_llr(priors: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0 or not np.any(mask):
        return priors
    clipped = np.clip(priors.astype(float, copy=True), 1e-12, 1 - 1e-12)
    llr = np.log((1 - clipped) / clipped)
    llr[mask] *= scale
    return np.clip(1 / (1 + np.exp(llr)), 1e-12, 1 - 1e-12)


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


def decoder_reliability(decoder, width: int) -> np.ndarray:
    try:
        raw = np.asarray(decoder.log_prob_ratios, dtype=float)
    except Exception:
        return np.full(width, np.inf, dtype=float)
    if raw.ndim == 2:
        rel = np.abs(raw.sum(axis=1))
    else:
        rel = np.abs(raw)
    if rel.size < width:
        out = np.full(width, np.inf, dtype=float)
        out[: rel.size] = rel
        return out
    return rel[:width]


def choose_boundary_feedback_flips(
    chk: np.ndarray,
    rows: slice,
    boundary_cols: np.ndarray,
    residual: np.ndarray,
    max_flips: int,
    candidate_cols: int,
) -> tuple[list[int], int]:
    residual_weight = int(residual.sum())
    if residual_weight == 0 or boundary_cols.size == 0 or max_flips <= 0:
        return [], residual_weight

    local_cols = np.asarray(chk[rows, boundary_cols], dtype=np.uint8)
    if local_cols.ndim != 2 or local_cols.shape[1] == 0:
        return [], residual_weight

    overlap = residual.astype(np.int16) @ local_cols.astype(np.int16)
    support = local_cols.sum(axis=0)
    score = 2 * overlap - support
    order = np.argsort(-score)
    positive = order[score[order] > 0]
    if positive.size == 0:
        positive = order[: min(candidate_cols, order.size)]
    else:
        positive = positive[: min(candidate_cols, positive.size)]

    best_weight = residual_weight
    best_cols: list[int] = []
    max_order = min(max_flips, positive.size)
    for flip_count in range(1, max_order + 1):
        for combo in itertools.combinations(positive, flip_count):
            contribution = np.bitwise_xor.reduce(local_cols[:, combo], axis=1)
            weight = int(((residual + contribution) % 2).sum())
            if weight < best_weight:
                best_weight = weight
                best_cols = [int(boundary_cols[index]) for index in combo]
                if best_weight == 0:
                    return best_cols, best_weight
    return best_cols, best_weight


def unique_concat(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [array.astype(int, copy=False) for array in arrays if array.size]
    if not nonempty:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(nonempty))


def decode_a_repair_payload(payload: dict):
    mat = payload["mat"]
    prior = payload["prior"]
    syndromes = payload["syndromes"]
    max_iter = payload["max_iter"]
    osd_order = payload["osd_order"]
    shorten = payload.get("shorten", False)
    shorten_pre_max_iter = payload.get("shorten_pre_max_iter", 8)
    value_start = payload["value_start"]
    value_stop = payload["value_stop"]

    decoder = make_bp_osd(mat, prior, max_iter, osd_order, shorten, shorten_pre_max_iter)
    shots = syndromes.shape[0]
    width = value_stop - value_start
    values = np.zeros((shots, width), dtype=np.uint8)
    reliabilities = np.full((shots, width), np.inf, dtype=float)

    flagged = 0
    for shot in range(shots):
        local_e = decoder.decode(syndromes[shot]).astype(np.uint8)
        if ((mat @ local_e + syndromes[shot]) % 2).any():
            flagged += 1
        values[shot] = local_e[value_start:value_stop]
        reliabilities[shot] = decoder_reliability(decoder, mat.shape[1])[value_start:value_stop]

    return payload["task_index"], values, reliabilities, flagged


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


def run_a_repair_payloads(payloads: list[dict], parallel_workers: int, parallel_backend: str):
    if parallel_workers <= 1 or len(payloads) <= 1:
        return [decode_a_repair_payload(payload) for payload in payloads]
    executor_cls = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=parallel_workers) as executor:
        return list(executor.map(decode_a_repair_payload, payloads))


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
    a_shifted_chain_dp: bool = False,
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
            a_shifted_chain_dp=a_shifted_chain_dp,
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
    a_shifted_chain_dp: bool = False,
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
    if a_shifted_chain_dp and not a_shifted_ensemble:
        raise ValueError("a_shifted_chain_dp requires a_shifted_ensemble")
    if a_shifted_chain_dp and tsae_stitch_mode != "none":
        raise ValueError("a_shifted_chain_dp is mutually exclusive with tsae_stitch_mode")
    if a_micro_sliding and not a_shifted_ensemble:
        raise ValueError("a_micro_sliding requires a_shifted_ensemble")
    if a_micro_sliding and (tsae_stitch_mode != "none" or a_shifted_chain_dp):
        raise ValueError("a_micro_sliding is mutually exclusive with tsae_stitch_mode and a_shifted_chain_dp")
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
        or a_shifted_ensemble
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
    chain_dp_shots = 0
    chain_dp_selected_nonzero = 0
    chain_dp_final_total_risk = 0.0
    chain_dp_total_pair_weight = 0.0
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

        for a_index, task in enumerate(a_tasks):
            seen_offsets: set[int] = set()
            for offset in a_shift_offsets:
                if offset in seen_offsets:
                    continue
                seen_offsets.add(int(offset))
                payload = shifted_a_payload(a_index, task, int(offset), len(candidate_payloads))
                if payload is None:
                    continue
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
            for payload_index, values, costs, flagged in run_a_candidate_payloads(
                candidate_payloads,
                parallel_workers,
                parallel_backend,
            ):
                for cand_index in range(values.shape[1]):
                    shifted_values.append(values[:, cand_index, :])
                    shifted_costs.append(costs[:, cand_index])
                    expanded_payload_to_a.append(payload_to_a[payload_index])
                    expanded_payload_offsets.append(payload_offsets[payload_index])
                a_flagged += flagged

            payload_to_a = expanded_payload_to_a
            payload_offsets = expanded_payload_offsets
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
    )
    a_only_total = total.copy()
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
                        z_repair_flipped_vars += int(delta.sum())
                        if not delta.any():
                            z_repair_weight_after += before_weight
                            continue

                        contribution = (delta @ problem.chk[:, candidate_cols].T) % 2
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
    parser.add_argument("--flag-triggered-single-flip", action="store_true")
    parser.add_argument("--bounded-two-flip", action="store_true")
    parser.add_argument("--two-flip-candidates", type=int, default=4)
    parser.add_argument("--a-boundary-weight-scale", type=float, default=1.0)
    parser.add_argument("--aba-boundary-feedback", action="store_true")
    parser.add_argument("--aba-max-flips", type=int, default=2)
    parser.add_argument("--aba-candidate-cols", type=int, default=16)
    parser.add_argument("--a-neighbor-rerank", action="store_true")
    parser.add_argument("--a-rerank-top-k", type=int, default=2)
    parser.add_argument("--a-neighbor-beta", type=float, default=1.0)
    parser.add_argument("--a-two-color-order", choices=["none", "odd-first", "even-first"], default="none")
    parser.add_argument("--a-shifted-ensemble", action="store_true")
    parser.add_argument("--a-shift-offsets", default="-2,0,2")
    parser.add_argument("--a-shifted-beta", type=float, default=1.0)
    parser.add_argument("--tsae-top-k", type=int, default=1)
    parser.add_argument("--tsae-boundary-top-k", type=int, default=1)
    parser.add_argument("--tsae-stitch-mode", choices=["none", "component", "pairwise"], default="none")
    parser.add_argument("--a-shifted-chain-dp", action="store_true")
    parser.add_argument("--a-micro-sliding", action="store_true")
    parser.add_argument("--a-micro-sliding-order", choices=["asc", "desc", "center-out"], default="asc")
    parser.add_argument("--tsae-oracle-diag", action="store_true")
    parser.add_argument("--z-boundary-repair", action="store_true")
    parser.add_argument("--z-repair-edge-width", type=int, default=2)
    parser.add_argument("--z-joint-retry", action="store_true")
    parser.add_argument("--seam-diagnostics", action="store_true")
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
        flag_triggered_single_flip=args.flag_triggered_single_flip,
        bounded_two_flip=args.bounded_two_flip,
        two_flip_candidates=args.two_flip_candidates,
        a_boundary_weight_scale=args.a_boundary_weight_scale,
        aba_boundary_feedback=args.aba_boundary_feedback,
        aba_max_flips=args.aba_max_flips,
        aba_candidate_cols=args.aba_candidate_cols,
        a_neighbor_rerank=args.a_neighbor_rerank,
        a_rerank_top_k=args.a_rerank_top_k,
        a_neighbor_beta=args.a_neighbor_beta,
        a_two_color_order=args.a_two_color_order,
        a_shifted_ensemble=args.a_shifted_ensemble,
        a_shift_offsets=tuple(int(item) for item in args.a_shift_offsets.split(",") if item.strip()),
        a_shifted_beta=args.a_shifted_beta,
        tsae_top_k=args.tsae_top_k,
        tsae_boundary_top_k=args.tsae_boundary_top_k,
        tsae_stitch_mode=args.tsae_stitch_mode,
        a_shifted_chain_dp=args.a_shifted_chain_dp,
        a_micro_sliding=args.a_micro_sliding,
        a_micro_sliding_order=args.a_micro_sliding_order,
        oracle_diag=err_data if args.tsae_oracle_diag else None,
        z_boundary_repair=args.z_boundary_repair,
        z_repair_edge_width=args.z_repair_edge_width,
        z_joint_retry=args.z_joint_retry,
        seam_diagnostics=args.seam_diagnostics,
    )
    print_result("parallel A/B BP+OSD", score(problem, det_data, obs_data, e_hat), time.perf_counter() - t0, diagnostics)

    if args.tsae_oracle_diag and diagnostics.get("tsae_diag_in_pool") is not None:
        in_pool = np.array(diagnostics["tsae_diag_in_pool"], dtype=bool)
        closest = np.array(diagnostics["tsae_diag_closest_hamming"], dtype=int)
        pool_size = np.array(diagnostics["tsae_diag_pool_size"], dtype=int)
        picked_off = np.array(diagnostics["tsae_diag_picked_offset"], dtype=int)
        oracle_off = np.array(diagnostics["tsae_diag_oracle_offset"], dtype=int)
        shots_n = in_pool.shape[0]
        interior_idx = np.where(pool_size > 1)[0]

        # per-shot outcome
        synd_res = (det_data + e_hat @ problem.chk.T) % 2
        flagged_mask = synd_res.any(axis=1)
        logical_res = (obs_data + e_hat @ problem.obs.T) % 2
        logical_mask = logical_res.any(axis=1)
        failed_mask = flagged_mask | logical_mask

        boundary_idx = np.where(pool_size == 1)[0]
        all_in_pool_interior = (
            in_pool[:, interior_idx].all(axis=1) if interior_idx.size > 0
            else np.ones(shots_n, dtype=bool)
        )
        all_in_pool_boundary = (
            in_pool[:, boundary_idx].all(axis=1) if boundary_idx.size > 0
            else np.ones(shots_n, dtype=bool)
        )
        all_in_pool_full = all_in_pool_interior & all_in_pool_boundary

        print()
        print("=== TSAE Oracle Diagnostic ===")
        print(f"shots={shots_n}, pool sizes per A={pool_size.tolist()}, "
              f"interior={interior_idx.tolist()}, boundary={boundary_idx.tolist()}")
        print(f"failures: total={int(failed_mask.sum())}, "
              f"flagged_only={int((flagged_mask & ~logical_mask).sum())}, "
              f"logical_only={int((logical_mask & ~flagged_mask).sum())}, "
              f"both={int((flagged_mask & logical_mask).sum())}")
        print(f"oracle ∈ pool: all-A={int(all_in_pool_full.sum())}/{shots_n}, "
              f"interior-only={int(all_in_pool_interior.sum())}/{shots_n}, "
              f"boundary-only={int(all_in_pool_boundary.sum())}/{shots_n}")
        print()
        # use full criterion (all A's including boundaries)
        all_in_pool = all_in_pool_full
        print(f"For {int(failed_mask.sum())} failed shots (full criterion = all A's including boundaries):")
        sel_fail = int((failed_mask & all_in_pool).sum())
        div_fail = int((failed_mask & ~all_in_pool).sum())
        only_int_fail = int((failed_mask & ~all_in_pool_interior & all_in_pool_boundary).sum())
        only_bdry_fail = int((failed_mask & all_in_pool_interior & ~all_in_pool_boundary).sum())
        both_div_fail = int((failed_mask & ~all_in_pool_interior & ~all_in_pool_boundary).sum())
        print(f"  selection failure  (all-A oracle ∈ pool, but picked wrong):     {sel_fail}")
        print(f"  diversity failure  (some A oracle ∉ pool):                      {div_fail}")
        print(f"    breakdown: interior-only={only_int_fail}, boundary-only={only_bdry_fail}, both={both_div_fail}")
        print()
        # split per failure category
        for label, mask in (
            ("flagged_only", flagged_mask & ~logical_mask),
            ("logical_only", logical_mask & ~flagged_mask),
            ("both",         flagged_mask & logical_mask),
        ):
            count = int(mask.sum())
            if count == 0:
                continue
            sel_c = int((mask & all_in_pool).sum())
            div_c = int((mask & ~all_in_pool).sum())
            print(f"  [{label}] total={count}, selection_fail={sel_c}, diversity_fail={div_c}")
            if div_c > 0:
                for a_idx in interior_idx:
                    h = closest[mask & ~all_in_pool, a_idx]
                    in_pool_subset = in_pool[mask & ~all_in_pool, a_idx]
                    missing_h = h[~in_pool_subset]
                    if missing_h.size > 0:
                        counts = np.bincount(missing_h, minlength=4)[:6]
                        print(f"    A_{a_idx} hamming-dist-of-closest (only when oracle ∉ pool): {counts.tolist()} (idx=hamming)")
        print()
        # per-A: how often picked == oracle offset
        for a_idx in interior_idx:
            matches = int(((picked_off[:, a_idx] == oracle_off[:, a_idx]) & in_pool[:, a_idx]).sum())
            pool_hit = int(in_pool[:, a_idx].sum())
            print(f"A_{a_idx}: oracle ∈ pool in {pool_hit}/{shots_n} shots, "
                  f"and TSAE picked oracle in {matches}/{pool_hit} of those")


if __name__ == "__main__":
    main()
