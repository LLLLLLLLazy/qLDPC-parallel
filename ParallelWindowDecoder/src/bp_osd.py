from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools

from . import bootstrap as _bootstrap  # noqa: F401

import numpy as np
from ldpc import BpOsdDecoder

try:
    from src import osd_window as ShortenedOsdWindow  # noqa: E402
except ImportError:
    ShortenedOsdWindow = None

try:
    from osd_list import osd_list_candidates as cython_osd_list_candidates
except ImportError:
    cython_osd_list_candidates = None


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
    fixed_positions = np.asarray(payload.get("fixed_value_positions", []), dtype=np.int64)
    fixed_values = np.asarray(payload.get("fixed_value_bits", []), dtype=np.uint8)

    decode_mat = mat
    decode_prior = prior
    decode_syndromes = syndromes
    keep_positions = None
    if fixed_positions.size:
        if fixed_positions.size != fixed_values.size:
            raise ValueError("fixed_value_positions and fixed_value_bits must have the same length")
        if np.any(fixed_positions < 0) or np.any(fixed_positions >= mat.shape[1]):
            raise ValueError("fixed_value_positions contains an out-of-range column")
        fixed_contribution = (fixed_values @ mat[:, fixed_positions].T) % 2
        decode_syndromes = (syndromes + fixed_contribution[None, :]) % 2
        fixed_mask = np.zeros(mat.shape[1], dtype=bool)
        fixed_mask[fixed_positions] = True
        keep_positions = np.flatnonzero(~fixed_mask)
        decode_mat = mat[:, keep_positions]
        decode_prior = prior[keep_positions]

    decoder = make_bp_osd(decode_mat, decode_prior, max_iter, osd_order, shorten, shorten_pre_max_iter)
    shots = syndromes.shape[0]
    width = value_stop - value_start
    values = np.zeros((shots, top_k, width), dtype=np.uint8)
    costs = np.full((shots, top_k), np.inf, dtype=float)
    flagged = 0

    for shot in range(shots):
        if hasattr(decoder, "decode_list"):
            candidates = decoder.decode_list(
                decode_syndromes[shot],
                top_k=top_k,
                force_osd=True,
            ).astype(np.uint8)
            local_e_decoded = candidates[0]
        else:
            local_e_decoded = decoder.decode(decode_syndromes[shot]).astype(np.uint8)
            reliabilities = np.abs(np.asarray(decoder.log_prob_ratios, dtype=float))
            if cython_osd_list_candidates is not None:
                candidates = cython_osd_list_candidates(
                    np.ascontiguousarray(decode_mat, dtype=np.uint8),
                    np.ascontiguousarray(decode_syndromes[shot], dtype=np.uint8),
                    np.ascontiguousarray(decode_prior, dtype=np.float64),
                    np.ascontiguousarray(local_e_decoded, dtype=np.uint8),
                    np.ascontiguousarray(reliabilities, dtype=np.float64),
                    top_k,
                )
            else:
                candidates = gf2_osd_list_candidates(
                    decode_mat,
                    decode_syndromes[shot],
                    decode_prior,
                    local_e_decoded,
                    reliabilities,
                    top_k,
                )

        if keep_positions is None:
            local_e = local_e_decoded
        else:
            local_e = np.zeros(mat.shape[1], dtype=np.uint8)
            local_e[fixed_positions] = fixed_values
            local_e[keep_positions] = local_e_decoded

        residual = (mat @ local_e + syndromes[shot]) % 2
        residual_weight = int(residual.sum())
        if residual_weight:
            flagged += 1

        for cand, candidate_full in enumerate(candidates[:top_k]):
            if keep_positions is not None:
                expanded = np.zeros(mat.shape[1], dtype=np.uint8)
                expanded[fixed_positions] = fixed_values
                expanded[keep_positions] = candidate_full
                candidate_full = expanded
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
