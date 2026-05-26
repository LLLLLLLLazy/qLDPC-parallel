from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from . import bootstrap as _bootstrap  # noqa: F401
from src.build_circuit import build_circuit, dem_to_check_matrices  # noqa: E402
from src.codes_q import create_bivariate_bicycle_codes  # noqa: E402


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

