import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from ParallelWindowDecoder.src.bp_osd import (  # noqa: E402
    choose_boundary_feedback_flips,
    decode_a_candidate_payload,
    decode_a_repair_payload,
    decode_window_payload,
    error_cost,
    gf2_osd_list_candidates,
    make_bp_osd,
    run_a_candidate_payloads,
    run_a_repair_payloads,
    run_window_payloads,
    scale_priors_by_llr,
    unique_concat,
)
from ParallelWindowDecoder.src.decoders_global import decode_global, decode_sliding  # noqa: E402
from ParallelWindowDecoder.src.noisy_boundary import (  # noqa: E402
    add_noisy_boundary_columns,
    estimate_noisy_boundary_prior,
)
from ParallelWindowDecoder.src.problem import (  # noqa: E402
    PreparedProblem,
    error_key,
    make_code,
    prepare_problem,
    raw_error_to_collapsed_cols,
    region_col_slice,
    row_slice,
    sample_dem,
    score,
)


from ParallelWindowDecoder.src.parallel_ab import decode_parallel_ab, decode_staggered_ab  # noqa: E402


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
    parser.add_argument("--tsae-interface-branch", action="store_true")
    parser.add_argument("--tsae-interface-cols-per-side", type=int, default=1)
    parser.add_argument("--tsae-interface-joint-score", action="store_true",
                        help="Pick interface branch b* by summing local costs across all shifted sub-windows, plus optional B-residual penalty.")
    parser.add_argument("--tsae-interface-zb-weight", type=float, default=1.0,
                        help="Weight on B-residual popcount in joint-factor score (only with --tsae-interface-joint-score).")
    parser.add_argument("--tsae-interface-commit-offset", type=int, default=0,
                        help="Offset to commit from after picking joint b*. Default 0 (W_C).")
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
        tsae_interface_branch=args.tsae_interface_branch,
        tsae_interface_cols_per_side=args.tsae_interface_cols_per_side,
        tsae_interface_joint_score=args.tsae_interface_joint_score,
        tsae_interface_zb_weight=args.tsae_interface_zb_weight,
        tsae_interface_commit_offset=args.tsae_interface_commit_offset,
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
