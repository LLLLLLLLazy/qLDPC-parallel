import argparse
import csv
import json
import time
from pathlib import Path

from parallel_window_decoder import (
    decode_global,
    decode_parallel_ab,
    decode_sliding,
    prepare_problem,
    sample_dem,
    score,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_decoder_list(value: str) -> list[str]:
    allowed = {"global", "sliding", "parallel", "oracle"}
    decoders = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(decoders) - allowed)
    if unknown:
        raise ValueError(f"unknown decoders: {unknown}; allowed={sorted(allowed)}")
    return decoders


def run_one_config(args, p: float, seed: int) -> list[dict]:
    problem = prepare_problem(args.N, p, args.num_repeat, z_basis=not args.x_basis)
    det_data, obs_data, err_data = sample_dem(problem, args.num_shots, seed)
    rows = []

    for decoder_name in parse_decoder_list(args.decoders):
        t0 = time.perf_counter()
        diagnostics = {}
        if decoder_name == "global":
            e_hat = decode_global(problem, det_data, args.max_iter, args.osd_order)
        elif decoder_name == "sliding":
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
            diagnostics = {
                "sliding_width": args.sliding_width,
                "sliding_method": args.sliding_method,
                "window_shorten": args.window_shorten,
                "shorten_pre_max_iter": args.shorten_pre_max_iter,
            }
        elif decoder_name == "parallel":
            e_hat, diagnostics = decode_parallel_ab(
                problem,
                det_data,
                args.b_width,
                args.a_radius,
                args.a_size,
                args.a_solve_size,
                args.max_iter,
                args.osd_order,
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
                a_shift_offsets=parse_int_tuple(args.a_shift_offsets),
                a_shifted_beta=args.a_shifted_beta,
                tsae_top_k=args.tsae_top_k,
                tsae_boundary_top_k=args.tsae_boundary_top_k,
                tsae_stitch_mode=args.tsae_stitch_mode,
                z_boundary_repair=args.z_boundary_repair,
                z_repair_edge_width=args.z_repair_edge_width,
                z_joint_retry=args.z_joint_retry,
                seam_diagnostics=args.seam_diagnostics,
            )
        elif decoder_name == "oracle":
            e_hat, diagnostics = decode_parallel_ab(
                problem,
                det_data,
                args.b_width,
                args.a_radius,
                args.a_size,
                args.a_solve_size,
                args.max_iter,
                args.osd_order,
                oracle_errors=err_data,
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
                flag_triggered_single_flip=False,
                bounded_two_flip=False,
                two_flip_candidates=args.two_flip_candidates,
                a_boundary_weight_scale=args.a_boundary_weight_scale,
                aba_boundary_feedback=False,
                aba_max_flips=args.aba_max_flips,
                aba_candidate_cols=args.aba_candidate_cols,
                a_neighbor_rerank=False,
                a_rerank_top_k=args.a_rerank_top_k,
                a_neighbor_beta=args.a_neighbor_beta,
                a_two_color_order="none",
                a_shifted_ensemble=False,
                a_shift_offsets=parse_int_tuple(args.a_shift_offsets),
                a_shifted_beta=args.a_shifted_beta,
                tsae_top_k=1,
                tsae_boundary_top_k=1,
                tsae_stitch_mode="none",
                z_boundary_repair=False,
                z_repair_edge_width=args.z_repair_edge_width,
                z_joint_retry=False,
                seam_diagnostics=args.seam_diagnostics,
            )
        else:
            raise AssertionError(decoder_name)

        elapsed = time.perf_counter() - t0
        metrics = score(problem, det_data, obs_data, e_hat)
        rows.append(
            {
                "decoder": decoder_name,
                "N": args.N,
                "p": p,
                "num_repeat": args.num_repeat,
                "num_shots": args.num_shots,
                "seed": seed,
                "max_iter": args.max_iter,
                "osd_order": args.osd_order,
                "n_a": args.a_size,
                "n_a_solve": args.a_solve_size or args.a_size,
                "n_b": args.b_width,
                "a_noisy_boundary": args.a_noisy_boundary,
                "b_noisy_boundary": args.b_noisy_boundary,
                "window_shorten": args.window_shorten,
                "shorten_pre_max_iter": args.shorten_pre_max_iter,
                "sliding_method": args.sliding_method,
                "flag_triggered_single_flip": args.flag_triggered_single_flip,
                "bounded_two_flip": args.bounded_two_flip,
                "two_flip_candidates": args.two_flip_candidates,
                "a_boundary_weight_scale": args.a_boundary_weight_scale,
                "aba_boundary_feedback": args.aba_boundary_feedback,
                "aba_max_flips": args.aba_max_flips,
                "aba_candidate_cols": args.aba_candidate_cols,
                "a_neighbor_rerank": args.a_neighbor_rerank,
                "a_rerank_top_k": args.a_rerank_top_k,
                "a_neighbor_beta": args.a_neighbor_beta,
                "a_two_color_order": args.a_two_color_order,
                "a_shifted_ensemble": args.a_shifted_ensemble,
                "a_shift_offsets": args.a_shift_offsets,
                "a_shifted_beta": args.a_shifted_beta,
                "tsae_top_k": args.tsae_top_k,
                "tsae_boundary_top_k": args.tsae_boundary_top_k,
                "tsae_stitch_mode": args.tsae_stitch_mode,
                "z_boundary_repair": args.z_boundary_repair,
                "z_repair_edge_width": args.z_repair_edge_width,
                "z_joint_retry": args.z_joint_retry,
                "seam_diagnostics": args.seam_diagnostics,
                "parallel_workers": args.parallel_workers,
                "parallel_backend": args.parallel_backend,
                "top_k_boundary": args.top_k_boundary,
                "soft_boundary_message": args.soft_boundary_message,
                "soft_boundary_one_prior": args.soft_boundary_one_prior,
                "soft_boundary_zero_prior": args.soft_boundary_zero_prior,
                "elapsed_sec": elapsed,
                "flagged": metrics["flagged"],
                "logical_or_flagged": metrics["logical_or_flagged"],
                "ler": metrics["ler"],
                "detector_blocks": problem.num_detector_blocks,
                "num_cols": problem.chk.shape[1],
                "diagnostics": json.dumps(diagnostics, ensure_ascii=False),
            }
        )

    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "decoder",
        "N",
        "p",
        "num_repeat",
        "num_shots",
        "seed",
        "max_iter",
        "osd_order",
        "n_a",
        "n_a_solve",
        "n_b",
        "a_noisy_boundary",
        "b_noisy_boundary",
        "window_shorten",
        "shorten_pre_max_iter",
        "sliding_method",
        "flag_triggered_single_flip",
        "bounded_two_flip",
        "two_flip_candidates",
        "a_boundary_weight_scale",
        "aba_boundary_feedback",
        "aba_max_flips",
        "aba_candidate_cols",
        "a_neighbor_rerank",
        "a_rerank_top_k",
        "a_neighbor_beta",
        "a_two_color_order",
        "a_shifted_ensemble",
        "a_shift_offsets",
        "a_shifted_beta",
        "tsae_top_k",
        "tsae_boundary_top_k",
        "tsae_stitch_mode",
        "z_boundary_repair",
        "z_repair_edge_width",
        "z_joint_retry",
        "seam_diagnostics",
        "parallel_workers",
        "parallel_backend",
        "top_k_boundary",
        "soft_boundary_message",
        "soft_boundary_one_prior",
        "soft_boundary_zero_prior",
        "elapsed_sec",
        "flagged",
        "logical_or_flagged",
        "ler",
        "detector_blocks",
        "num_cols",
        "diagnostics",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=72)
    parser.add_argument("--p-list", default="0.002,0.003,0.004")
    parser.add_argument("--num-repeat", type=int, default=10)
    parser.add_argument("--num-shots", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--osd-order", type=int, default=10)
    parser.add_argument("--sliding-width", type=int, default=3)
    parser.add_argument("--b-width", type=int, default=3)
    parser.add_argument("--a-radius", type=int, default=1)
    parser.add_argument("--a-size", type=int, default=3)
    parser.add_argument("--a-solve-size", type=int, default=None)
    parser.add_argument("--x-basis", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoders", default="global,sliding,parallel,oracle")
    parser.add_argument("--out", default="ParallelWindowDecoder/results/ab_window_results.csv")
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

    all_rows = []
    for index, p in enumerate(parse_float_list(args.p_list)):
        seed = args.seed + index
        print(f"running N={args.N} p={p} shots={args.num_shots} seed={seed}")
        rows = run_one_config(args, p, seed)
        for row in rows:
            print(
                row["decoder"],
                f"LER={row['ler']:.6g}",
                f"flagged={row['flagged']}/{row['num_shots']}",
                f"elapsed={row['elapsed_sec']:.3f}s",
            )
        all_rows.extend(rows)

    out = Path(args.out)
    write_rows(out, all_rows)
    print(f"wrote {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
