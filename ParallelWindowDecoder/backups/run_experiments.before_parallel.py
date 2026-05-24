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
            )
            diagnostics = {"sliding_width": args.sliding_width}
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
