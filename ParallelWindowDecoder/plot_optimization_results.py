import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

INPUTS = [
    RESULTS / "N144_repeat12_shots1000_global_dense_p1e-4_to_0.005.csv",
    RESULTS / "N144_repeat12_shots1000_sliding_vs_parallel_a_solve9_lowp_process_workers4.csv",
    RESULTS / "N144_repeat12_shots1000_sliding_vs_parallel_a_solve9_midp_process_workers4.csv",
    RESULTS / "N144_repeat12_shots1000_sliding_vs_parallel_a_solve9_process_workers4.csv",
]
COMBINED = RESULTS / "N144_repeat12_shots1000_global_sliding_parallel_a_solve9_dense_p1e-4_to_0.005.csv"


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                row["p"] = float(row["p"])
                row["num_shots"] = int(row["num_shots"])
                row["elapsed_sec"] = float(row["elapsed_sec"])
                row["flagged"] = int(row["flagged"])
                row["logical_or_flagged"] = int(row["logical_or_flagged"])
                row["ler"] = float(row["ler"])
                rows.append(row)
    return sorted(rows, key=lambda row: (row["p"], row["decoder"]))


def write_combined(rows):
    with COMBINED.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def by_decoder(rows, decoder):
    return sorted([row for row in rows if row["decoder"] == decoder], key=lambda row: row["p"])


def plot_ler(rows):
    plt.figure(figsize=(7.2, 4.4))
    for decoder, marker in [("global", "^"), ("sliding", "o"), ("parallel", "s")]:
        subset = by_decoder(rows, decoder)
        plt.plot(
            [row["p"] for row in subset],
            [row["ler"] for row in subset],
            marker=marker,
            linewidth=2,
            label=decoder,
        )
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-3)
    plt.xlabel("physical error rate p")
    plt.ylabel("logical-or-flagged error rate")
    plt.title("LER vs p, N=144, repeat=12, shots=1000")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = FIGURES / "N144_a_solve9_ler_vs_p.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_runtime(rows):
    plt.figure(figsize=(7.2, 4.4))
    for decoder, marker in [("global", "^"), ("sliding", "o"), ("parallel", "s")]:
        subset = by_decoder(rows, decoder)
        plt.plot(
            [row["p"] for row in subset],
            [row["elapsed_sec"] for row in subset],
            marker=marker,
            linewidth=2,
            label=decoder,
        )
    plt.xscale("log")
    plt.xlabel("physical error rate p")
    plt.ylabel("elapsed seconds")
    plt.title("Runtime vs p, N=144, repeat=12, shots=1000")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = FIGURES / "N144_a_solve9_runtime_vs_p.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def plot_speedup(rows):
    global_rows = {row["p"]: row for row in by_decoder(rows, "global")}
    sliding = {row["p"]: row for row in by_decoder(rows, "sliding")}
    parallel = {row["p"]: row for row in by_decoder(rows, "parallel")}
    ps = sorted(set(sliding) & set(parallel))
    sliding_speedups = [sliding[p]["elapsed_sec"] / parallel[p]["elapsed_sec"] for p in ps]

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(ps, sliding_speedups, marker="D", linewidth=2, color="#2f6f4e", label="sliding / parallel")
    ps_global = sorted(set(global_rows) & set(parallel))
    global_speedups = [global_rows[p]["elapsed_sec"] / parallel[p]["elapsed_sec"] for p in ps_global]
    plt.plot(ps_global, global_speedups, marker="^", linewidth=2, color="#8f3f2f", label="global / parallel")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.xlabel("physical error rate p")
    plt.ylabel("sliding time / parallel time")
    plt.title("Parallel Speedup vs p")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = FIGURES / "N144_a_solve9_speedup_vs_p.png"
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def main():
    rows = load_rows(INPUTS)
    write_combined(rows)
    outputs = [plot_ler(rows), plot_runtime(rows), plot_speedup(rows)]
    print(f"wrote combined csv: {COMBINED}")
    for output in outputs:
        print(f"wrote figure: {output}")


if __name__ == "__main__":
    main()
