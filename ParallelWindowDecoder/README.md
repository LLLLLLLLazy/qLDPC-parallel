# Parallel Window Decoder Experiments

This directory contains a small experimental harness for the qLDPC A/B parallel
window idea described in `../qLDPC_parallel_window_decoding.md`.

It intentionally reuses the circuit construction, DEM-to-matrix conversion, and
code families from `../SlidingWindowDecoder`.  The new part is the schedule:

1. Build the same circuit-level matrix used by the sliding-window notebooks.
2. Reorder fault columns into time regions:
   `H0(0), bridge(0,1), H0(1), bridge(1,2), ...`.
3. Decode all selected A boundary windows independently and commit only their
   bridge regions.
4. Apply the residual update.
5. Decode all B segments independently on the residual syndrome.
6. Check the final global detector equation and logical observables.

## Quick Smoke Test

From the repository root:

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/parallel_window_decoder.py \
  --N 72 --p 0.003 --num-repeat 4 --num-shots 20 --b-width 2 --a-radius 1
```

For the first validation pass, run the oracle-boundary diagnostic.  This uses
Stim's sampled faults only for the A-layer bridge variables, then tests whether
the B-layer residual equations close:

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/parallel_window_decoder.py \
  --N 72 --p 0.003 --num-repeat 4 --num-shots 20 --b-width 2 --a-radius 1 \
  --osd-order 0 --oracle-boundaries --run-global --run-sliding
```

On the checked smoke test this gives `flagged: 0/20` for the parallel A/B
decoder, which verifies the B-window matrix and residual update.

Then test decoded A boundaries with a wider A context:

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/parallel_window_decoder.py \
  --N 72 --p 0.003 --num-repeat 4 --num-shots 20 --b-width 2 --a-radius 2 \
  --osd-order 10 --max-iter 200 --run-global
```

On the same small setting this also closes with `flagged: 0/20`.  With
`--a-radius 1`, the A windows are usually too weak and the final residual often
stays nonzero; that is useful as a sanity check that boundary quality is the
main experimental knob.

Useful options:

- `--run-global`: include a full-matrix BP+OSD baseline.
- `--run-sliding`: include a sequential sliding-window baseline.
- `--oracle-boundaries`: use true sampled bridge faults for the A layer. This is
  a diagnostic for the B-layer equations, not a decoder.
- `--a-size`: set `n_A` directly and use the staggered A/B schedule. For
  example, `--b-width 3 --a-size 3` gives
  `A1:s1-s2`, `B1:s2-s4`, `A2:s4-s6`, `B2:s6-s8`, ...
- `--a-solve-size`: make each A window solve a larger buffered detector range
  while keeping the same A commit size. For example, `--a-size 3
  --a-solve-size 5` makes an interior A solve `s3-s7` but still commit only the
  three central boundary regions such as `e7,e8,e9`.
- `--a-radius`: legacy centered A-window control used when `--a-size` is not
  provided.
- `--osd-order`: OSD order for local windows. The default is `0` so that quick
  validation finishes fast.
- `--seed`: fix Stim/NumPy sampling for reproducibility.
- `--a-noisy-boundary`: add virtual noisy boundary columns to A windows before
  decoding. This is the first pass at the same kind of boundary absorption used
  in the sliding-window OSD baseline.
- `--parallel-workers`: run A-layer windows in parallel, then B-layer windows in
  parallel. `1` keeps the serial reference path.
- `--parallel-backend`: choose `thread` or `process`. The `process` backend is
  true multi-process parallelism and usually gives the useful speedup, but may
  require running outside the sandbox because Python's process pool uses system
  semaphore/process APIs.

Backups of the pre-parallel implementation are in:

```text
ParallelWindowDecoder/backups/
```

## Sweep Experiments

Use `run_experiments.py` when you want SlidingWindowDecoder-style repeated
experiments and CSV output:

```bash
SlidingWindowDecoder/.conda-gdg/bin/python ParallelWindowDecoder/run_experiments.py \
  --N 72 --p-list 0.002,0.003,0.004 \
  --num-repeat 10 --num-shots 100 \
  --a-size 3 --b-width 3 \
  --osd-order 10 --max-iter 200 \
  --parallel-workers 4 --parallel-backend process \
  --decoders global,sliding,parallel,oracle \
  --out ParallelWindowDecoder/results/ab_window_results.csv
```

The generated CSV includes `LER`, flagged counts, elapsed time, detector block
count, and JSON diagnostics such as actual `A` and `B` row ranges.

For notebook analysis, open:

```text
ParallelWindowDecoder/Parallel Window AB Analysis.ipynb
```

The notebook can run the sweep, load the CSV, print summary tables, plot LER and
runtime, and inspect the staggered window layout.
