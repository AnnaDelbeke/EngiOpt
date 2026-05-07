"""Build a one-row-per-task manifest from a WandB cgan_cnn_2d sweep.

Lists every successfully-finished run in the given sweep(s) and writes a CSV
with columns ``problem,run_id,run_name`` (one row per evaluation task). The
slurm sweep evaluator reads this file by line number.

Examples:
--------
::

    python -m engiopt.build_sweep_manifest \
        --sweep-id  beams2d=abc123 heatconduction2d=def456 \
        --output    sweep_manifest.csv

If you don't know the sweep IDs, list them with::

    wandb sweep ls --entity engibench --project engiopt
"""

from __future__ import annotations

import argparse
import csv
import sys

import wandb


def main() -> None:
    """Build a sweep manifest CSV for the slurm sweep evaluator."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="engibench")
    p.add_argument("--project", default="engiopt")
    p.add_argument(
        "--sweep-id",
        nargs="+",
        required=True,
        metavar="PROBLEM=SWEEP_ID",
        help="One or more PROBLEM=SWEEP_ID pairs, e.g. beams2d=abc123 heatconduction2d=def456.",
    )
    p.add_argument(
        "--include-states",
        nargs="+",
        default=["finished"],
        help="Only include runs with these states (default: finished).",
    )
    p.add_argument("--output", default="sweep_manifest.csv")
    args = p.parse_args()

    api = wandb.Api()  # type: ignore[attr-defined]

    rows: list[tuple[str, str, str]] = []
    for spec in args.sweep_id:
        if "=" not in spec:
            sys.exit(f"--sweep-id expects PROBLEM=SWEEP_ID, got {spec!r}")
        problem, sweep_id = spec.split("=", 1)
        sweep = api.sweep(f"{args.entity}/{args.project}/{sweep_id}")
        kept = 0
        for run in sweep.runs:
            if run.state not in args.include_states:
                continue
            rows.append((problem, run.id, run.name or ""))
            kept += 1
        print(f"  {problem}  sweep={sweep_id}  kept {kept}/{len(sweep.runs)} runs (states={args.include_states})")

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "run_id", "run_name"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")
    print(f"Set --array=0-{len(rows) - 1} on the slurm script.")


if __name__ == "__main__":
    main()
