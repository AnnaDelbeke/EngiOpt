"""Delete only LV-metric summary keys from WandB runs.

Targets summary keys matching ``^eval/lv_rec`` on every run whose name
matches ``{problem_id}__{model}__*`` for the given problems and models.
Base metrics (IOG/COG/FOG/MMD/DPP) under ``eval/<key>`` are left untouched.

Run with ``--dry-run`` first to print what would be deleted.

Examples:
--------
Dry run::

    python -m engiopt.cleanup_lv_metrics \
        --problems beams2d heatconduction2d \
        --models cgan_cnn_2d gan_cnn_2d diffusion_2d_cond vqgan \
        --dry-run

Apply::

    python -m engiopt.cleanup_lv_metrics \
        --problems beams2d heatconduction2d \
        --models cgan_cnn_2d gan_cnn_2d diffusion_2d_cond vqgan
"""

from __future__ import annotations

import argparse
import re

import wandb

LV_KEY_RE = re.compile(r"^eval/lv_rec")
KNOWN_MODELS = ["cgan_cnn_2d", "gan_cnn_2d", "diffusion_2d_cond", "vqgan"]


_PREVIEW_KEYS_PER_RUN = 5


def main() -> None:  # noqa: C901, PLR0912
    """Delete LV-metric summary keys from WandB runs (or report them in --dry-run)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="engibench")
    p.add_argument("--project", default="engiopt")
    p.add_argument("--problems", nargs="+", required=True, help="Problem IDs to clean (e.g. beams2d heatconduction2d).")
    p.add_argument(
        "--models", nargs="+", default=KNOWN_MODELS, choices=KNOWN_MODELS, help="Model families to clean (run-name match)."
    )
    p.add_argument("--dry-run", action="store_true", help="Report what would be deleted without modifying any runs.")
    args = p.parse_args()

    api = wandb.Api()  # type: ignore[attr-defined]

    # Match run names of the form `{problem}__{model}__...`
    name_re = re.compile(
        r"^(?:" + "|".join(re.escape(p) for p in args.problems) + r")__"
        r"(?:" + "|".join(re.escape(m) for m in args.models) + r")__"
    )

    print(f"Querying runs in {args.entity}/{args.project} for problems={args.problems}...")
    matched = []
    for problem in args.problems:
        runs = api.runs(
            f"{args.entity}/{args.project}",
            filters={"config.problem_id": problem},
            per_page=200,
        )
        matched.extend(run for run in runs if name_re.match(run.name or ""))
    print(f"Matched {len(matched)} runs by name pattern.")

    by_run: dict[str, tuple] = {}
    for run in matched:
        keys = [k for k in run.summary if LV_KEY_RE.match(k)]
        if keys:
            by_run[run.id] = (run, keys)

    total_keys = sum(len(k) for _, k in by_run.values())
    print(f"\n{len(by_run)} runs hold LV-metric summary keys ({total_keys} keys total).")

    # Sample preview so the user can sanity-check the prefix before applying.
    for run_id, (run, keys) in list(by_run.items())[:3]:
        print(f"  {run.name}  ({run_id})  -- {len(keys)} keys")
        for k in keys[:_PREVIEW_KEYS_PER_RUN]:
            print(f"      {k}")
        if len(keys) > _PREVIEW_KEYS_PER_RUN:
            print(f"      ... +{len(keys) - _PREVIEW_KEYS_PER_RUN} more")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return

    if not by_run:
        print("Nothing to do.")
        return

    print(f"\nDeleting {total_keys} LV-metric summary keys across {len(by_run)} runs...")
    failures: dict[str, str] = {}
    for run_id, (run, keys) in by_run.items():
        try:
            for k in keys:
                del run.summary[k]
            run.summary.update()
            print(f"  cleaned {run.name}  ({run_id})  -- {len(keys)} keys")
        except Exception as e:  # noqa: BLE001, PERF203
            failures[run_id] = str(e)
            print(f"  FAILED  {run.name}  ({run_id}): {e}")

    if failures:
        print(f"\n{len(failures)} runs failed:")
        for rid, msg in failures.items():
            print(f"  {rid}: {msg}")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
