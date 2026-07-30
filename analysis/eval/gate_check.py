"""Non-fatal per-step failure-rate gate for analysis/run_all_offline.sh.

Reads a driver's --manifest JSON (run_eval_matrix.py / run_eval_tau_sweep.py's
own {job_key: {"status": ..., ...}} bookkeeping - see both scripts' main())
and:
  - always prints every failed job's key + error (logged, never silently
    swallowed)
  - additionally flags any "success"-status job whose simulation_log.csv is
    header-only (1 line) - a silent-corruption anomaly that would not
    otherwise show up as a manifest failure
  - exits 1 (the calling step should abort) only if failed/total exceeds
    --max_fail_rate (default 0.20); otherwise exits 0 so the pipeline
    continues even though some individual cells failed.

Failed cells are NOT cleared or modified here - they remain in the manifest
with status="failed", so simply re-running that step's exact same driver
command later (as analysis/run_all_offline.sh does on every invocation) will
retry them automatically and leave every already-"success" cell alone (see
run_eval_matrix.py/run_eval_tau_sweep.py's "already recorded as success" skip
check). This script only ever reads the manifest.

No isaaclab/torch import - stdlib json + pathlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _is_header_only(csv_path: Path) -> bool:
    """True if csv_path has a header row but no data row (cheap: reads at most 2 lines)."""
    with open(csv_path) as f:
        next(f, None)  # header
        return next(f, None) is None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to the step's manifest JSON.")
    parser.add_argument("--eval_logs_root", required=True, help="Path to the step's eval_logs output root.")
    parser.add_argument("--max_fail_rate", type=float, default=0.20,
                         help="Abort (exit 1) if failed/total exceeds this fraction. Default 0.20.")
    parser.add_argument("--label", required=True, help="Short label for this step, used in log lines.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[gate:{args.label}] FAILED: manifest {manifest_path} does not exist")
        sys.exit(1)

    manifest: dict = json.loads(manifest_path.read_text())
    total = len(manifest)
    if total == 0:
        print(f"[gate:{args.label}] WARNING: manifest {manifest_path} has 0 job(s) recorded - nothing to gate on")
        sys.exit(0)

    failed = {k: v for k, v in manifest.items() if v.get("status") == "failed"}
    fail_rate = len(failed) / total

    print(f"[gate:{args.label}] {len(failed)}/{total} job(s) failed ({fail_rate:.1%})")
    for key, info in sorted(failed.items()):
        print(f"  [FAILED] {key}: {info.get('error', '<no error recorded>')}")

    root = Path(args.eval_logs_root)
    if root.is_dir():
        anomalies = [p for p in root.glob("*/*/*/simulation_log.csv") if _is_header_only(p)]
        if anomalies:
            print(f"[gate:{args.label}] WARNING: {len(anomalies)} header-only simulation_log.csv file(s) "
                  "found (silent-corruption check; not necessarily counted in the failure rate above):")
            for p in anomalies:
                print(f"  [HEADER-ONLY] {p}")

    if fail_rate > args.max_fail_rate:
        print(f"[gate:{args.label}] ABORTING: failure rate {fail_rate:.1%} exceeds --max_fail_rate "
              f"{args.max_fail_rate:.1%}. Failed cells stay in {manifest_path} with status=\"failed\" - "
              "re-run this step's command later to retry just those.")
        sys.exit(1)

    print(f"[gate:{args.label}] OK: failure rate {fail_rate:.1%} <= --max_fail_rate {args.max_fail_rate:.1%}, "
          "continuing to the next step.")
    sys.exit(0)


if __name__ == "__main__":
    main()
