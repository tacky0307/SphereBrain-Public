from __future__ import annotations

import argparse
from pathlib import Path

import distributed_class_formation_mechanism_audit_config_v1 as cfg
from distributed_class_formation_mechanism_audit_v1 import run_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "development":
        if cfg.FROZEN_THRESHOLDS is not None:
            raise SystemExit("development source is already frozen")
        seeds = cfg.DEV_FAMILY_SEEDS
    else:
        if cfg.FROZEN_THRESHOLDS is None:
            raise SystemExit("formal holdout requires frozen thresholds")
        seeds = cfg.HOLDOUT_FAMILY_SEEDS
    result = run_bank(args.phase, seeds, args.output)
    print(
        {
            "phase": result["phase"],
            "case_count": result["case_count"],
            "decision": result["decision"],
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
