from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import distributed_class_formation_mechanism_audit_config_v1 as cfg
import distributed_class_formation_mechanism_audit_v1 as core
import distributed_temporal_boundary_equivalence_config_v1 as v1b_cfg


def audit(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_seeds = (
        list(cfg.DEV_FAMILY_SEEDS)
        if result["phase"] == "development"
        else list(cfg.HOLDOUT_FAMILY_SEEDS)
    )
    expected_cases = len(expected_seeds) * cfg.QUERY_COUNT * len(v1b_cfg.SUITES)
    manifest_sha, members = core.source_manifest()
    case_sha_exact = all(
        case["case_sha256"]
        == core.payload_hash({key: value for key, value in case.items() if key != "case_sha256"})
        for case in result["cases"]
    )
    checks = {
        "study_exact": result["study"] == cfg.STUDY,
        "target_branch_exact": result["target_branch"] == cfg.TARGET_BRANCH,
        "phase_valid": result["phase"] in ("development", "holdout"),
        "family_seed_bank_exact": result["family_seeds"] == expected_seeds,
        "suite_bank_exact": result["suite_names"] == [suite["name"] for suite in v1b_cfg.SUITES],
        "query_count_exact": result["query_count"] == cfg.QUERY_COUNT,
        "case_count_exact": result["case_count"] == expected_cases,
        "case_list_count_exact": len(result["cases"]) == expected_cases,
        "source_manifest_exact": result["source_manifest_sha256"] == manifest_sha,
        "source_manifest_members_exact": result["source_manifest_members"] == members,
        "case_hashes_exact": case_sha_exact,
        "all_information_boundaries_pass": result["aggregate"]["audits"]["all_information_boundaries_pass"],
        "all_control_identity_pass": result["aggregate"]["audits"]["all_control_identity_pass"],
        "all_subset_sizes_equal": result["aggregate"]["mechanism"]["all_subset_sizes_equal"],
        "all_feedback_delayed": result["aggregate"]["mechanism"]["all_feedback_delayed"],
        "all_held_query_updates_zero": result["aggregate"]["mechanism"]["all_held_query_updates_zero"],
        "all_snapshot_roundtrips_exact": result["aggregate"]["mechanism"]["all_snapshot_roundtrips_exact"],
        "snapshot_forbidden_zero": result["aggregate"]["mechanism"]["maximum_snapshot_forbidden_token_count"] == 0,
        "conditional_marginals_preserved": result["aggregate"]["mechanism"]["maximum_conditional_marginal_preservation_error"] <= 1e-10,
        "ancestry_pipeline_complete": result["ancestry"]["completion_receipt"]["pipeline_complete"] is True,
        "ancestry_result_hash_match": (
            result["ancestry"]["completion_receipt"]["formal_result_sha256"]
            == result["ancestry"]["formal_summary"]["formal_result_sha256"]
        ),
        "development_has_no_formal_decision": (
            result["phase"] != "development" or result["decision"] is None
        ),
        "development_thresholds_unfrozen": (
            result["phase"] != "development" or result["thresholds"] is None
        ),
        "holdout_has_formal_decision": (
            result["phase"] != "holdout" or result["decision"] is not None
        ),
        "holdout_thresholds_frozen": (
            result["phase"] != "holdout" or result["thresholds"] is not None
        ),
    }
    payload = {
        "study": cfg.STUDY,
        "phase": result["phase"],
        "case_count": result["case_count"],
        "artifact_sha256": core.sha256_path(result_path),
        "source_manifest_sha256": result["source_manifest_sha256"],
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    output = result_path.with_name(result_path.stem + "_compact_audit.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = audit(args.result)
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
