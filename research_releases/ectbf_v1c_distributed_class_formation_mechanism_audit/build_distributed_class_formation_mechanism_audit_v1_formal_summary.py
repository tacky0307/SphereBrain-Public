from __future__ import annotations

import argparse
import json
from pathlib import Path

import distributed_class_formation_mechanism_audit_config_v1 as cfg
import distributed_class_formation_mechanism_audit_v1 as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    decision = result["decision"]
    aggregate = result["aggregate"]
    summary = {
        "study": cfg.STUDY,
        "formal_status": decision["status"],
        "candidate_promoted": decision["candidate_promoted"],
        "case_count": result["case_count"],
        "compact_audit_all_pass": audit["all_pass"],
        "formal_result_sha256": core.sha256_path(args.result),
        "formal_audit_sha256": core.sha256_path(args.audit),
        "frozen_source_manifest_sha256": manifest["source_manifest_sha256"],
        "threshold_declaration_sha256": manifest["threshold_declaration_sha256"],
        "primary_metrics": {
            "full_relation_accuracy": aggregate["methods"][cfg.FULL]["relation_accuracy"],
            "cooperative_relation_accuracy": aggregate["methods"][cfg.COOPERATIVE]["relation_accuracy"],
            "marginal_relation_accuracy": aggregate["methods"][cfg.MARGINAL]["relation_accuracy"],
            "cooperative_complete_sequence": aggregate["methods"][cfg.COOPERATIVE]["complete_sequence_identity_rate"],
            "combined_complete_sequence": aggregate["methods"][cfg.COOPERATIVE_COMBINED]["complete_sequence_identity_rate"],
        },
        "paired_mechanism_evidence": aggregate["paired"],
        "mechanism_metrics": aggregate["mechanism"],
        "causal_metrics": aggregate["causal"],
        "formal_decision": decision,
        "ancestry": result["ancestry"],
        "supported_conclusion": (
            "The frozen result determines whether preserving joint cross-member covariance "
            "improves unseen consequence decisions and finite plan continuation beyond "
            "cardinality-matched marginal members while individual conditional marginals "
            "remain fixed."
        ),
        "unsupported_conclusion": (
            "This does not establish autonomous event invention, removal of the ordered-"
            "moment vocabulary, reward discovery, unrestricted planning, natural-language "
            "reasoning, consciousness, or real-world agency."
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
