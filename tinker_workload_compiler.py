#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_contracts import (
    DEFAULT_SCENARIO_OUTPUT,
    REQUIRED_WORKLOAD_METADATA_FIELDS,
    compile_contracts_to_dataset,
    load_contracts,
    write_dataset,
)

DEFAULT_CONTRACT_DIR = "benchmark_contracts"
DEFAULT_OUTPUT_DIR = "workload_bundles"


@dataclass(slots=True)
class CLIConfig:
    contract_input: str
    output_dir: str
    scenario_output: str
    family_names: list[str]
    scenario_names: list[str]
    compile_only: bool
    list_families: bool
    list_scenarios: bool
    self_test: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_input": self.contract_input,
            "output_dir": self.output_dir,
            "scenario_output": self.scenario_output,
            "family_names": self.family_names,
            "scenario_names": self.scenario_names,
            "compile_only": self.compile_only,
            "list_families": self.list_families,
            "list_scenarios": self.list_scenarios,
            "self_test": self.self_test,
        }


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _select_contracts(
    contracts: list[dict[str, Any]],
    family_names: list[str],
    scenario_names: list[str],
) -> list[dict[str, Any]]:
    selected = contracts
    if family_names:
        requested = {item.strip() for item in family_names if item.strip()}
        selected = [contract for contract in selected if contract["benchmark_family"] in requested]
    if scenario_names:
        requested = {item.strip() for item in scenario_names if item.strip()}
        by_name = {contract["name"]: contract for contract in selected}
        missing = sorted(name for name in requested if name not in by_name)
        if missing:
            raise ValueError(f"Unknown scenario(s): {', '.join(missing)}")
        selected = [by_name[name] for name in scenario_names if name in by_name]
    if not selected:
        raise ValueError("No contracts matched the requested filters.")
    return sorted(selected, key=lambda item: item["name"])


def _metadata_with_compiler_fields(scenario: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(scenario.get("workload_metadata", {}))
    missing = [
        field
        for field in REQUIRED_WORKLOAD_METADATA_FIELDS
        if not str(metadata.get(field, "")).strip()
    ]
    if missing:
        raise ValueError(
            f"Scenario {scenario['name']} missing workload metadata fields: {', '.join(missing)}"
        )
    output_contract_type = metadata.get("output_contract_type", [])
    if not output_contract_type:
        raise ValueError(f"Scenario {scenario['name']} did not produce output_contract_type metadata.")
    metadata["output_contract_type"] = list(output_contract_type)
    metadata["benchmark_family"] = scenario.get("benchmark_family", "")
    metadata["turn_count"] = len(scenario.get("turns", []))
    return metadata


def _build_conversation(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if str(scenario.get("system_prompt", "")).strip():
        steps.append(
            {
                "step_id": "system_prompt",
                "phase": "system_prompt",
                "role": "system",
                "source": "contract",
                "masked": False,
                "content": scenario["system_prompt"],
            }
        )

    for index, turn in enumerate(scenario.get("turns", []), start=1):
        turn_id = str(turn.get("turn_id", f"turn_{index:02d}"))
        phase = f"turn_{index:02d}"
        steps.append(
            {
                "step_id": f"{turn_id}_user",
                "phase": phase,
                "role": "user",
                "source": "contract",
                "masked": False,
                "content": turn["user"],
                "prompt": turn.get("prompt", ""),
                "input_contract": turn.get("input_contract", {}),
            }
        )
        steps.append(
            {
                "step_id": f"{turn_id}_assistant",
                "phase": phase,
                "role": "assistant",
                "source": "model_fill",
                "masked": True,
                "content": None,
                "output_contract": turn.get("output_contract", {}),
            }
        )
    return steps


def _build_checks(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    compiled_checks: list[dict[str, Any]] = []
    for index, turn in enumerate(scenario.get("turns", []), start=1):
        turn_id = str(turn.get("turn_id", f"turn_{index:02d}"))
        phase = f"turn_{index:02d}"
        for check in turn.get("checks", []):
            params = {
                key: value
                for key, value in check.items()
                if key not in {"name", "rule", "weight", "description"}
            }
            payload = {
                "step_id": f"{turn_id}_assistant",
                "phase": phase,
                "name": str(check["name"]),
                "rule": str(check["rule"]),
                "weight": float(check.get("weight", 1.0)),
                "params": params,
            }
            description = str(check.get("description", "")).strip()
            if description:
                payload["description"] = description
            compiled_checks.append(payload)
    return compiled_checks


def _build_compiled_instances(
    contracts: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario_by_name = {scenario["name"]: scenario for scenario in scenarios}
    compiled_instances: list[dict[str, Any]] = []

    for contract in contracts:
        scenario = scenario_by_name[contract["name"]]
        authored_fingerprint = _stable_hash(contract)
        conversation = _build_conversation(scenario)
        checks = _build_checks(scenario)
        workload_metadata = _metadata_with_compiler_fields(scenario)
        payload_for_instance = {
            "contract": contract,
            "scenario": scenario,
            "conversation": conversation,
            "checks": checks,
            "workload_metadata": workload_metadata,
        }
        instance_fingerprint = _stable_hash(payload_for_instance)
        compiled_instances.append(
            {
                "instance_id": f"{scenario['name']}-{instance_fingerprint[:8]}",
                "name": scenario["name"],
                "benchmark_family": scenario.get("benchmark_family", ""),
                "spec_fingerprint": authored_fingerprint,
                "instance_fingerprint": instance_fingerprint,
                "contract_source": contract.get("contract_source", ""),
                "contract_schema_version": contract.get("schema_version", ""),
                "workload_metadata": workload_metadata,
                "scenario": scenario,
                "conversation": conversation,
                "checks": checks,
            }
        )
    return compiled_instances


def _build_export_payload(compiled_instances: list[dict[str, Any]]) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    for item in compiled_instances:
        instances.append(
            {
                "instance_id": item["instance_id"],
                "name": item["name"],
                "family": item["benchmark_family"],
                "workload_metadata": item["workload_metadata"],
                "conversation": item["conversation"],
                "checks": item["checks"],
            }
        )
    return {
        "schema_version": "inferencex_v1",
        "instances": instances,
    }


def _compute_run_fingerprint(compiled_instances: list[dict[str, Any]]) -> str:
    return _stable_hash(
        {
            "compiled_instance_fingerprints": [
                item["instance_fingerprint"] for item in compiled_instances
            ]
        }
    )


def _build_bundle(
    config: CLIConfig,
    contracts: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    compiled_instances: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    run_fingerprint = _compute_run_fingerprint(compiled_instances)
    families = sorted({item["benchmark_family"] for item in compiled_instances})
    return {
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "run_fingerprint": run_fingerprint,
            "bundle_id": f"{finished_at.replace(':', '').replace('-', '')}_{run_fingerprint[:8]}",
        },
        "config": config.to_dict(),
        "summary": {
            "compile_only": config.compile_only,
            "contract_count": len(contracts),
            "scenario_count": len(scenarios),
            "compiled_instance_count": len(compiled_instances),
            "family_count": len(families),
            "families": families,
        },
        "authored_contracts": contracts,
        "compiled_instances": compiled_instances,
        "export_payloads": {"inferencex_v1": _build_export_payload(compiled_instances)},
    }


def _write_bundle(output_dir: Path, bundle: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    fingerprint = bundle["meta"]["run_fingerprint"][:8]
    output_path = output_dir / f"workload_bundle_{timestamp}_{fingerprint}.json"
    output_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def _print_families(contracts: list[dict[str, Any]]) -> None:
    family_counts: dict[str, int] = {}
    for contract in contracts:
        family = contract["benchmark_family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    for family in sorted(family_counts):
        print(f"{family}: {family_counts[family]}")


def _print_scenarios(contracts: list[dict[str, Any]]) -> None:
    for contract in sorted(contracts, key=lambda item: item["name"]):
        print(f"{contract['name']}: {contract['description']}")


def _print_bundle_summary(bundle: dict[str, Any], bundle_path: Path, scenario_path: Path) -> None:
    summary = bundle["summary"]
    print(f"Bundle ID: {bundle['meta']['bundle_id']}")
    print(f"Run fingerprint: {bundle['meta']['run_fingerprint']}")
    print(f"Contract count: {summary['contract_count']}")
    print(f"Scenario count: {summary['scenario_count']}")
    print(f"Compiled instances: {summary['compiled_instance_count']}")
    print(f"Families: {', '.join(summary['families'])}")
    print(f"Runtime scenarios: {scenario_path}")
    print(f"Bundle: {bundle_path}")


def _run_self_test() -> int:
    sample_contract = {
        "schema_version": "benchmark_contract_v1",
        "name": "self_test_contract",
        "description": "Self-test contract.",
        "benchmark_family": "interactive_short",
        "system_prompt": "Answer directly.",
        "facts": {"item": "Atlas", "date": "July 22"},
        "tags": ["self-test"],
        "workload_metadata": {
            "reasoning_class": "light",
            "context_growth_profile": "flat",
            "decode_stress": "low",
            "expected_response_size_class": "short",
            "isl_bucket": "small",
            "osl_bucket": "small",
        },
        "turns": [
            {
                "turn_id": "turn_01",
                "prompt": "What is the launch date for {{item}}?",
                "input_contract": {"task_type": "lookup"},
                "output_contract": {
                    "sentence_count": 1,
                    "keyword_counts": [{"keyword": "date", "count": 2}],
                },
                "correctness_checks": [
                    {"name": "mentions_date", "rule": "contains_all", "needles": ["{{date}}"]},
                ],
            }
        ],
    }

    from benchmark_contracts import compile_contract, validate_contract

    contract = validate_contract(sample_contract, source="<self-test>")
    scenario_a = compile_contract(contract)
    scenario_b = compile_contract(contract)
    if scenario_a != scenario_b:
        raise AssertionError("Contract compilation must be deterministic.")

    compiled_instances_a = _build_compiled_instances([contract], [scenario_a])
    compiled_instances_b = _build_compiled_instances([contract], [scenario_b])
    if compiled_instances_a[0]["instance_fingerprint"] != compiled_instances_b[0]["instance_fingerprint"]:
        raise AssertionError("Instance fingerprints should be stable for identical inputs.")

    metadata = compiled_instances_a[0]["workload_metadata"]
    expected_fields = set(REQUIRED_WORKLOAD_METADATA_FIELDS) | {"output_contract_type", "benchmark_family", "turn_count"}
    missing = sorted(field for field in expected_fields if field not in metadata)
    if missing:
        raise AssertionError(f"Compiled workload metadata missing fields: {', '.join(missing)}")

    print("Self-test passed.")
    return 0


def _parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compile contract-authored performance workloads into runtime eval scenarios and "
            "InferenceX-style export bundles."
        )
    )
    parser.add_argument("--contract-input", default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-output", default=DEFAULT_SCENARIO_OUTPUT)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--list-families", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    family_names: list[str] = []
    for item in [*args.family, *args.pattern]:
        family_names.extend(part.strip() for part in str(item).split(",") if part.strip())

    scenario_names: list[str] = []
    for item in args.scenario:
        scenario_names.extend(part.strip() for part in str(item).split(",") if part.strip())

    return CLIConfig(
        contract_input=str(args.contract_input),
        output_dir=str(args.output_dir),
        scenario_output=str(args.scenario_output),
        family_names=family_names,
        scenario_names=scenario_names,
        compile_only=bool(args.compile_only),
        list_families=bool(args.list_families),
        list_scenarios=bool(args.list_scenarios),
        self_test=bool(args.self_test),
    )


def main() -> int:
    try:
        config = _parse_args()
        if config.self_test:
            return _run_self_test()

        contracts = load_contracts(Path(config.contract_input))
        selected_contracts = _select_contracts(contracts, config.family_names, config.scenario_names)

        if config.list_families:
            _print_families(selected_contracts)
            if not config.list_scenarios:
                return 0
        if config.list_scenarios:
            _print_scenarios(selected_contracts)
            return 0

        started_at = _now_iso()
        scenarios = compile_contracts_to_dataset(selected_contracts)
        scenario_path = write_dataset(Path(config.scenario_output), scenarios)
        compiled_instances = _build_compiled_instances(selected_contracts, scenarios)
        finished_at = _now_iso()
        bundle = _build_bundle(
            config=config,
            contracts=selected_contracts,
            scenarios=scenarios,
            compiled_instances=compiled_instances,
            started_at=started_at,
            finished_at=finished_at,
        )
        bundle_path = _write_bundle(Path(config.output_dir), bundle)
        _print_bundle_summary(bundle, bundle_path, scenario_path)
        if not config.compile_only:
            print("Execution is intentionally handled by tinker_conversation_eval.py; this command compiled only.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
