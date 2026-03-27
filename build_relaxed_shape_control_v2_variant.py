#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_contracts import compile_contracts_to_dataset, validate_contract, write_dataset

DEFAULT_SOURCE = "benchmark_contracts/shape_control_v2.json"
DEFAULT_CONTRACT_OUTPUT = "benchmark_contracts/shape_control_v2_relaxed.json"
DEFAULT_DATASET_OUTPUT = "scenario_datasets/shape_control_v2_relaxed.json"


def _rename_scenario(name: str) -> str:
    return name.replace("shape_control_v2_", "shape_control_v2_relaxed_", 1)


def _rename_tag(tag: str) -> str:
    if tag == "shape_control_v2":
        return "shape_control_v2_relaxed"
    return tag


def _transform_turn(turn: dict[str, Any]) -> dict[str, Any]:
    output_contract = dict(turn["output_contract"])
    output_contract.pop("min_line_count", None)

    correctness_checks = [
        check
        for check in turn.get("correctness_checks", [])
        if str(check.get("rule", "")) != "contains_all"
    ]

    return {
        **turn,
        "output_contract": output_contract,
        "correctness_checks": correctness_checks,
    }


def _transform_contract(contract: dict[str, Any]) -> dict[str, Any]:
    tags = [_rename_tag(str(tag)) for tag in contract.get("tags", [])]
    if "relaxed-scoring" not in tags:
        tags.append("relaxed-scoring")

    transformed = {
        **contract,
        "name": _rename_scenario(str(contract["name"])),
        "description": f"{contract['description']} Relaxed scoring removes line-count and contains-all checks.",
        "tags": tags,
        "turns": [_transform_turn(turn) for turn in contract.get("turns", [])],
    }
    return validate_contract(transformed, source="<generated shape_control_v2_relaxed>")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a relaxed shape_control_v2 contract variant that keeps the same scenarios "
            "and turns while removing line-count and contains-all checks."
        )
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--contract-output", default=DEFAULT_CONTRACT_OUTPUT)
    parser.add_argument("--dataset-output", default=DEFAULT_DATASET_OUTPUT)
    args = parser.parse_args()

    source_path = Path(args.source)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of contracts in {source_path}")

    transformed_contracts = [_transform_contract(contract) for contract in payload]

    contract_output_path = Path(args.contract_output)
    contract_output_path.parent.mkdir(parents=True, exist_ok=True)
    contract_output_path.write_text(
        json.dumps(transformed_contracts, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    dataset = compile_contracts_to_dataset(transformed_contracts)
    dataset_output_path = write_dataset(Path(args.dataset_output), dataset)

    print(contract_output_path)
    print(dataset_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
