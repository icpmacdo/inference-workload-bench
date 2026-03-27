#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Contract file must contain a top-level JSON object.")
    return payload


def make_user_prompt(question: str, output_contract: dict[str, Any]) -> str:
    sentence_count = int(output_contract["sentence_count"])
    keyword = str(output_contract["keyword"])
    keyword_count = int(output_contract["keyword_count"])
    sentence_phrase = "one sentence" if sentence_count == 1 else f"exactly {sentence_count} sentences"
    return (
        f"{question} Answer in {sentence_phrase} and use the lowercase word "
        f'"{keyword}" exactly {keyword_count} times.'
    )


def make_checks(turn_contract: dict[str, Any]) -> list[dict[str, Any]]:
    output_contract = dict(turn_contract["output_contract"])
    checks = [
        {
            "name": f"uses_{output_contract['keyword']}_exactly_{output_contract['keyword_count']}_times",
            "rule": "word_count_equals",
            "weight": 1.0,
            "word": output_contract["keyword"],
            "count": output_contract["keyword_count"],
        },
        {
            "name": f"uses_exactly_{output_contract['sentence_count']}_sentences",
            "rule": "sentence_count_equals",
            "weight": 1.0,
            "count": output_contract["sentence_count"],
        },
    ]
    for check in turn_contract.get("correctness_checks", []):
        normalized = {
            "name": check["name"],
            "rule": check["rule"],
            "weight": float(check.get("weight", 1.0)),
        }
        for key, value in check.items():
            if key not in {"name", "rule", "weight"}:
                normalized[key] = value
        checks.append(normalized)
    return checks


def build_scenario(contract: dict[str, Any]) -> list[dict[str, Any]]:
    turns = []
    for turn_contract in contract["turns"]:
        turns.append(
            {
                "user": make_user_prompt(str(turn_contract["question"]), dict(turn_contract["output_contract"])),
                "input_contract": turn_contract["input_contract"],
                "output_contract": turn_contract["output_contract"],
                "checks": make_checks(turn_contract),
            }
        )

    scenario = {
        "name": contract["name"],
        "description": contract["description"],
        "system_prompt": contract["system_prompt"],
        "facts": {},
        "turns": turns,
        "tags": contract.get("tags", []),
    }
    return [scenario]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a conversation-eval scenario JSON from a keyword-count contract file."
    )
    parser.add_argument("input", help="Contract JSON file path.")
    parser.add_argument(
        "--output",
        default="scenario_datasets/next_run_keyword_count_10_turn.json",
        help="Output scenario JSON file path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    contract = load_contract(input_path)
    scenario_payload = build_scenario(contract)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenario_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
