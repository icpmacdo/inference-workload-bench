#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = "benchmark_contract_v1"
DEFAULT_SCENARIO_OUTPUT = "scenario_datasets/benchmark_contract_dataset_v1.json"

TEMPLATE_RE = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")

SUPPORTED_INPUT_SHAPE_FIELDS: tuple[str, ...] = ("min_input_character_count",)

SUPPORTED_OUTPUT_CONTRACT_FIELDS: tuple[str, ...] = (
    "keyword_counts",
    "sentence_count",
    "bullet_count",
    "max_line_count",
    "min_line_count",
    "min_character_count",
    "required_section_headers",
)
SUPPORTED_OUTPUT_CONTRACT_FIELD_SET = set(SUPPORTED_OUTPUT_CONTRACT_FIELDS)

REQUIRED_WORKLOAD_METADATA_FIELDS: tuple[str, ...] = (
    "reasoning_class",
    "context_growth_profile",
    "decode_stress",
    "expected_response_size_class",
    "isl_bucket",
    "osl_bucket",
)
REQUIRED_WORKLOAD_METADATA_FIELD_SET = set(REQUIRED_WORKLOAD_METADATA_FIELDS)

OUTPUT_CONTRACT_TYPE_NAMES: dict[str, str] = {
    "keyword_counts": "exact_keyword_count",
    "sentence_count": "exact_sentence_count",
    "bullet_count": "exact_bullet_count",
    "max_line_count": "max_line_count",
    "min_line_count": "min_line_count",
    "min_character_count": "min_character_count",
    "required_section_headers": "required_section_headers",
}


def _validate_allowed_keys(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys in {label}: {', '.join(unknown)}")


def _ensure_non_empty_string(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string.")
    return text


def _ensure_positive_int(value: Any, label: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _resolve_templates(value: Any, facts: dict[str, str]) -> Any:
    if isinstance(value, str):
        return TEMPLATE_RE.sub(lambda match: facts.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_resolve_templates(item, facts) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_templates(item, facts) for key, item in value.items()}
    return value


def _normalize_check_payload(
    payload: dict[str, Any],
    *,
    label: str,
    facts: dict[str, str] | None = None,
) -> dict[str, Any]:
    _validate_allowed_keys(payload, set(payload), label)
    if "name" not in payload or "rule" not in payload:
        raise ValueError(f"{label} must include name and rule.")
    normalized = {
        "name": _ensure_non_empty_string(payload["name"], f"{label}.name"),
        "rule": _ensure_non_empty_string(payload["rule"], f"{label}.rule"),
        "weight": float(payload.get("weight", 1.0)),
    }
    description = str(payload.get("description", "")).strip()
    if description:
        normalized["description"] = description
    for key, value in payload.items():
        if key not in {"name", "rule", "weight", "description"}:
            normalized[key] = value
    if facts is not None:
        normalized = _resolve_templates(normalized, facts)
    return normalized


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _validate_output_contract(payload: dict[str, Any], label: str) -> dict[str, Any]:
    _validate_allowed_keys(payload, SUPPORTED_OUTPUT_CONTRACT_FIELD_SET, label)
    if not payload:
        raise ValueError(f"{label} must define at least one supported output rule.")

    normalized: dict[str, Any] = {}
    if "keyword_counts" in payload:
        raw_items = payload["keyword_counts"]
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"{label}.keyword_counts must be a non-empty list.")
        items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{label}.keyword_counts[{index}] must be an object.")
            _validate_allowed_keys(item, {"keyword", "count"}, f"{label}.keyword_counts[{index}]")
            items.append(
                {
                    "keyword": _ensure_non_empty_string(
                        item.get("keyword", ""), f"{label}.keyword_counts[{index}].keyword"
                    ),
                    "count": _ensure_positive_int(
                        item.get("count"), f"{label}.keyword_counts[{index}].count"
                    ),
                }
            )
        normalized["keyword_counts"] = items

    for key in ("sentence_count", "bullet_count", "max_line_count", "min_line_count", "min_character_count"):
        if key in payload:
            normalized[key] = _ensure_positive_int(payload[key], f"{label}.{key}")

    if "required_section_headers" in payload:
        raw_headers = payload["required_section_headers"]
        if not isinstance(raw_headers, list) or not raw_headers:
            raise ValueError(f"{label}.required_section_headers must be a non-empty list.")
        normalized["required_section_headers"] = [
            _ensure_non_empty_string(item, f"{label}.required_section_headers[{index}]")
            for index, item in enumerate(raw_headers, start=1)
        ]

    return normalized


def _validate_input_contract(payload: dict[str, Any], label: str) -> dict[str, Any]:
    normalized = dict(payload)
    for key in SUPPORTED_INPUT_SHAPE_FIELDS:
        if key in normalized:
            normalized[key] = _ensure_positive_int(normalized[key], f"{label}.{key}")
    return normalized


def _validate_turn_payload(payload: dict[str, Any], label: str) -> dict[str, Any]:
    _validate_allowed_keys(
        payload,
        {"turn_id", "prompt", "input_contract", "output_contract", "correctness_checks"},
        label,
    )
    if (
        "prompt" not in payload
        or "input_contract" not in payload
        or "output_contract" not in payload
        or "correctness_checks" not in payload
    ):
        raise ValueError(
            f"{label} must include prompt, input_contract, output_contract, and correctness_checks."
        )
    if not isinstance(payload["input_contract"], dict):
        raise ValueError(f"{label}.input_contract must be a JSON object.")
    turn_id = str(payload.get("turn_id", "")).strip()
    normalized = {
        "turn_id": turn_id,
        "prompt": _ensure_non_empty_string(payload["prompt"], f"{label}.prompt"),
        "input_contract": _validate_input_contract(
            dict(payload["input_contract"]),
            f"{label}.input_contract",
        ),
        "output_contract": _validate_output_contract(
            dict(payload["output_contract"]),
            f"{label}.output_contract",
        ),
        "correctness_checks": [],
    }
    raw_checks = payload.get("correctness_checks", [])
    if not isinstance(raw_checks, list):
        raise ValueError(f"{label}.correctness_checks must be a list.")
    for index, check in enumerate(raw_checks, start=1):
        if not isinstance(check, dict):
            raise ValueError(f"{label}.correctness_checks[{index}] must be an object.")
        normalized["correctness_checks"].append(
            _normalize_check_payload(check, label=f"{label}.correctness_checks[{index}]")
        )
    return normalized


def _validate_workload_metadata(payload: dict[str, Any], label: str) -> dict[str, Any]:
    _validate_allowed_keys(payload, REQUIRED_WORKLOAD_METADATA_FIELD_SET, label)
    missing = [field for field in REQUIRED_WORKLOAD_METADATA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")
    return {
        key: _ensure_non_empty_string(payload[key], f"{label}.{key}")
        for key in REQUIRED_WORKLOAD_METADATA_FIELDS
    }


def validate_contract(payload: dict[str, Any], *, source: str = "<memory>") -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Contract {source} must be a JSON object.")
    _validate_allowed_keys(
        payload,
        {
            "schema_version",
            "name",
            "description",
            "benchmark_family",
            "system_prompt",
            "facts",
            "tags",
            "workload_metadata",
            "turns",
        },
        f"contract {source}",
    )
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"Contract {source} must use schema_version={CONTRACT_SCHEMA_VERSION!r}."
        )
    facts_raw = payload.get("facts", {})
    if not isinstance(facts_raw, dict):
        raise ValueError(f"Contract {source}.facts must be a JSON object when present.")
    facts = {str(key): str(value) for key, value in facts_raw.items()}
    tags_raw = payload.get("tags", [])
    if not isinstance(tags_raw, list):
        raise ValueError(f"Contract {source}.tags must be a list.")
    turns_raw = payload.get("turns", [])
    if not isinstance(turns_raw, list) or not turns_raw:
        raise ValueError(f"Contract {source}.turns must be a non-empty list.")

    normalized = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "name": _ensure_non_empty_string(payload.get("name"), f"contract {source}.name"),
        "description": _ensure_non_empty_string(
            payload.get("description"), f"contract {source}.description"
        ),
        "benchmark_family": _ensure_non_empty_string(
            payload.get("benchmark_family"), f"contract {source}.benchmark_family"
        ),
        "system_prompt": _ensure_non_empty_string(
            payload.get("system_prompt"), f"contract {source}.system_prompt"
        ),
        "facts": facts,
        "tags": [_ensure_non_empty_string(item, f"contract {source}.tags") for item in tags_raw],
        "workload_metadata": _validate_workload_metadata(
            dict(payload.get("workload_metadata", {})),
            f"contract {source}.workload_metadata",
        ),
        "turns": [],
        "contract_source": source,
    }

    for index, turn in enumerate(turns_raw, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"Contract {source}.turns[{index}] must be an object.")
        normalized["turns"].append(
            _validate_turn_payload(turn, f"contract {source}.turns[{index}]")
        )
    return normalized


def load_contracts(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_file():
        return _load_contract_file(input_path)
    if input_path.is_dir():
        contracts: list[dict[str, Any]] = []
        for path in sorted(input_path.glob("*.json")):
            contracts.extend(_load_contract_file(path))
        if not contracts:
            raise FileNotFoundError(f"No contract JSON files found in directory: {input_path}")
        return contracts
    raise FileNotFoundError(f"Contract input not found: {input_path}")


def _load_contract_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return [validate_contract(payload, source=str(path))]
    if isinstance(payload, list):
        return [validate_contract(item, source=str(path)) for item in payload]
    raise ValueError(f"Contract file must contain a JSON object or list: {path}")


def _output_instruction_fragments(output_contract: dict[str, Any]) -> list[str]:
    fragments: list[str] = []

    for item in output_contract.get("keyword_counts", []):
        fragments.append(
            f'Use the lowercase word "{item["keyword"]}" exactly {int(item["count"])} times.'
        )

    if "sentence_count" in output_contract:
        count = int(output_contract["sentence_count"])
        fragments.append(
            "Answer in exactly one sentence." if count == 1 else f"Answer in exactly {count} sentences."
        )

    if "bullet_count" in output_contract:
        count = int(output_contract["bullet_count"])
        fragments.append(
            f"Respond with exactly {count} bullet {'point' if count == 1 else 'points'}."
        )

    if "max_line_count" in output_contract:
        fragments.append(
            f'Use at most {int(output_contract["max_line_count"])} non-empty lines.'
        )

    if "min_line_count" in output_contract:
        fragments.append(
            f'Use at least {int(output_contract["min_line_count"])} non-empty lines.'
        )

    if "min_character_count" in output_contract:
        fragments.append(
            f'Write at least {int(output_contract["min_character_count"])} characters.'
        )

    if "required_section_headers" in output_contract:
        header_list = ", ".join(str(item) for item in output_contract["required_section_headers"])
        fragments.append(
            f"Include these section headers exactly, each on its own line: {header_list}."
        )

    return fragments


def build_output_checks(output_contract: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    for item in output_contract.get("keyword_counts", []):
        keyword = str(item["keyword"])
        count = int(item["count"])
        checks.append(
            {
                "name": f"uses_{_slug(keyword)}_exactly_{count}_times",
                "rule": "word_count_equals",
                "weight": 1.0,
                "word": keyword,
                "count": count,
            }
        )

    if "sentence_count" in output_contract:
        count = int(output_contract["sentence_count"])
        checks.append(
            {
                "name": f"uses_exactly_{count}_sentences",
                "rule": "sentence_count_equals",
                "weight": 1.0,
                "count": count,
            }
        )

    if "bullet_count" in output_contract:
        count = int(output_contract["bullet_count"])
        checks.append(
            {
                "name": f"uses_exactly_{count}_bullets",
                "rule": "bullet_count_equals",
                "weight": 1.0,
                "count": count,
            }
        )

    if "max_line_count" in output_contract:
        maximum = int(output_contract["max_line_count"])
        checks.append(
            {
                "name": f"uses_at_most_{maximum}_lines",
                "rule": "line_count_at_most",
                "weight": 1.0,
                "max": maximum,
            }
        )

    if "min_line_count" in output_contract:
        minimum = int(output_contract["min_line_count"])
        checks.append(
            {
                "name": f"uses_at_least_{minimum}_lines",
                "rule": "line_count_at_least",
                "weight": 1.0,
                "count": minimum,
            }
        )

    if "min_character_count" in output_contract:
        minimum = int(output_contract["min_character_count"])
        checks.append(
            {
                "name": f"uses_at_least_{minimum}_characters",
                "rule": "char_count_at_least",
                "weight": 1.0,
                "count": minimum,
            }
        )

    if "required_section_headers" in output_contract:
        checks.append(
            {
                "name": "required_section_headers_present",
                "rule": "section_headers_present",
                "weight": 1.0,
                "headers": list(output_contract["required_section_headers"]),
            }
        )

    return checks


def output_contract_types(output_contract: dict[str, Any]) -> list[str]:
    return [
        OUTPUT_CONTRACT_TYPE_NAMES[key]
        for key in SUPPORTED_OUTPUT_CONTRACT_FIELDS
        if key in output_contract
    ]


def _enforce_input_shape(
    *,
    contract_name: str,
    turn_id: str,
    input_contract: dict[str, Any],
    user_message: str,
) -> None:
    minimum_chars = input_contract.get("min_input_character_count")
    if minimum_chars is None:
        return
    actual = len(user_message.strip())
    minimum = int(minimum_chars)
    if actual < minimum:
        raise ValueError(
            f"{contract_name}.{turn_id} produced input length {actual}, below required "
            f"min_input_character_count={minimum}."
        )


def compile_contract(contract: dict[str, Any]) -> dict[str, Any]:
    facts = dict(contract.get("facts", {}))
    resolved_system_prompt = str(_resolve_templates(contract["system_prompt"], facts)).strip()
    resolved_workload_metadata = _resolve_templates(contract["workload_metadata"], facts)
    turns: list[dict[str, Any]] = []
    scenario_contract_types: list[str] = []

    for index, raw_turn in enumerate(contract["turns"], start=1):
        resolved_input_contract = _resolve_templates(raw_turn["input_contract"], facts)
        output_contract = _resolve_templates(raw_turn["output_contract"], facts)
        prompt = str(_resolve_templates(raw_turn["prompt"], facts)).strip()
        instruction_fragments = _output_instruction_fragments(output_contract)
        user_message = " ".join([prompt, *instruction_fragments]).strip()
        turn_id = raw_turn.get("turn_id") or f"turn_{index:02d}"
        _enforce_input_shape(
            contract_name=contract["name"],
            turn_id=str(turn_id),
            input_contract=resolved_input_contract,
            user_message=user_message,
        )
        authored_checks = [
            _normalize_check_payload(
                _resolve_templates(check, facts),
                label=f"{contract['name']}.turn_{index}.correctness_checks",
            )
            for check in raw_turn.get("correctness_checks", [])
        ]
        generated_checks = build_output_checks(output_contract)
        turn_output_types = output_contract_types(output_contract)
        scenario_contract_types.extend(turn_output_types)

        turns.append(
            {
                "turn_id": turn_id,
                "prompt": prompt,
                "user": user_message,
                "input_contract": resolved_input_contract,
                "output_contract": output_contract,
                "correctness_checks": authored_checks,
                "generated_checks": generated_checks,
                "checks": generated_checks + authored_checks,
            }
        )

    workload_metadata = dict(resolved_workload_metadata)
    workload_metadata["output_contract_type"] = sorted(dict.fromkeys(scenario_contract_types))

    return {
        "name": contract["name"],
        "description": contract["description"],
        "system_prompt": resolved_system_prompt,
        "facts": facts,
        "tags": list(contract.get("tags", [])),
        "benchmark_family": contract["benchmark_family"],
        "workload_metadata": workload_metadata,
        "contract_source": contract.get("contract_source"),
        "contract_schema_version": contract["schema_version"],
        "turns": turns,
    }


def compile_contracts_to_dataset(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compile_contract(contract) for contract in sorted(contracts, key=lambda item: item["name"])]


def write_dataset(output_path: Path, dataset: list[dict[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path
