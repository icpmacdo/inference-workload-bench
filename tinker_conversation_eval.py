#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent
COOKBOOK_ROOT = REPO_ROOT / "tinker-cookbook"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TEMPLATE_RE = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


@dataclass(slots=True)
class RuntimeConfig:
    base_model: str
    model_path: str | None
    base_url: str | None
    renderer_name: str | None
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    output_dir: str
    scenario_names: list[str]
    scenario_file: str | None
    list_scenarios: bool
    self_test: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "model_path": self.model_path,
            "base_url": self.base_url,
            "renderer_name": self.renderer_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "scenario_names": self.scenario_names,
            "scenario_file": self.scenario_file,
            "list_scenarios": self.list_scenarios,
            "self_test": self.self_test,
        }


@dataclass(slots=True)
class Check:
    name: str
    rule: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "rule": self.rule,
            "weight": self.weight,
        }
        if self.description:
            payload["description"] = self.description
        payload.update(self.params)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Check:
        params = {
            key: value
            for key, value in payload.items()
            if key not in {"name", "rule", "weight", "description"}
        }
        return cls(
            name=str(payload["name"]),
            rule=str(payload["rule"]),
            params=params,
            weight=float(payload.get("weight", 1.0)),
            description=str(payload.get("description", "")),
        )


@dataclass(slots=True)
class ScenarioTurn:
    user: str
    checks: list[Check]
    prompt: str = ""
    turn_id: str = ""
    input_contract: dict[str, Any] = field(default_factory=dict)
    output_contract: dict[str, Any] = field(default_factory=dict)
    correctness_checks: list[Check] = field(default_factory=list)
    generated_checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "user": self.user,
            "checks": [check.to_dict() for check in self.checks],
        }
        if self.prompt:
            payload["prompt"] = self.prompt
        if self.turn_id:
            payload["turn_id"] = self.turn_id
        if self.input_contract:
            payload["input_contract"] = self.input_contract
        if self.output_contract:
            payload["output_contract"] = self.output_contract
        if self.correctness_checks:
            payload["correctness_checks"] = [check.to_dict() for check in self.correctness_checks]
        if self.generated_checks:
            payload["generated_checks"] = [check.to_dict() for check in self.generated_checks]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScenarioTurn:
        checks = [Check.from_dict(item) for item in payload.get("checks", [])]
        correctness_checks = [
            Check.from_dict(item) for item in payload.get("correctness_checks", [])
        ]
        generated_checks = [
            Check.from_dict(item) for item in payload.get("generated_checks", [])
        ]
        return cls(
            user=str(payload["user"]),
            checks=checks,
            prompt=str(payload.get("prompt", "")),
            turn_id=str(payload.get("turn_id", "")),
            input_contract=dict(payload.get("input_contract", {})),
            output_contract=dict(payload.get("output_contract", {})),
            correctness_checks=correctness_checks,
            generated_checks=generated_checks,
        )


@dataclass(slots=True)
class Scenario:
    name: str
    description: str
    system_prompt: str
    facts: dict[str, str]
    turns: list[ScenarioTurn]
    tags: list[str] = field(default_factory=list)
    benchmark_family: str = ""
    workload_metadata: dict[str, Any] = field(default_factory=dict)
    contract_source: str = ""
    contract_schema_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "facts": self.facts,
            "turns": [turn.to_dict() for turn in self.turns],
            "tags": self.tags,
        }
        if self.benchmark_family:
            payload["benchmark_family"] = self.benchmark_family
        if self.workload_metadata:
            payload["workload_metadata"] = self.workload_metadata
        if self.contract_source:
            payload["contract_source"] = self.contract_source
        if self.contract_schema_version:
            payload["contract_schema_version"] = self.contract_schema_version
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Scenario:
        turns = [ScenarioTurn.from_dict(item) for item in payload.get("turns", [])]
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            system_prompt=str(payload.get("system_prompt", "")),
            facts={str(key): str(value) for key, value in payload.get("facts", {}).items()},
            turns=turns,
            tags=[str(tag) for tag in payload.get("tags", [])],
            benchmark_family=str(payload.get("benchmark_family", "")),
            workload_metadata=dict(payload.get("workload_metadata", {})),
            contract_source=str(payload.get("contract_source", "")),
            contract_schema_version=str(payload.get("contract_schema_version", "")),
        )


@dataclass(slots=True)
class CheckResult:
    name: str
    rule: str
    passed: bool
    weight: float
    score: float
    details: str
    resolved_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule": self.rule,
            "passed": self.passed,
            "weight": self.weight,
            "score": self.score,
            "details": self.details,
            "resolved_params": self.resolved_params,
        }


@dataclass(slots=True)
class TurnTrace:
    index: int
    user_message: str
    prompt_messages: list[dict[str, str]]
    assistant_message: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int
    stop_reason: str
    checks: list[CheckResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "user_message": self.user_message,
            "prompt_messages": self.prompt_messages,
            "assistant_message": self.assistant_message,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stop_reason": self.stop_reason,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class ScenarioResult:
    name: str
    description: str
    passed: bool
    score: float
    max_score: float
    traces: list[TurnTrace]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "score_pct": 0.0 if self.max_score == 0 else round((self.score / self.max_score) * 100, 2),
            "traces": [trace.to_dict() for trace in self.traces],
        }


@dataclass(slots=True)
class RuntimeHandles:
    sampling_client: Any
    renderer: Any
    renderers_module: Any
    resolved_renderer_name: str
    tinker_types: Any


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sanitize_text(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _bullet_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if BULLET_RE.match(line)]


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
    return len(parts)


def _section_headers_present(text: str, headers: list[str]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for header in headers:
        pattern = re.compile(rf"(?im)^\s*{re.escape(header)}\s*:?\s*$")
        if pattern.search(text) is None:
            missing.append(header)
    return (not missing, missing)


def _resolve_templates(value: Any, facts: dict[str, str]) -> Any:
    if isinstance(value, str):
        return TEMPLATE_RE.sub(lambda match: facts.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_resolve_templates(item, facts) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_templates(item, facts) for key, item in value.items()}
    return value


def _ordered_contains(text: str, needles: list[str]) -> bool:
    cursor = 0
    haystack = _normalize_text(text)
    for needle in needles:
        normalized = _normalize_text(needle)
        if not normalized:
            continue
        position = haystack.find(normalized, cursor)
        if position == -1:
            return False
        cursor = position + len(normalized)
    return True


def _evaluate_check(response_text: str, check: Check, facts: dict[str, str]) -> CheckResult:
    resolved_params = _resolve_templates(check.params, facts)
    response_normalized = _normalize_text(response_text)
    passed = False
    details = ""

    if check.rule == "contains_all":
        needles = [str(item) for item in resolved_params.get("needles", [])]
        missing = [needle for needle in needles if _normalize_text(needle) not in response_normalized]
        passed = not missing
        details = "all required substrings present" if passed else f"missing: {missing}"
    elif check.rule == "contains_none":
        needles = [str(item) for item in resolved_params.get("needles", [])]
        present = [needle for needle in needles if _normalize_text(needle) in response_normalized]
        passed = not present
        details = "no forbidden substrings present" if passed else f"found forbidden: {present}"
    elif check.rule == "contains_any":
        needles = [str(item) for item in resolved_params.get("needles", [])]
        present = [needle for needle in needles if _normalize_text(needle) in response_normalized]
        passed = bool(present)
        details = f"matched: {present}" if passed else f"expected one of: {needles}"
    elif check.rule == "regex":
        pattern = str(resolved_params["pattern"])
        matched = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        passed = matched is not None
        details = "regex matched" if passed else f"regex did not match: {pattern}"
    elif check.rule == "regex_none":
        pattern = str(resolved_params["pattern"])
        matched = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        passed = matched is None
        details = "regex absent as expected" if passed else f"forbidden regex matched: {pattern}"
    elif check.rule == "ordered_substrings":
        needles = [str(item) for item in resolved_params.get("needles", [])]
        passed = _ordered_contains(response_text, needles)
        details = "substrings appeared in order" if passed else f"order mismatch: {needles}"
    elif check.rule == "line_count_at_most":
        maximum = int(resolved_params["max"])
        line_count = len(_nonempty_lines(response_text))
        passed = line_count <= maximum
        details = (
            f"non-empty line count {line_count} <= {maximum}"
            if passed
            else f"non-empty line count {line_count} > {maximum}"
        )
    elif check.rule == "line_count_at_least":
        minimum = int(resolved_params["count"])
        line_count = len(_nonempty_lines(response_text))
        passed = line_count >= minimum
        details = (
            f"non-empty line count {line_count} >= {minimum}"
            if passed
            else f"non-empty line count {line_count} < {minimum}"
        )
    elif check.rule == "bullet_count_equals":
        expected = int(resolved_params["count"])
        actual = len(_bullet_lines(response_text))
        passed = actual == expected
        details = f"bullet count {actual} == {expected}" if passed else f"bullet count {actual} != {expected}"
    elif check.rule == "bullet_count_at_most":
        maximum = int(resolved_params["max"])
        actual = len(_bullet_lines(response_text))
        passed = actual <= maximum
        details = f"bullet count {actual} <= {maximum}" if passed else f"bullet count {actual} > {maximum}"
    elif check.rule == "question_count_equals":
        expected = int(resolved_params["count"])
        actual = response_text.count("?")
        passed = actual == expected
        details = (
            f"question count {actual} == {expected}"
            if passed
            else f"question count {actual} != {expected}"
        )
    elif check.rule == "sentence_count_equals":
        expected = int(resolved_params["count"])
        actual = _sentence_count(response_text)
        passed = actual == expected
        details = (
            f"sentence count {actual} == {expected}"
            if passed
            else f"sentence count {actual} != {expected}"
        )
    elif check.rule == "word_count_equals":
        word = str(resolved_params["word"])
        expected = int(resolved_params["count"])
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        actual = len(pattern.findall(response_text))
        passed = actual == expected
        details = (
            f"word count for '{word}' {actual} == {expected}"
            if passed
            else f"word count for '{word}' {actual} != {expected}"
        )
    elif check.rule == "char_count_at_least":
        minimum = int(resolved_params["count"])
        actual = len(response_text.strip())
        passed = actual >= minimum
        details = (
            f"character count {actual} >= {minimum}"
            if passed
            else f"character count {actual} < {minimum}"
        )
    elif check.rule == "section_headers_present":
        headers = [str(item) for item in resolved_params.get("headers", [])]
        passed, missing = _section_headers_present(response_text, headers)
        details = "all required section headers present" if passed else f"missing headers: {missing}"
    else:
        raise ValueError(f"Unsupported rule: {check.rule}")

    return CheckResult(
        name=check.name,
        rule=check.rule,
        passed=passed,
        weight=check.weight,
        score=check.weight if passed else 0.0,
        details=details,
        resolved_params=resolved_params,
    )


def _build_builtin_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="expense_policy_memory",
            description=(
                "Tests whether the assistant can carry policy facts forward, classify multiple "
                "items correctly, and avoid reintroducing rejected items later."
            ),
            system_prompt=(
                "You are an internal operations assistant. Stay grounded in the conversation, "
                "answer directly, and do not invent policy exceptions."
            ),
            facts={
                "trip_code": "TRIP-4821",
                "hotel_cap": "$180",
                "meal_cap": "$65",
            },
            turns=[
                ScenarioTurn(
                    user=(
                        "I'm going to give you a reimbursement policy to use later. Hotel: up to "
                        "$180/night with receipt. Meals: up to $65/day. Airport rideshare is "
                        "reimbursable. Minibar or alcohol is never reimbursable. My trip code is "
                        "TRIP-4821. Reply with one sentence confirming the trip code and the hotel "
                        "and meal caps."
                    ),
                    checks=[
                        Check(
                            name="acknowledges_key_policy_facts",
                            rule="contains_all",
                            params={"needles": ["{{trip_code}}", "{{hotel_cap}}", "{{meal_cap}}"]},
                            description="The assistant should capture the exact values it must remember.",
                        ),
                        Check(
                            name="ack_is_compact",
                            rule="line_count_at_most",
                            params={"max": 2},
                            description="The acknowledgement should stay concise.",
                        ),
                    ],
                ),
                ScenarioTurn(
                    user="Now evaluate these expenses and answer in exactly three bullets: $172 hotel, $24 airport rideshare, $19 minibar.",
                    checks=[
                        Check(
                            name="uses_three_bullets",
                            rule="bullet_count_equals",
                            params={"count": 3},
                        ),
                        Check(
                            name="hotel_is_approved",
                            rule="regex",
                            params={
                                "pattern": r"(?is)(\$172|hotel).*(reimburs|covered|eligible|approved)"
                            },
                        ),
                        Check(
                            name="rideshare_is_approved",
                            rule="regex",
                            params={
                                "pattern": r"(?is)(\$24|rideshare).*(reimburs|covered|eligible|approved)"
                            },
                        ),
                        Check(
                            name="minibar_is_rejected",
                            rule="regex",
                            params={
                                "pattern": r"(?is)(\$19|minibar).*((not|never|isn't|is not).*(reimburs|covered|eligible|approved)|excluded|ineligible)"
                            },
                        ),
                    ],
                ),
                ScenarioTurn(
                    user="Give me a one-sentence summary of only the approved items and include the trip code exactly.",
                    checks=[
                        Check(
                            name="summary_keeps_trip_code",
                            rule="contains_all",
                            params={"needles": ["{{trip_code}}"]},
                        ),
                        Check(
                            name="summary_mentions_both_approved_items",
                            rule="regex",
                            params={
                                "pattern": r"(?is)((hotel|\$172).*(rideshare|\$24))|((rideshare|\$24).*(hotel|\$172))"
                            },
                        ),
                        Check(
                            name="summary_omits_rejected_item",
                            rule="regex_none",
                            params={"pattern": r"(?is)(minibar|\$19)"},
                        ),
                    ],
                ),
            ],
            tags=["memory", "policy", "consistency"],
        ),
        Scenario(
            name="support_handoff_precision",
            description=(
                "Tests conversational discipline, exact field formatting, and retention of facts "
                "introduced across multiple turns."
            ),
            system_prompt=(
                "You are a support operations assistant. Follow formatting instructions exactly, "
                "retain incident details across turns, and do not invent missing evidence."
            ),
            facts={
                "ticket_id": "SUP-7714",
                "customer_name": "Mina Patel",
                "environment": "staging",
                "region": "us-west-2",
            },
            turns=[
                ScenarioTurn(
                    user=(
                        "I need a support handoff note later. Facts to remember: customer Mina "
                        "Patel, ticket SUP-7714, issue started right after a password reset, and "
                        "it only reproduces in staging. Before drafting anything, ask me exactly "
                        "one clarifying question."
                    ),
                    checks=[
                        Check(
                            name="asks_one_clarifying_question",
                            rule="question_count_equals",
                            params={"count": 1},
                        ),
                    ],
                ),
                ScenarioTurn(
                    user=(
                        "The affected region is us-west-2 and there were no 2FA errors. Now draft "
                        "the handoff note using exactly these labels on separate lines: Ticket, "
                        "Customer, Problem, Environment, Region, Next Step."
                    ),
                    checks=[
                        Check(
                            name="ticket_line_exact",
                            rule="regex",
                            params={"pattern": r"(?im)^Ticket:\s*SUP-7714\s*$"},
                        ),
                        Check(
                            name="customer_line_exact",
                            rule="regex",
                            params={"pattern": r"(?im)^Customer:\s*Mina Patel\s*$"},
                        ),
                        Check(
                            name="problem_mentions_password_reset",
                            rule="regex",
                            params={"pattern": r"(?im)^Problem:.*password reset.*$"},
                        ),
                        Check(
                            name="environment_line_exact",
                            rule="regex",
                            params={"pattern": r"(?im)^Environment:\s*staging\s*$"},
                        ),
                        Check(
                            name="region_line_exact",
                            rule="regex",
                            params={"pattern": r"(?im)^Region:\s*us-west-2\s*$"},
                        ),
                        Check(
                            name="mentions_no_2fa_errors",
                            rule="regex",
                            params={"pattern": r"(?is)(no|without).*(2FA|two-factor).*(errors?)"},
                        ),
                        Check(
                            name="next_step_line_present",
                            rule="regex",
                            params={"pattern": r"(?im)^Next Step:\s*\S.+$"},
                        ),
                    ],
                ),
            ],
            tags=["formatting", "memory", "handoff"],
        ),
        Scenario(
            name="launch_brief_consistency",
            description=(
                "Tests whether the assistant preserves exact factual details across a drafting and "
                "revision cycle rather than drifting toward a looser summary."
            ),
            system_prompt=(
                "You are a product communications assistant. Keep every explicit fact stable "
                "unless the user changes it."
            ),
            facts={
                "product_name": "Northwind Copilot",
                "launch_date": "June 18",
                "rollout_group": "12 design partners",
                "launch_url": "northwind.ai/copilot",
            },
            turns=[
                ScenarioTurn(
                    user=(
                        "Use these launch facts later: product Northwind Copilot, public launch "
                        "June 18, initial rollout to 12 design partners, and URL "
                        "northwind.ai/copilot. Reply with a short acknowledgement that repeats the "
                        "product name and launch date."
                    ),
                    checks=[
                        Check(
                            name="ack_repeats_product_and_date",
                            rule="contains_all",
                            params={"needles": ["{{product_name}}", "{{launch_date}}"]},
                        ),
                    ],
                ),
                ScenarioTurn(
                    user="Draft a two-sentence announcement paragraph.",
                    checks=[
                        Check(
                            name="draft_keeps_all_launch_facts",
                            rule="contains_all",
                            params={
                                "needles": [
                                    "{{product_name}}",
                                    "{{launch_date}}",
                                    "{{rollout_group}}",
                                    "{{launch_url}}",
                                ]
                            },
                        ),
                    ],
                ),
                ScenarioTurn(
                    user="Revise the announcement to be shorter, but keep every factual detail unchanged.",
                    checks=[
                        Check(
                            name="revision_preserves_all_facts",
                            rule="contains_all",
                            params={
                                "needles": [
                                    "{{product_name}}",
                                    "{{launch_date}}",
                                    "{{rollout_group}}",
                                    "{{launch_url}}",
                                ]
                            },
                        ),
                        Check(
                            name="revision_does_not_drift",
                            rule="contains_none",
                            params={"needles": ["June 19", "13 design partners"]},
                        ),
                    ],
                ),
            ],
            tags=["revision", "consistency", "factuality"],
        ),
    ]


def _load_scenarios_from_file(path: Path) -> list[Scenario]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Scenario file must contain a top-level JSON list.")
    return [Scenario.from_dict(item) for item in payload]


def _select_scenarios(all_scenarios: list[Scenario], requested_names: list[str]) -> list[Scenario]:
    if not requested_names:
        return all_scenarios
    requested = {name.strip() for name in requested_names if name.strip()}
    by_name = {scenario.name: scenario for scenario in all_scenarios}
    missing = sorted(name for name in requested if name not in by_name)
    if missing:
        raise ValueError(f"Unknown scenario(s): {', '.join(missing)}")
    return [by_name[name] for name in requested_names if name in by_name]


def _resolve_renderer_name(base_model: str, renderer_override: str | None) -> str:
    if renderer_override:
        return renderer_override
    if str(COOKBOOK_ROOT) not in sys.path:
        sys.path.insert(0, str(COOKBOOK_ROOT))
    from tinker_cookbook.model_info import get_recommended_renderer_names

    recommended = get_recommended_renderer_names(base_model)
    for candidate in recommended:
        if "disable_thinking" in candidate:
            return candidate
    return recommended[0]


def _build_runtime(config: RuntimeConfig) -> RuntimeHandles:
    if str(COOKBOOK_ROOT) not in sys.path:
        sys.path.insert(0, str(COOKBOOK_ROOT))

    import tinker
    from tinker import types as tinker_types

    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    resolved_renderer_name = _resolve_renderer_name(config.base_model, config.renderer_name)
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(
        resolved_renderer_name,
        tokenizer,
        model_name=config.base_model,
    )
    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(
        base_model=config.base_model,
        model_path=config.model_path,
    )
    return RuntimeHandles(
        sampling_client=sampling_client,
        renderer=renderer,
        renderers_module=renderers,
        resolved_renderer_name=resolved_renderer_name,
        tinker_types=tinker_types,
    )


async def _generate_assistant_turn(
    runtime: RuntimeHandles,
    config: RuntimeConfig,
    prompt_messages: list[dict[str, str]],
) -> tuple[str, float, int | None, int, str]:
    model_input = runtime.renderer.build_generation_prompt(prompt_messages)
    prompt_tokens = getattr(model_input, "length", None)
    if prompt_tokens is None and hasattr(model_input, "to_ints"):
        prompt_tokens = len(model_input.to_ints())

    sampling_params = runtime.tinker_types.SamplingParams(
        max_tokens=config.max_tokens,
        seed=config.seed,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        stop=runtime.renderer.get_stop_sequences(),
    )

    started = time.perf_counter()
    response = await runtime.sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    sequence = response.sequences[0]
    parsed_message, parse_success = runtime.renderer.parse_response(sequence.tokens)
    if parse_success:
        assistant_text = runtime.renderers_module.get_text_content(parsed_message)
    else:
        assistant_text = runtime.renderer.tokenizer.decode(sequence.tokens, skip_special_tokens=False)

    assistant_text = _sanitize_text(assistant_text)
    return assistant_text, latency_ms, prompt_tokens, len(sequence.tokens), str(sequence.stop_reason)


async def _run_scenario(
    runtime: RuntimeHandles,
    config: RuntimeConfig,
    scenario: Scenario,
) -> ScenarioResult:
    history: list[dict[str, str]] = []
    traces: list[TurnTrace] = []

    if scenario.system_prompt.strip():
        history.append({"role": "system", "content": scenario.system_prompt.strip()})

    for index, turn in enumerate(scenario.turns, start=1):
        history.append({"role": "user", "content": turn.user})
        prompt_messages = [dict(message) for message in history]

        assistant_text, latency_ms, prompt_tokens, completion_tokens, stop_reason = (
            await _generate_assistant_turn(runtime, config, prompt_messages)
        )
        history.append({"role": "assistant", "content": assistant_text})

        check_results = [_evaluate_check(assistant_text, check, scenario.facts) for check in turn.checks]
        traces.append(
            TurnTrace(
                index=index,
                user_message=turn.user,
                prompt_messages=prompt_messages,
                assistant_message=assistant_text,
                latency_ms=round(latency_ms, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                stop_reason=stop_reason,
                checks=check_results,
            )
        )

    score = sum(check.score for trace in traces for check in trace.checks)
    max_score = sum(check.weight for turn in scenario.turns for check in turn.checks)
    passed = all(check.passed for trace in traces for check in trace.checks)
    return ScenarioResult(
        name=scenario.name,
        description=scenario.description,
        passed=passed,
        score=score,
        max_score=max_score,
        traces=traces,
    )


def _build_report(
    config: RuntimeConfig,
    runtime: RuntimeHandles,
    scenarios: list[Scenario],
    results: list[ScenarioResult],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    scenario_payloads = [scenario.to_dict() for scenario in scenarios]
    workload_fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "config": {
                    "base_model": config.base_model,
                    "model_path": config.model_path,
                    "renderer_name": runtime.resolved_renderer_name,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "seed": config.seed,
                },
                "scenarios": scenario_payloads,
            }
        ).encode("utf-8")
    ).hexdigest()

    total_score = sum(result.score for result in results)
    total_max_score = sum(result.max_score for result in results)
    total_checks = sum(len(trace.checks) for result in results for trace in result.traces)
    passed_checks = sum(
        1 for result in results for trace in result.traces for check in trace.checks if check.passed
    )
    total_latency_ms = sum(trace.latency_ms for result in results for trace in result.traces)
    total_completion_tokens = sum(
        trace.completion_tokens for result in results for trace in result.traces
    )

    return {
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "run_id": f"{finished_at.replace(':', '').replace('-', '')}_{workload_fingerprint[:8]}",
            "workload_fingerprint": workload_fingerprint,
        },
        "config": {
            **config.to_dict(),
            "resolved_renderer_name": runtime.resolved_renderer_name,
        },
        "summary": {
            "scenario_count": len(results),
            "passed_scenarios": sum(1 for result in results if result.passed),
            "failed_scenarios": sum(1 for result in results if not result.passed),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "score": total_score,
            "max_score": total_max_score,
            "score_pct": 0.0 if total_max_score == 0 else round((total_score / total_max_score) * 100, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "total_completion_tokens": total_completion_tokens,
        },
        "scenarios": scenario_payloads,
        "results": [result.to_dict() for result in results],
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    fingerprint = report["meta"]["workload_fingerprint"][:8]
    output_path = output_dir / f"conversation_eval_{timestamp}_{fingerprint}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return output_path


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    config = report["config"]
    print(f"Run ID: {report['meta']['run_id']}")
    print(f"Workload fingerprint: {report['meta']['workload_fingerprint']}")
    print(f"Model: {config['base_model']}")
    print(f"Renderer: {config['resolved_renderer_name']}")
    print(
        "Overall: "
        f"{summary['score_pct']:.2f}% "
        f"({summary['score']:.1f}/{summary['max_score']:.1f})"
    )
    print(
        "Scenarios: "
        f"{summary['passed_scenarios']} passed, "
        f"{summary['failed_scenarios']} failed, "
        f"{summary['passed_checks']}/{summary['total_checks']} checks passed"
    )

    for result in report["results"]:
        print(
            f"- {result['name']}: "
            f"{result['score_pct']:.2f}% "
            f"({result['score']:.1f}/{result['max_score']:.1f}) "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
        failures = [
            check
            for trace in result["traces"]
            for check in trace["checks"]
            if not check["passed"]
        ]
        for failure in failures:
            print(f"  failed {failure['name']}: {failure['details']}")


def _print_scenarios(scenarios: list[Scenario]) -> None:
    for scenario in scenarios:
        print(f"{scenario.name}: {scenario.description}")


def _build_rule_case(
    rule: str,
    params: dict[str, Any],
    response_text: str,
    expected: bool,
) -> tuple[Check, str, bool]:
    return (
        Check(
            name=f"{rule}_self_test",
            rule=rule,
            params=params,
        ),
        response_text,
        expected,
    )


def _run_self_test() -> int:
    cases = [
        _build_rule_case("word_count_equals", {"word": "note", "count": 2}, "note 27 note", True),
        _build_rule_case(
            "sentence_count_equals",
            {"count": 2},
            "The answer is 27. Keep it short.",
            True,
        ),
        _build_rule_case("line_count_at_least", {"count": 3}, "one\ntwo\nthree", True),
        _build_rule_case("char_count_at_least", {"count": 20}, "abcdefghijklmnopqrst", True),
        _build_rule_case(
            "section_headers_present",
            {"headers": ["Summary", "Decision"]},
            "Summary:\nShort note.\nDecision:\nApprove.",
            True,
        ),
    ]

    for check, response_text, expected in cases:
        result = _evaluate_check(response_text, check, {})
        if result.passed != expected:
            raise AssertionError(
                f"Rule self-test failed for {check.rule}: expected {expected}, got {result.passed}"
            )

    print("Self-test passed.")
    return 0


def _parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Purpose-built evaluation harness for realistic, multi-turn Tinker conversations "
            "with deterministic scoring and fully inspectable turn-by-turn traces."
        )
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--renderer", default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="eval_runs")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name to run. Repeat the flag to run multiple named scenarios.",
    )
    parser.add_argument(
        "--scenario-file",
        default=None,
        help="Optional JSON file containing scenario definitions.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print the available scenario names and exit.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    scenario_names = []
    for item in args.scenario:
        scenario_names.extend(part.strip() for part in str(item).split(",") if part.strip())

    config = RuntimeConfig(
        base_model=args.base_model,
        model_path=_clean_optional(args.model_path),
        base_url=_clean_optional(args.base_url),
        renderer_name=_clean_optional(args.renderer),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        output_dir=args.output_dir,
        scenario_names=scenario_names,
        scenario_file=_clean_optional(args.scenario_file),
        list_scenarios=args.list_scenarios,
        self_test=bool(args.self_test),
    )
    return config


def _load_selected_scenarios(config: RuntimeConfig) -> list[Scenario]:
    scenarios = (
        _load_scenarios_from_file(Path(config.scenario_file))
        if config.scenario_file
        else _build_builtin_scenarios()
    )
    return _select_scenarios(scenarios, config.scenario_names)


async def _async_main(config: RuntimeConfig) -> int:
    if config.self_test:
        return _run_self_test()

    selected_scenarios = _load_selected_scenarios(config)
    if config.list_scenarios:
        _print_scenarios(selected_scenarios)
        return 0

    runtime = _build_runtime(config)
    started_at = _now_iso()
    results = [await _run_scenario(runtime, config, scenario) for scenario in selected_scenarios]
    finished_at = _now_iso()
    report = _build_report(
        config=config,
        runtime=runtime,
        scenarios=selected_scenarios,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    output_path = _write_report(Path(config.output_dir), report)
    _print_summary(report)
    print(f"Report: {output_path}")
    return 0 if report["summary"]["failed_scenarios"] == 0 else 1


def main() -> int:
    try:
        config = _parse_args()
        return asyncio.run(_async_main(config))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
