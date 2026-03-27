#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parent
COOKBOOK_ROOT = REPO_ROOT / "tinker-cookbook"

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_TOKENS = 768
DEFAULT_OUTPUT_DIR = "workload_bundles"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_SEED = 7

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TEMPLATE_RE = re.compile(r"\{\{([a-zA-Z0-9_.]+)\}\}")
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")

SUPPORTED_RULES: list[tuple[str, str]] = [
    ("contains_all", "All required substrings must appear."),
    ("contains_none", "No forbidden substrings may appear."),
    ("contains_any", "At least one allowed substring must appear."),
    ("regex", "A regular expression must match."),
    ("regex_none", "A regular expression must not match."),
    ("ordered_substrings", "Required substrings must appear in the given order."),
    ("line_count_at_most", "The number of non-empty lines must not exceed a maximum."),
    ("bullet_count_equals", "The number of bullet lines must equal a target."),
    ("bullet_count_at_most", "The number of bullet lines must not exceed a maximum."),
    ("question_count_equals", "The number of question marks must equal a target."),
    ("exact_line_set", "The exact set of non-empty lines must match."),
    ("field_equals", "A labeled field line must match an expected value."),
    ("required_omission", "Specific values must be omitted."),
    ("state_preserved", "Required state values must still be present."),
    ("state_updated_correctly", "Updated state must appear and old state must not."),
    ("no_new_allowed_terms", "Only explicitly allowed monitored terms may appear."),
]
SUPPORTED_RULE_NAMES = {name for name, _description in SUPPORTED_RULES}

FAMILY_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "support_escalation": {
        "clarification_count": 1,
        "revision_count": 0,
        "final_artifact": "handoff_note",
        "turn_count_hint": 4,
        "assistant_phases": ("clarification_response", "final_handoff"),
        "all_phases": (
            "system_context",
            "intake_request",
            "clarification_response",
            "followup_details",
            "final_handoff",
        ),
    },
    "policy_memory": {
        "clarification_count": 0,
        "revision_count": 0,
        "final_artifact": "approved_items_summary",
        "turn_count_hint": 6,
        "assistant_phases": ("policy_ack", "expense_classification", "approved_summary"),
        "all_phases": (
            "system_context",
            "policy_ingest",
            "policy_ack",
            "expense_review",
            "expense_classification",
            "summary_request",
            "approved_summary",
        ),
    },
    "revision_brief": {
        "clarification_count": 0,
        "revision_count": 1,
        "final_artifact": "announcement_paragraph",
        "turn_count_hint": 6,
        "assistant_phases": ("launch_ack", "launch_draft", "launch_revision"),
        "all_phases": (
            "system_context",
            "launch_facts",
            "launch_ack",
            "draft_request",
            "launch_draft",
            "revision_request",
            "launch_revision",
        ),
    },
    "selective_omission": {
        "clarification_count": 0,
        "revision_count": 1,
        "final_artifact": "customer_safe_summary",
        "turn_count_hint": 6,
        "assistant_phases": ("intake_ack", "internal_recap", "external_summary"),
        "all_phases": (
            "system_context",
            "intake_request",
            "intake_ack",
            "internal_recap_request",
            "internal_recap",
            "external_summary_request",
            "external_summary",
        ),
    },
}


@dataclass(slots=True)
class InteractionPattern:
    clarification_count: int
    revision_count: int
    final_artifact: str
    pressure_tags: list[str] = field(default_factory=list)
    turn_count_hint: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarification_count": self.clarification_count,
            "revision_count": self.revision_count,
            "final_artifact": self.final_artifact,
            "pressure_tags": self.pressure_tags,
            "turn_count_hint": self.turn_count_hint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InteractionPattern:
        _validate_allowed_keys(
            payload,
            {"clarification_count", "revision_count", "final_artifact", "pressure_tags", "turn_count_hint"},
            "interaction_pattern",
        )
        return cls(
            clarification_count=int(payload.get("clarification_count", 0)),
            revision_count=int(payload.get("revision_count", 0)),
            final_artifact=str(payload.get("final_artifact", "")),
            pressure_tags=[str(item) for item in payload.get("pressure_tags", [])],
            turn_count_hint=(
                None if payload.get("turn_count_hint") is None else int(payload["turn_count_hint"])
            ),
        )


@dataclass(slots=True)
class MutableFactTemplate:
    key: str
    phase: str
    value: Any | None = None
    from_pool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"key": self.key, "phase": self.phase}
        if self.value is not None:
            payload["value"] = self.value
        if self.from_pool is not None:
            payload["from_pool"] = self.from_pool
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MutableFactTemplate:
        _validate_allowed_keys(payload, {"key", "phase", "value", "from_pool"}, "mutable_fact")
        return cls(
            key=str(payload["key"]),
            phase=str(payload["phase"]),
            value=payload.get("value"),
            from_pool=None if payload.get("from_pool") is None else str(payload["from_pool"]),
        )


@dataclass(slots=True)
class CanaryTemplate:
    key: str
    mode: str
    phase: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "mode": self.mode,
        }
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.description:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanaryTemplate:
        _validate_allowed_keys(payload, {"key", "mode", "phase", "description"}, "canary")
        return cls(
            key=str(payload["key"]),
            mode=str(payload["mode"]),
            phase=None if payload.get("phase") is None else str(payload["phase"]),
            description=str(payload.get("description", "")),
        )


@dataclass(slots=True)
class HiddenStateTemplate:
    fact_pools: dict[str, Any] = field(default_factory=dict)
    mutable_facts: list[MutableFactTemplate] = field(default_factory=list)
    distractors: dict[str, Any] = field(default_factory=dict)
    omission_keys: list[str] = field(default_factory=list)
    canaries: list[CanaryTemplate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_pools": self.fact_pools,
            "mutable_facts": [item.to_dict() for item in self.mutable_facts],
            "distractors": self.distractors,
            "omission_keys": self.omission_keys,
            "canaries": [item.to_dict() for item in self.canaries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HiddenStateTemplate:
        _validate_allowed_keys(
            payload,
            {"fact_pools", "mutable_facts", "distractors", "omission_keys", "canaries"},
            "hidden_state_template",
        )
        return cls(
            fact_pools=dict(payload.get("fact_pools", {})),
            mutable_facts=[MutableFactTemplate.from_dict(item) for item in payload.get("mutable_facts", [])],
            distractors=dict(payload.get("distractors", {})),
            omission_keys=[str(item) for item in payload.get("omission_keys", [])],
            canaries=[CanaryTemplate.from_dict(item) for item in payload.get("canaries", [])],
        )


@dataclass(slots=True)
class MaskPolicy:
    direction: str
    masked_phases: list[str] = field(default_factory=list)
    visible_anchor_assistant_steps: list[str] = field(default_factory=list)
    fill_order: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "masked_phases": self.masked_phases,
            "visible_anchor_assistant_steps": self.visible_anchor_assistant_steps,
            "fill_order": self.fill_order,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskPolicy:
        _validate_allowed_keys(
            payload,
            {"direction", "masked_phases", "visible_anchor_assistant_steps", "fill_order"},
            "mask_policy",
        )
        return cls(
            direction=str(payload.get("direction", "assistant_only")),
            masked_phases=[str(item) for item in payload.get("masked_phases", [])],
            visible_anchor_assistant_steps=[
                str(item) for item in payload.get("visible_anchor_assistant_steps", [])
            ],
            fill_order=str(payload.get("fill_order", "sequential")),
        )


@dataclass(slots=True)
class AuthoredSpec:
    name: str
    description: str
    family: str
    tags: list[str]
    system_prompt: str
    interaction_pattern: InteractionPattern
    hidden_state_template: HiddenStateTemplate
    mask_policy: MaskPolicy
    artifact_contract: dict[str, Any]
    style_options: dict[str, Any]
    instance_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "tags": self.tags,
            "system_prompt": self.system_prompt,
            "interaction_pattern": self.interaction_pattern.to_dict(),
            "hidden_state_template": self.hidden_state_template.to_dict(),
            "mask_policy": self.mask_policy.to_dict(),
            "artifact_contract": self.artifact_contract,
            "style_options": self.style_options,
        }
        if self.instance_count is not None:
            payload["instance_count"] = self.instance_count
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuthoredSpec:
        _validate_allowed_keys(
            payload,
            {
                "name",
                "description",
                "family",
                "tags",
                "system_prompt",
                "interaction_pattern",
                "hidden_state_template",
                "mask_policy",
                "artifact_contract",
                "style_options",
                "instance_count",
            },
            "authored_spec",
        )
        family = str(payload["family"])
        if family not in FAMILY_EXPECTATIONS:
            raise ValueError(f"Unknown family: {family}")
        merged = _deep_merge_dicts(_build_builtin_spec(family).to_dict(), payload)
        return cls(
            name=str(merged["name"]),
            description=str(merged.get("description", "")),
            family=str(merged["family"]),
            tags=[str(item) for item in merged.get("tags", [])],
            system_prompt=str(merged.get("system_prompt", "")),
            interaction_pattern=InteractionPattern.from_dict(dict(merged.get("interaction_pattern", {}))),
            hidden_state_template=HiddenStateTemplate.from_dict(
                dict(merged.get("hidden_state_template", {}))
            ),
            mask_policy=MaskPolicy.from_dict(dict(merged.get("mask_policy", {}))),
            artifact_contract=dict(merged.get("artifact_contract", {})),
            style_options=dict(merged.get("style_options", {})),
            instance_count=(
                None if merged.get("instance_count") is None else int(merged["instance_count"])
            ),
        )


@dataclass(slots=True)
class CheckBlueprint:
    name: str
    rule: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    description: str = ""


@dataclass(slots=True)
class StepBlueprint:
    step_id: str
    phase: str
    role: str
    content: Any | None = None
    checks: list[CheckBlueprint] = field(default_factory=list)


@dataclass(slots=True)
class CompiledCheck:
    step_id: str
    phase: str
    name: str
    rule: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step_id": self.step_id,
            "phase": self.phase,
            "name": self.name,
            "rule": self.rule,
            "weight": self.weight,
            "params": self.params,
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(slots=True)
class CompiledStep:
    step_id: str
    phase: str
    role: str
    source: str
    masked: bool
    content: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "phase": self.phase,
            "role": self.role,
            "source": self.source,
            "masked": self.masked,
            "content": self.content,
        }


@dataclass(slots=True)
class CompiledInstance:
    instance_id: str
    family: str
    seed: int
    spec_fingerprint: str
    instance_fingerprint: str
    interaction_pattern: dict[str, Any]
    hidden_state: dict[str, Any]
    mask_policy: dict[str, Any]
    steps: list[CompiledStep]
    checks: list[CompiledCheck]
    workload_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "family": self.family,
            "seed": self.seed,
            "spec_fingerprint": self.spec_fingerprint,
            "instance_fingerprint": self.instance_fingerprint,
            "interaction_pattern": self.interaction_pattern,
            "hidden_state": self.hidden_state,
            "mask_policy": self.mask_policy,
            "steps": [step.to_dict() for step in self.steps],
            "checks": [check.to_dict() for check in self.checks],
            "workload_metadata": self.workload_metadata,
        }


@dataclass(slots=True)
class CheckResult:
    name: str
    rule: str
    passed: bool
    weight: float
    score: float
    details: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule": self.rule,
            "passed": self.passed,
            "weight": self.weight,
            "score": self.score,
            "details": self.details,
            "params": self.params,
        }


@dataclass(slots=True)
class MaskedStepResult:
    step_id: str
    prompt_messages: list[dict[str, str]]
    assistant_message: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int
    stop_reason: str
    checks: list[CheckResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "prompt_messages": self.prompt_messages,
            "assistant_message": self.assistant_message,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stop_reason": self.stop_reason,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class InstanceRunResult:
    instance_id: str
    masked_step_results: list[MaskedStepResult]
    instance_score: float
    instance_max_score: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "masked_step_results": [item.to_dict() for item in self.masked_step_results],
            "instance_score": self.instance_score,
            "instance_max_score": self.instance_max_score,
            "instance_score_pct": (
                0.0
                if self.instance_max_score == 0
                else round((self.instance_score / self.instance_max_score) * 100, 2)
            ),
            "passed": self.passed,
        }


@dataclass(slots=True)
class CLIConfig:
    base_model: str
    model_path: str | None
    base_url: str | None
    renderer_name: str | None
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    output_dir: str
    pattern_names: list[str]
    spec_file: str | None
    instance_count: int
    compile_only: bool
    list_patterns: bool
    list_rules: bool
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
            "pattern_names": self.pattern_names,
            "spec_file": self.spec_file,
            "instance_count": self.instance_count,
            "compile_only": self.compile_only,
            "list_patterns": self.list_patterns,
            "list_rules": self.list_rules,
            "self_test": self.self_test,
        }


@dataclass(slots=True)
class RuntimeHandles:
    sampling_client: Any
    renderer: Any
    renderers_module: Any
    resolved_renderer_name: str
    tinker_types: Any


def _validate_allowed_keys(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {', '.join(unknown)}")


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sanitize_text(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _bullet_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if BULLET_RE.match(line)]


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


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _ensure_nonempty_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list.")
    return value


def _resolve_pool_value(value: Any, rng: random.Random) -> Any:
    if isinstance(value, list):
        chosen = rng.choice(_ensure_nonempty_list(value, "fact pool"))
        return _resolve_pool_value(chosen, rng)
    if isinstance(value, dict):
        return {key: _resolve_pool_value(item, rng) for key, item in value.items()}
    return copy.deepcopy(value)


def _lookup_path(context: dict[str, Any], path: str, location: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Unknown template variable '{path}' in {location}")
        current = current[part]
    return current


def _stringify_template_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_stringify_template_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _resolve_templates(value: Any, context: dict[str, Any], location: str) -> Any:
    if isinstance(value, str):
        return TEMPLATE_RE.sub(
            lambda match: _stringify_template_value(
                _lookup_path(context, match.group(1), location)
            ),
            value,
        )
    if isinstance(value, list):
        return [_resolve_templates(item, context, location) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_templates(item, context, f"{location}.{key}")
            for key, item in value.items()
        }
    return value


def _resolve_variant(value: Any, context: dict[str, Any], rng: random.Random, location: str) -> Any:
    if isinstance(value, list):
        chosen = rng.choice(_ensure_nonempty_list(value, location))
        return _resolve_variant(chosen, context, rng, location)
    return _resolve_templates(value, context, location)


def _build_builtin_spec(family: str) -> AuthoredSpec:
    if family == "support_escalation":
        return AuthoredSpec(
            name="support_escalation",
            description=(
                "A support escalation workload with one clarifying question and a strict handoff."
            ),
            family=family,
            tags=["support", "clarification", "handoff", "no-drift"],
            system_prompt=(
                "You are a support operations assistant. Follow formatting instructions exactly, "
                "retain incident details across turns, and do not invent missing evidence."
            ),
            interaction_pattern=InteractionPattern(
                clarification_count=1,
                revision_count=0,
                final_artifact="handoff_note",
                pressure_tags=["clarification", "formatting", "state_preserved", "no_drift"],
                turn_count_hint=4,
            ),
            hidden_state_template=HiddenStateTemplate(
                fact_pools={
                    "ticket_id": ["SUP-7714", "SUP-9132", "SUP-4805"],
                    "customer_name": ["Mina Patel", "Jordan Alvarez", "Riley Chen"],
                    "trigger": ["password reset", "email change", "SSO reconnect"],
                    "environment": ["staging", "sandbox", "preprod"],
                    "region": ["us-west-2", "eu-west-1", "ap-southeast-1"],
                    "absent_issue": ["2FA errors", "billing failures", "email bounces"],
                    "next_step": [
                        "Investigate the authentication flow in the affected environment.",
                        "Audit the reset flow and session state in the affected environment.",
                        "Trace the sign-in path in the affected environment.",
                    ],
                },
                distractors={
                    "wrong_region": ["us-east-1", "eu-central-1", "ap-northeast-1"],
                },
                canaries=[
                    CanaryTemplate(key="ticket_id", mode="recall"),
                    CanaryTemplate(key="customer_name", mode="recall"),
                    CanaryTemplate(key="region", mode="recall"),
                ],
            ),
            mask_policy=MaskPolicy(
                direction="assistant_only",
                masked_phases=["clarification_response", "final_handoff"],
                fill_order="sequential",
            ),
            artifact_contract={
                "required_fields": [
                    "Ticket",
                    "Customer",
                    "Problem",
                    "Environment",
                    "Region",
                    "Next Step",
                ]
            },
            style_options={
                "intake_variants": [
                    "I need a support handoff note later. Facts to remember: customer {{facts.customer_name}}, ticket {{facts.ticket_id}}, issue started right after a {{facts.trigger}}, and it only reproduces in {{facts.environment}}. Before drafting anything, ask me exactly one clarifying question.",
                    "Please hold these details for a support handoff: customer {{facts.customer_name}}, ticket {{facts.ticket_id}}, issue began right after a {{facts.trigger}}, and it only repros in {{facts.environment}}. Before you draft anything, ask exactly one clarifying question.",
                ],
            },
        )
    if family == "policy_memory":
        return AuthoredSpec(
            name="policy_memory",
            description=(
                "A reimbursement policy workload that stresses carry-forward memory and selective omission."
            ),
            family=family,
            tags=["policy", "memory", "classification", "selective-omission"],
            system_prompt=(
                "You are an internal operations assistant. Stay grounded in the conversation, "
                "answer directly, and do not invent policy exceptions."
            ),
            interaction_pattern=InteractionPattern(
                clarification_count=0,
                revision_count=0,
                final_artifact="approved_items_summary",
                pressure_tags=["memory", "selective_omission", "carry_forward"],
                turn_count_hint=6,
            ),
            hidden_state_template=HiddenStateTemplate(
                fact_pools={
                    "trip_code": ["TRIP-4821", "TRIP-5907", "TRIP-6114"],
                    "hotel_cap": ["$180", "$200", "$220"],
                    "meal_cap": ["$65", "$70", "$75"],
                    "hotel_expense": ["$148", "$172", "$179"],
                    "rideshare_expense": ["$24", "$31", "$42"],
                    "minibar_expense": ["$17", "$19", "$23"],
                },
                omission_keys=["minibar_expense"],
                canaries=[
                    CanaryTemplate(key="trip_code", mode="recall"),
                    CanaryTemplate(key="hotel_cap", mode="recall"),
                    CanaryTemplate(key="meal_cap", mode="recall"),
                    CanaryTemplate(key="minibar_expense", mode="omit"),
                ],
            ),
            mask_policy=MaskPolicy(
                direction="assistant_only",
                masked_phases=["policy_ack", "expense_classification", "approved_summary"],
                fill_order="sequential",
            ),
            artifact_contract={"required_fields": ["Trip Code", "Approved Items"]},
            style_options={
                "policy_variants": [
                    "I'm going to give you a reimbursement policy to use later. Hotel: up to {{facts.hotel_cap}}/night with receipt. Meals: up to {{facts.meal_cap}}/day. Airport rideshare is reimbursable. Minibar or alcohol is never reimbursable. My trip code is {{facts.trip_code}}. Reply with one sentence confirming the trip code and the hotel and meal caps.",
                    "Use this reimbursement policy for later turns. Hotel: max {{facts.hotel_cap}} per night with a receipt. Meals: max {{facts.meal_cap}} per day. Airport rideshare is reimbursable. Minibar or alcohol is never reimbursable. The trip code is {{facts.trip_code}}. Reply with one sentence confirming the trip code and both caps.",
                ],
            },
        )
    if family == "revision_brief":
        return AuthoredSpec(
            name="revision_brief",
            description=(
                "A drafting-and-revision workload that requires fact preservation across a shorter rewrite."
            ),
            family=family,
            tags=["revision", "consistency", "factuality"],
            system_prompt=(
                "You are a product communications assistant. Keep every explicit fact stable "
                "unless the user changes it."
            ),
            interaction_pattern=InteractionPattern(
                clarification_count=0,
                revision_count=1,
                final_artifact="announcement_paragraph",
                pressure_tags=["revision", "state_preserved", "no_drift"],
                turn_count_hint=6,
            ),
            hidden_state_template=HiddenStateTemplate(
                fact_pools={
                    "product_name": ["Northwind Copilot", "Beacon Assist", "Harbor Notes"],
                    "launch_date": ["June 18", "July 9", "August 2"],
                    "rollout_group": ["12 design partners", "15 pilot customers", "18 beta teams"],
                    "launch_url": [
                        "northwind.ai/copilot",
                        "beacon.ai/assist",
                        "harbor.ai/notes",
                    ],
                },
                distractors={
                    "wrong_launch_date": ["June 19", "July 10", "August 3"],
                    "wrong_rollout_group": [
                        "13 design partners",
                        "16 pilot customers",
                        "19 beta teams",
                    ],
                    "wrong_url": [
                        "northwind.com/copilot",
                        "beacon.ai/copilot",
                        "harbor.ai/write",
                    ],
                },
                canaries=[
                    CanaryTemplate(key="product_name", mode="recall"),
                    CanaryTemplate(key="launch_date", mode="recall"),
                    CanaryTemplate(key="launch_url", mode="recall"),
                ],
            ),
            mask_policy=MaskPolicy(
                direction="assistant_only",
                masked_phases=["launch_ack", "launch_draft", "launch_revision"],
                fill_order="sequential",
            ),
            artifact_contract={"required_fields": ["Product", "Launch Date", "Rollout", "URL"]},
            style_options={
                "fact_variants": [
                    "Use these launch facts later: product {{facts.product_name}}, public launch {{facts.launch_date}}, initial rollout to {{facts.rollout_group}}, and URL {{facts.launch_url}}. Reply with a short acknowledgement that repeats the product name and launch date.",
                    "Hold these launch facts for later turns: product {{facts.product_name}}, public launch on {{facts.launch_date}}, first rollout to {{facts.rollout_group}}, and URL {{facts.launch_url}}. Reply with a short acknowledgement that repeats the product name and date.",
                ],
            },
        )
    if family == "selective_omission":
        return AuthoredSpec(
            name="selective_omission",
            description=(
                "A workload that updates hidden state mid-conversation and then requires a customer-safe omission."
            ),
            family=family,
            tags=["omission", "state-update", "customer-safe", "memory"],
            system_prompt=(
                "You are an operations assistant. Preserve exact case details, update state when the user changes it, "
                "and omit internal-only details when asked."
            ),
            interaction_pattern=InteractionPattern(
                clarification_count=0,
                revision_count=1,
                final_artifact="customer_safe_summary",
                pressure_tags=["state_update", "required_omission", "carry_forward"],
                turn_count_hint=6,
            ),
            hidden_state_template=HiddenStateTemplate(
                fact_pools={
                    "customer_name": ["Avery Brooks", "Samira Khan", "Diego Flores"],
                    "case_ref": ["CASE-4412", "CASE-5830", "CASE-6201"],
                    "issue_summary": [
                        "dashboard access failed after a session timeout",
                        "invoice export failed after a plan change",
                        "team invites stopped sending after a role update",
                    ],
                    "status": ["investigating", "pending review", "awaiting mitigation"],
                    "root_cause_code": ["RC-17A", "RC-28C", "RC-33F"],
                },
                mutable_facts=[
                    MutableFactTemplate(
                        key="status",
                        phase="internal_recap_request",
                        value=["resolved", "fixed", "fully mitigated"],
                    )
                ],
                omission_keys=["root_cause_code", "customer_name"],
                canaries=[
                    CanaryTemplate(key="case_ref", mode="recall"),
                    CanaryTemplate(key="status", mode="update", phase="internal_recap_request"),
                    CanaryTemplate(key="root_cause_code", mode="omit", phase="external_summary"),
                ],
            ),
            mask_policy=MaskPolicy(
                direction="assistant_only",
                masked_phases=["intake_ack", "internal_recap", "external_summary"],
                fill_order="sequential",
            ),
            artifact_contract={"required_fields": ["Case Reference", "Current Status"]},
            style_options={
                "intake_variants": [
                    "Please remember these case details for later: customer {{facts.customer_name}}, case {{facts.case_ref}}, issue {{facts.issue_summary}}, current status {{facts.status}}, and internal root cause code {{facts.root_cause_code}}. Reply with one sentence confirming the case and status.",
                    "Hold these case details for later turns: customer {{facts.customer_name}}, case {{facts.case_ref}}, issue {{facts.issue_summary}}, current status {{facts.status}}, and internal root cause code {{facts.root_cause_code}}. Reply with one sentence confirming the case reference and current status.",
                ],
            },
        )
    raise ValueError(f"Unknown family: {family}")


def _list_builtin_patterns() -> list[AuthoredSpec]:
    return [_build_builtin_spec(name) for name in FAMILY_EXPECTATIONS]


def _render_required_fields(spec: AuthoredSpec) -> str:
    required_fields = [str(item) for item in spec.artifact_contract.get("required_fields", [])]
    return ", ".join(required_fields)


def _support_escalation_blueprints(spec: AuthoredSpec, rng: random.Random) -> list[StepBlueprint]:
    labels = _render_required_fields(spec)
    return [
        StepBlueprint(
            step_id="system_context",
            phase="system_context",
            role="system",
            content=spec.system_prompt,
        ),
        StepBlueprint(
            step_id="intake_request",
            phase="intake_request",
            role="user",
            content=spec.style_options["intake_variants"],
        ),
        StepBlueprint(
            step_id="clarification_response",
            phase="clarification_response",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="asks_one_clarifying_question",
                    rule="question_count_equals",
                    params={"count": 1},
                ),
                CheckBlueprint(
                    name="clarification_is_compact",
                    rule="line_count_at_most",
                    params={"max": 2},
                ),
            ],
        ),
        StepBlueprint(
            step_id="followup_details",
            phase="followup_details",
            role="user",
            content=[
                "The affected region is {{facts.region}} and there were no {{facts.absent_issue}}. Now draft the handoff note using exactly these labels on separate lines: "
                + labels
                + ". Use the values exactly as given.",
                "The affected region is {{facts.region}} and there were no {{facts.absent_issue}}. Draft the handoff note using exactly these labels on separate lines: "
                + labels
                + ". Keep the field values exactly as given.",
            ],
        ),
        StepBlueprint(
            step_id="final_handoff",
            phase="final_handoff",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="ticket_field_exact",
                    rule="field_equals",
                    params={"field": "Ticket", "value": "{{facts.ticket_id}}"},
                ),
                CheckBlueprint(
                    name="customer_field_exact",
                    rule="field_equals",
                    params={"field": "Customer", "value": "{{facts.customer_name}}"},
                ),
                CheckBlueprint(
                    name="environment_field_exact",
                    rule="field_equals",
                    params={
                        "field": "Environment",
                        "value": "{{facts.environment}}",
                        "case_sensitive": False,
                    },
                ),
                CheckBlueprint(
                    name="region_field_exact",
                    rule="field_equals",
                    params={"field": "Region", "value": "{{facts.region}}"},
                ),
                CheckBlueprint(
                    name="problem_mentions_trigger",
                    rule="regex",
                    params={"pattern": r"(?im)^Problem:.*{{facts.trigger}}.*$"},
                ),
                CheckBlueprint(
                    name="mentions_absent_issue_correctly",
                    rule="regex",
                    params={"pattern": r"(?is)(no|without).*(?:{{facts.absent_issue}})"},
                ),
                CheckBlueprint(
                    name="next_step_present",
                    rule="regex",
                    params={"pattern": r"(?im)^Next Step:\s*\S.+$"},
                ),
                CheckBlueprint(
                    name="key_state_preserved",
                    rule="state_preserved",
                    params={
                        "expected_values": [
                            "{{facts.ticket_id}}",
                            "{{facts.customer_name}}",
                            "{{facts.environment}}",
                            "{{facts.region}}",
                        ]
                    },
                ),
            ],
        ),
    ]


def _policy_memory_blueprints(spec: AuthoredSpec, rng: random.Random) -> list[StepBlueprint]:
    del rng
    return [
        StepBlueprint(
            step_id="system_context",
            phase="system_context",
            role="system",
            content=spec.system_prompt,
        ),
        StepBlueprint(
            step_id="policy_ingest",
            phase="policy_ingest",
            role="user",
            content=spec.style_options["policy_variants"],
        ),
        StepBlueprint(
            step_id="policy_ack",
            phase="policy_ack",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="acknowledges_key_policy_facts",
                    rule="contains_all",
                    params={
                        "needles": [
                            "{{facts.trip_code}}",
                            "{{facts.hotel_cap}}",
                            "{{facts.meal_cap}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="policy_state_preserved",
                    rule="state_preserved",
                    params={
                        "expected_values": [
                            "{{facts.trip_code}}",
                            "{{facts.hotel_cap}}",
                            "{{facts.meal_cap}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="ack_is_compact",
                    rule="line_count_at_most",
                    params={"max": 1},
                ),
            ],
        ),
        StepBlueprint(
            step_id="expense_review",
            phase="expense_review",
            role="user",
            content=(
                "Now evaluate these expenses and answer in exactly three bullets: "
                "{{facts.hotel_expense}} hotel, {{facts.rideshare_expense}} airport rideshare, "
                "{{facts.minibar_expense}} minibar."
            ),
        ),
        StepBlueprint(
            step_id="expense_classification",
            phase="expense_classification",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="uses_three_bullets",
                    rule="bullet_count_equals",
                    params={"count": 3},
                ),
                CheckBlueprint(
                    name="hotel_is_approved",
                    rule="regex",
                    params={"pattern": r"(?is)(hotel).*(reimburs|covered|eligible|approved)"},
                ),
                CheckBlueprint(
                    name="rideshare_is_approved",
                    rule="regex",
                    params={"pattern": r"(?is)(rideshare).*(reimburs|covered|eligible|approved)"},
                ),
                CheckBlueprint(
                    name="minibar_is_rejected",
                    rule="regex",
                    params={
                        "pattern": r"(?is)(minibar).*((not|never|isn't|is not).*(reimburs|covered|eligible|approved)|excluded|ineligible)"
                    },
                ),
            ],
        ),
        StepBlueprint(
            step_id="summary_request",
            phase="summary_request",
            role="user",
            content=(
                "Give me a one-sentence summary of only the approved items and include the trip code exactly."
            ),
        ),
        StepBlueprint(
            step_id="approved_summary",
            phase="approved_summary",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="summary_keeps_trip_code",
                    rule="contains_all",
                    params={"needles": ["{{facts.trip_code}}"]},
                ),
                CheckBlueprint(
                    name="summary_mentions_both_approved_items",
                    rule="contains_all",
                    params={
                        "needles": [
                            "{{facts.hotel_expense}}",
                            "{{facts.rideshare_expense}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="summary_omits_rejected_item",
                    rule="required_omission",
                    params={"needles": ["{{facts.minibar_expense}}", "minibar"]},
                ),
                CheckBlueprint(
                    name="summary_is_compact",
                    rule="line_count_at_most",
                    params={"max": 1},
                ),
            ],
        ),
    ]


def _revision_brief_blueprints(spec: AuthoredSpec, rng: random.Random) -> list[StepBlueprint]:
    del rng
    return [
        StepBlueprint(
            step_id="system_context",
            phase="system_context",
            role="system",
            content=spec.system_prompt,
        ),
        StepBlueprint(
            step_id="launch_facts",
            phase="launch_facts",
            role="user",
            content=spec.style_options["fact_variants"],
        ),
        StepBlueprint(
            step_id="launch_ack",
            phase="launch_ack",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="ack_repeats_product_and_date",
                    rule="contains_all",
                    params={
                        "needles": ["{{facts.product_name}}", "{{facts.launch_date}}"]
                    },
                ),
                CheckBlueprint(
                    name="ack_is_compact",
                    rule="line_count_at_most",
                    params={"max": 1},
                ),
            ],
        ),
        StepBlueprint(
            step_id="draft_request",
            phase="draft_request",
            role="user",
            content="Draft a two-sentence announcement paragraph.",
        ),
        StepBlueprint(
            step_id="launch_draft",
            phase="launch_draft",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="draft_preserves_facts",
                    rule="state_preserved",
                    params={
                        "expected_values": [
                            "{{facts.product_name}}",
                            "{{facts.launch_date}}",
                            "{{facts.rollout_group}}",
                            "{{facts.launch_url}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="draft_is_short",
                    rule="line_count_at_most",
                    params={"max": 2},
                ),
            ],
        ),
        StepBlueprint(
            step_id="revision_request",
            phase="revision_request",
            role="user",
            content="Revise the announcement to be shorter, but keep every factual detail unchanged.",
        ),
        StepBlueprint(
            step_id="launch_revision",
            phase="launch_revision",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="revision_preserves_facts",
                    rule="state_preserved",
                    params={
                        "expected_values": [
                            "{{facts.product_name}}",
                            "{{facts.launch_date}}",
                            "{{facts.rollout_group}}",
                            "{{facts.launch_url}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="revision_avoids_known_drift",
                    rule="contains_none",
                    params={
                        "needles": [
                            "{{distractors.wrong_launch_date}}",
                            "{{distractors.wrong_rollout_group}}",
                            "{{distractors.wrong_url}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="revision_is_short",
                    rule="line_count_at_most",
                    params={"max": 2},
                ),
            ],
        ),
    ]


def _selective_omission_blueprints(spec: AuthoredSpec, rng: random.Random) -> list[StepBlueprint]:
    del rng
    return [
        StepBlueprint(
            step_id="system_context",
            phase="system_context",
            role="system",
            content=spec.system_prompt,
        ),
        StepBlueprint(
            step_id="intake_request",
            phase="intake_request",
            role="user",
            content=spec.style_options["intake_variants"],
        ),
        StepBlueprint(
            step_id="intake_ack",
            phase="intake_ack",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="ack_preserves_case_and_status",
                    rule="state_preserved",
                    params={
                        "expected_values": ["{{facts.case_ref}}", "{{facts.status}}"]
                    },
                ),
                CheckBlueprint(
                    name="ack_is_compact",
                    rule="line_count_at_most",
                    params={"max": 1},
                ),
            ],
        ),
        StepBlueprint(
            step_id="internal_recap_request",
            phase="internal_recap_request",
            role="user",
            content=(
                "Update: the current status is now {{facts_current.status}}. Draft an internal recap in exactly three bullets. "
                "Include the root cause code and mention the customer name."
            ),
        ),
        StepBlueprint(
            step_id="internal_recap",
            phase="internal_recap",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="internal_recap_uses_three_bullets",
                    rule="bullet_count_equals",
                    params={"count": 3},
                ),
                CheckBlueprint(
                    name="status_updates_correctly",
                    rule="state_updated_correctly",
                    params={
                        "current_values": ["{{facts_current.status}}"],
                        "forbidden_values": ["{{facts.status}}"],
                    },
                ),
                CheckBlueprint(
                    name="internal_recap_keeps_internal_details",
                    rule="contains_all",
                    params={
                        "needles": [
                            "{{facts.root_cause_code}}",
                            "{{facts.customer_name}}",
                        ]
                    },
                ),
            ],
        ),
        StepBlueprint(
            step_id="external_summary_request",
            phase="external_summary_request",
            role="user",
            content=(
                "Now write a customer-safe summary in one sentence. Include the case reference and current status, "
                "but do not mention the internal root cause code or the customer name."
            ),
        ),
        StepBlueprint(
            step_id="external_summary",
            phase="external_summary",
            role="assistant",
            checks=[
                CheckBlueprint(
                    name="external_summary_keeps_case_and_current_status",
                    rule="contains_all",
                    params={
                        "needles": [
                            "{{facts.case_ref}}",
                            "{{facts_current.status}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="external_summary_omits_internal_details",
                    rule="required_omission",
                    params={
                        "needles": [
                            "{{facts.root_cause_code}}",
                            "{{facts.customer_name}}",
                        ]
                    },
                ),
                CheckBlueprint(
                    name="external_summary_uses_updated_state",
                    rule="state_updated_correctly",
                    params={
                        "current_values": ["{{facts_current.status}}"],
                        "forbidden_values": ["{{facts.status}}"],
                    },
                ),
                CheckBlueprint(
                    name="external_summary_is_compact",
                    rule="line_count_at_most",
                    params={"max": 1},
                ),
            ],
        ),
    ]


FAMILY_BLUEPRINT_BUILDERS: dict[str, Callable[[AuthoredSpec, random.Random], list[StepBlueprint]]] = {
    "support_escalation": _support_escalation_blueprints,
    "policy_memory": _policy_memory_blueprints,
    "revision_brief": _revision_brief_blueprints,
    "selective_omission": _selective_omission_blueprints,
}


def _validate_spec(spec: AuthoredSpec) -> None:
    if spec.family not in FAMILY_EXPECTATIONS:
        raise ValueError(f"Unknown family: {spec.family}")
    expectations = FAMILY_EXPECTATIONS[spec.family]
    if spec.interaction_pattern.clarification_count != expectations["clarification_count"]:
        raise ValueError(
            f"{spec.family} requires clarification_count={expectations['clarification_count']}"
        )
    if spec.interaction_pattern.revision_count != expectations["revision_count"]:
        raise ValueError(
            f"{spec.family} requires revision_count={expectations['revision_count']}"
        )
    if spec.interaction_pattern.final_artifact != expectations["final_artifact"]:
        raise ValueError(
            f"{spec.family} requires final_artifact={expectations['final_artifact']!r}"
        )
    if (
        spec.interaction_pattern.turn_count_hint is not None
        and spec.interaction_pattern.turn_count_hint != expectations["turn_count_hint"]
    ):
        raise ValueError(
            f"{spec.family} requires turn_count_hint={expectations['turn_count_hint']}"
        )

    if spec.mask_policy.direction != "assistant_only":
        raise ValueError("mask_policy.direction must be 'assistant_only' in v1.")
    if spec.mask_policy.fill_order != "sequential":
        raise ValueError("mask_policy.fill_order must be 'sequential' in v1.")

    valid_phases = set(expectations["all_phases"])
    assistant_phases = set(expectations["assistant_phases"])
    unknown_masked = sorted(set(spec.mask_policy.masked_phases) - valid_phases)
    if unknown_masked:
        raise ValueError(
            f"Unknown masked phases for {spec.family}: {', '.join(unknown_masked)}"
        )
    wrong_mask_target = sorted(set(spec.mask_policy.masked_phases) - assistant_phases)
    if wrong_mask_target:
        raise ValueError(
            f"Masked phases must be assistant phases for {spec.family}: {', '.join(wrong_mask_target)}"
        )
    unknown_anchor_steps = sorted(
        set(spec.mask_policy.visible_anchor_assistant_steps) - valid_phases
    )
    if unknown_anchor_steps:
        raise ValueError(
            f"Unknown visible assistant anchor phases for {spec.family}: {', '.join(unknown_anchor_steps)}"
        )
    wrong_anchor_target = sorted(
        set(spec.mask_policy.visible_anchor_assistant_steps) - assistant_phases
    )
    if wrong_anchor_target:
        raise ValueError(
            "visible_anchor_assistant_steps must reference assistant phases only: "
            + ", ".join(wrong_anchor_target)
        )

    fact_pools = spec.hidden_state_template.fact_pools
    if not fact_pools:
        raise ValueError(f"{spec.name} must define hidden_state_template.fact_pools.")

    for mutable in spec.hidden_state_template.mutable_facts:
        if mutable.key not in fact_pools:
            raise ValueError(
                f"mutable_facts entry references unknown fact key '{mutable.key}'."
            )
        if mutable.phase not in valid_phases:
            raise ValueError(
                f"mutable_facts entry for '{mutable.key}' references unknown phase '{mutable.phase}'."
            )
        if mutable.value is None and mutable.from_pool is None:
            raise ValueError(
                f"mutable_facts entry for '{mutable.key}' must set value or from_pool."
            )
        if mutable.from_pool is not None and mutable.from_pool not in fact_pools:
            raise ValueError(
                f"mutable_facts entry for '{mutable.key}' references unknown from_pool '{mutable.from_pool}'."
            )

    for omission_key in spec.hidden_state_template.omission_keys:
        if omission_key not in fact_pools:
            raise ValueError(f"omission_keys references unknown fact key '{omission_key}'.")

    for canary in spec.hidden_state_template.canaries:
        if canary.key not in fact_pools:
            raise ValueError(f"canary references unknown fact key '{canary.key}'.")
        if canary.mode not in {"recall", "omit", "update"}:
            raise ValueError(
                f"canary '{canary.key}' uses unsupported mode '{canary.mode}'."
            )
        if canary.phase is not None and canary.phase not in valid_phases:
            raise ValueError(
                f"canary '{canary.key}' references unknown phase '{canary.phase}'."
            )


def _choose_distinct_pool_value(pool: Any, current_value: Any, rng: random.Random) -> Any:
    if not isinstance(pool, list):
        return _resolve_pool_value(pool, rng)
    candidates = [_resolve_pool_value(item, rng) for item in pool]
    for candidate in candidates:
        if candidate != current_value:
            return candidate
    return candidates[0]


def _resolve_hidden_state(
    spec: AuthoredSpec,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = {
        key: _resolve_pool_value(value, rng)
        for key, value in spec.hidden_state_template.fact_pools.items()
    }
    facts_current = copy.deepcopy(facts)
    distractors = {
        key: _resolve_pool_value(value, rng)
        for key, value in spec.hidden_state_template.distractors.items()
    }

    mutable: dict[str, dict[str, Any]] = {}
    for item in spec.hidden_state_template.mutable_facts:
        if item.value is not None:
            updated_value = _resolve_pool_value(item.value, rng)
        elif item.from_pool is not None:
            updated_value = _choose_distinct_pool_value(
                spec.hidden_state_template.fact_pools[item.from_pool],
                facts[item.key],
                rng,
            )
        else:
            raise ValueError(
                f"mutable fact '{item.key}' must define value or from_pool."
            )
        mutable[item.key] = {
            "phase": item.phase,
            "initial": facts[item.key],
            "current": updated_value,
        }
        facts_current[item.key] = updated_value

    omission_values = [
        facts_current.get(key, facts.get(key)) for key in spec.hidden_state_template.omission_keys
    ]

    canaries_by_key: dict[str, dict[str, Any]] = {}
    for item in spec.hidden_state_template.canaries:
        canaries_by_key[item.key] = {
            "key": item.key,
            "mode": item.mode,
            "phase": item.phase,
            "description": item.description,
            "initial_value": facts[item.key],
            "current_value": facts_current.get(item.key, facts[item.key]),
        }

    hidden_state = {
        "facts": facts,
        "facts_current": facts_current,
        "distractors": distractors,
        "mutable": mutable,
        "omission_keys": list(spec.hidden_state_template.omission_keys),
        "omission_values": omission_values,
        "canaries": canaries_by_key,
    }
    context = {
        "facts": facts,
        "facts_current": facts_current,
        "distractors": distractors,
        "mutable": mutable,
        "omissions": {
            "keys": list(spec.hidden_state_template.omission_keys),
            "values": omission_values,
        },
        "canaries": canaries_by_key,
        "artifact": spec.artifact_contract,
        "interaction": spec.interaction_pattern.to_dict(),
        "style": spec.style_options,
    }
    return hidden_state, context


def _build_blueprints(spec: AuthoredSpec, rng: random.Random) -> list[StepBlueprint]:
    builder = FAMILY_BLUEPRINT_BUILDERS[spec.family]
    return builder(spec, rng)


def _compile_spec_instance(
    spec: AuthoredSpec,
    spec_fingerprint: str,
    instance_seed: int,
    instance_index: int,
) -> CompiledInstance:
    rng = random.Random(instance_seed)
    hidden_state, template_context = _resolve_hidden_state(spec, rng)
    blueprints = _build_blueprints(spec, rng)
    assistant_phase_to_blueprint = {
        item.phase: item for item in blueprints if item.role == "assistant"
    }
    for phase in spec.mask_policy.visible_anchor_assistant_steps:
        if phase not in assistant_phase_to_blueprint:
            raise ValueError(
                f"visible assistant anchor phase '{phase}' does not exist for {spec.family}."
            )

    steps: list[CompiledStep] = []
    checks: list[CompiledCheck] = []
    for blueprint in blueprints:
        location = f"{spec.name}.{blueprint.phase}.content"
        content: str | None = None
        masked = False
        source = "authored_anchor"

        if blueprint.role == "assistant":
            if blueprint.phase in spec.mask_policy.visible_anchor_assistant_steps:
                if blueprint.content is None:
                    raise ValueError(
                        f"Assistant phase '{blueprint.phase}' was requested as a visible anchor but has no anchor content."
                    )
                content = str(_resolve_variant(blueprint.content, template_context, rng, location))
            elif blueprint.phase in spec.mask_policy.masked_phases:
                masked = True
                source = "model_fill"
                content = None
            else:
                if blueprint.content is None:
                    raise ValueError(
                        f"Assistant phase '{blueprint.phase}' is not masked and has no authored anchor content."
                    )
                content = str(_resolve_variant(blueprint.content, template_context, rng, location))
        else:
            if blueprint.content is None:
                raise ValueError(f"Authored anchor phase '{blueprint.phase}' is missing content.")
            content = str(_resolve_variant(blueprint.content, template_context, rng, location))

        compiled_step = CompiledStep(
            step_id=blueprint.step_id,
            phase=blueprint.phase,
            role=blueprint.role,
            source=source,
            masked=masked,
            content=content,
        )
        steps.append(compiled_step)

        for item in blueprint.checks:
            if item.rule not in SUPPORTED_RULE_NAMES:
                raise ValueError(f"Unsupported rule: {item.rule}")
            params = _resolve_templates(
                item.params,
                template_context,
                f"{spec.name}.{blueprint.phase}.{item.name}.params",
            )
            checks.append(
                CompiledCheck(
                    step_id=blueprint.step_id,
                    phase=blueprint.phase,
                    name=item.name,
                    rule=item.rule,
                    params=params,
                    weight=item.weight,
                    description=item.description,
                )
            )

    workload_metadata = {
        "name": spec.name,
        "description": spec.description,
        "tags": list(spec.tags),
        "family": spec.family,
        "pressure_tags": list(spec.interaction_pattern.pressure_tags),
        "final_artifact": spec.interaction_pattern.final_artifact,
        "clarification_count": spec.interaction_pattern.clarification_count,
        "revision_count": spec.interaction_pattern.revision_count,
        "turn_count": sum(1 for step in steps if step.role != "system"),
        "masked_step_count": sum(1 for step in steps if step.masked),
        "masked_phases": list(spec.mask_policy.masked_phases),
        "visible_anchor_assistant_steps": list(
            spec.mask_policy.visible_anchor_assistant_steps
        ),
    }

    payload_for_fingerprint = {
        "family": spec.family,
        "seed": instance_seed,
        "spec_fingerprint": spec_fingerprint,
        "interaction_pattern": spec.interaction_pattern.to_dict(),
        "hidden_state": hidden_state,
        "mask_policy": spec.mask_policy.to_dict(),
        "steps": [step.to_dict() for step in steps],
        "checks": [check.to_dict() for check in checks],
        "workload_metadata": workload_metadata,
    }
    instance_fingerprint = _stable_hash(payload_for_fingerprint)
    instance_id = f"{spec.name}-{instance_index + 1:03d}-{instance_fingerprint[:8]}"
    return CompiledInstance(
        instance_id=instance_id,
        family=spec.family,
        seed=instance_seed,
        spec_fingerprint=spec_fingerprint,
        instance_fingerprint=instance_fingerprint,
        interaction_pattern=spec.interaction_pattern.to_dict(),
        hidden_state=hidden_state,
        mask_policy=spec.mask_policy.to_dict(),
        steps=steps,
        checks=checks,
        workload_metadata=workload_metadata,
    )


def _compile_specs(config: CLIConfig, specs: list[AuthoredSpec]) -> tuple[list[dict[str, Any]], list[CompiledInstance]]:
    authored_specs: list[dict[str, Any]] = []
    compiled_instances: list[CompiledInstance] = []
    next_instance_seed = config.seed

    for spec in specs:
        _validate_spec(spec)
        authored_payload = spec.to_dict()
        authored_specs.append(authored_payload)
        spec_fingerprint = _stable_hash(authored_payload)
        count = spec.instance_count if spec.instance_count is not None else config.instance_count
        if count <= 0:
            raise ValueError(f"{spec.name} must compile at least one instance.")
        for index in range(count):
            compiled_instances.append(
                _compile_spec_instance(
                    spec=spec,
                    spec_fingerprint=spec_fingerprint,
                    instance_seed=next_instance_seed,
                    instance_index=index,
                )
            )
            next_instance_seed += 1
    return authored_specs, compiled_instances


def _evaluate_check(response_text: str, check: CompiledCheck) -> CheckResult:
    response_normalized = _normalize_text(response_text)
    passed = False
    details = ""
    params = check.params

    if check.rule == "contains_all":
        needles = [str(item) for item in params.get("needles", [])]
        missing = [needle for needle in needles if _normalize_text(needle) not in response_normalized]
        passed = not missing
        details = "all required substrings present" if passed else f"missing: {missing}"
    elif check.rule == "contains_none":
        needles = [str(item) for item in params.get("needles", [])]
        present = [needle for needle in needles if _normalize_text(needle) in response_normalized]
        passed = not present
        details = "no forbidden substrings present" if passed else f"found forbidden: {present}"
    elif check.rule == "contains_any":
        needles = [str(item) for item in params.get("needles", [])]
        present = [needle for needle in needles if _normalize_text(needle) in response_normalized]
        passed = bool(present)
        details = f"matched: {present}" if passed else f"expected one of: {needles}"
    elif check.rule == "regex":
        pattern = str(params["pattern"])
        matched = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        passed = matched is not None
        details = "regex matched" if passed else f"regex did not match: {pattern}"
    elif check.rule == "regex_none":
        pattern = str(params["pattern"])
        matched = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        passed = matched is None
        details = "regex absent as expected" if passed else f"forbidden regex matched: {pattern}"
    elif check.rule == "ordered_substrings":
        needles = [str(item) for item in params.get("needles", [])]
        passed = _ordered_contains(response_text, needles)
        details = "substrings appeared in order" if passed else f"order mismatch: {needles}"
    elif check.rule == "line_count_at_most":
        maximum = int(params["max"])
        line_count = len(_nonempty_lines(response_text))
        passed = line_count <= maximum
        details = (
            f"non-empty line count {line_count} <= {maximum}"
            if passed
            else f"non-empty line count {line_count} > {maximum}"
        )
    elif check.rule == "bullet_count_equals":
        expected = int(params["count"])
        actual = len(_bullet_lines(response_text))
        passed = actual == expected
        details = (
            f"bullet count {actual} == {expected}"
            if passed
            else f"bullet count {actual} != {expected}"
        )
    elif check.rule == "bullet_count_at_most":
        maximum = int(params["max"])
        actual = len(_bullet_lines(response_text))
        passed = actual <= maximum
        details = f"bullet count {actual} <= {maximum}" if passed else f"bullet count {actual} > {maximum}"
    elif check.rule == "question_count_equals":
        expected = int(params["count"])
        actual = response_text.count("?")
        passed = actual == expected
        details = (
            f"question count {actual} == {expected}"
            if passed
            else f"question count {actual} != {expected}"
        )
    elif check.rule == "exact_line_set":
        expected_lines = sorted(str(item).strip() for item in params.get("expected_lines", []))
        actual_lines = sorted(_nonempty_lines(response_text))
        missing = [line for line in expected_lines if line not in actual_lines]
        extra = [line for line in actual_lines if line not in expected_lines]
        passed = not missing and not extra
        details = (
            "exact line set matched"
            if passed
            else f"missing lines: {missing}; extra lines: {extra}"
        )
    elif check.rule == "field_equals":
        field_name = str(params["field"])
        expected_value = str(params["value"])
        case_sensitive = bool(params.get("case_sensitive", True))
        flags = re.MULTILINE
        if not case_sensitive:
            flags |= re.IGNORECASE
        pattern = re.compile(rf"^{re.escape(field_name)}:\s*(.+?)\s*$", flags)
        match = pattern.search(response_text)
        if match is None:
            passed = False
            details = f"field '{field_name}' not found"
        else:
            actual_value = match.group(1).strip()
            if case_sensitive:
                passed = actual_value == expected_value
            else:
                passed = actual_value.casefold() == expected_value.casefold()
            details = (
                f"field '{field_name}' matched exactly"
                if passed
                else f"field '{field_name}' value mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    elif check.rule == "required_omission":
        needles = [str(item) for item in params.get("needles", [])]
        present = [needle for needle in needles if _normalize_text(needle) in response_normalized]
        passed = not present
        details = (
            "required omissions honored"
            if passed
            else f"found omitted values: {present}"
        )
    elif check.rule == "state_preserved":
        expected_values = [str(item) for item in params.get("expected_values", [])]
        missing = [value for value in expected_values if _normalize_text(value) not in response_normalized]
        passed = not missing
        details = (
            "required state preserved"
            if passed
            else f"missing preserved values: {missing}"
        )
    elif check.rule == "state_updated_correctly":
        current_values = [str(item) for item in params.get("current_values", [])]
        forbidden_values = [str(item) for item in params.get("forbidden_values", [])]
        missing_current = [
            value for value in current_values if _normalize_text(value) not in response_normalized
        ]
        present_old = [
            value for value in forbidden_values if _normalize_text(value) in response_normalized
        ]
        passed = not missing_current and not present_old
        details = (
            "updated state applied correctly"
            if passed
            else f"missing current values: {missing_current}; old values still present: {present_old}"
        )
    elif check.rule == "no_new_allowed_terms":
        monitored_terms = [str(item) for item in params.get("monitored_terms", [])]
        allowed_terms = {str(item) for item in params.get("allowed_terms", [])}
        present_terms = [
            term for term in monitored_terms if _normalize_text(term) in response_normalized
        ]
        unexpected = [term for term in present_terms if term not in allowed_terms]
        passed = not unexpected
        details = (
            "no unexpected monitored terms appeared"
            if passed
            else f"unexpected monitored terms: {unexpected}"
        )
    else:
        raise ValueError(f"Unsupported rule: {check.rule}")

    return CheckResult(
        name=check.name,
        rule=check.rule,
        passed=passed,
        weight=check.weight,
        score=check.weight if passed else 0.0,
        details=details,
        params=params,
    )


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


def _validate_runtime_preconditions(config: CLIConfig) -> None:
    if config.compile_only:
        return
    if not config.base_url and not os.environ.get("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY must be set for run mode when base_url is not provided.")


def _build_runtime(config: CLIConfig) -> RuntimeHandles:
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
    config: CLIConfig,
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


async def _run_compiled_instances(
    runtime: RuntimeHandles,
    config: CLIConfig,
    compiled_instances: list[CompiledInstance],
) -> list[InstanceRunResult]:
    results: list[InstanceRunResult] = []
    for instance in compiled_instances:
        history: list[dict[str, str]] = []
        checks_by_step: dict[str, list[CompiledCheck]] = {}
        for check in instance.checks:
            checks_by_step.setdefault(check.step_id, []).append(check)

        masked_step_results: list[MaskedStepResult] = []
        for step in instance.steps:
            if step.source == "authored_anchor":
                if step.content is None:
                    raise ValueError(
                        f"Authored anchor step '{step.step_id}' is missing content in compiled instance '{instance.instance_id}'."
                    )
                history.append({"role": step.role, "content": step.content})
                continue

            if step.source != "model_fill" or step.role != "assistant" or not step.masked:
                raise ValueError(
                    f"Invalid compiled step state for '{step.step_id}' in instance '{instance.instance_id}'."
                )

            prompt_messages = [dict(message) for message in history]
            assistant_text, latency_ms, prompt_tokens, completion_tokens, stop_reason = (
                await _generate_assistant_turn(runtime, config, prompt_messages)
            )
            history.append({"role": "assistant", "content": assistant_text})
            check_results = [
                _evaluate_check(assistant_text, check)
                for check in checks_by_step.get(step.step_id, [])
            ]
            masked_step_results.append(
                MaskedStepResult(
                    step_id=step.step_id,
                    prompt_messages=prompt_messages,
                    assistant_message=assistant_text,
                    latency_ms=round(latency_ms, 2),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    stop_reason=stop_reason,
                    checks=check_results,
                )
            )

        instance_score = sum(
            check.score for result in masked_step_results for check in result.checks
        )
        instance_max_score = sum(
            check.weight for result in masked_step_results for check in result.checks
        )
        results.append(
            InstanceRunResult(
                instance_id=instance.instance_id,
                masked_step_results=masked_step_results,
                instance_score=instance_score,
                instance_max_score=instance_max_score,
                passed=all(
                    check.passed
                    for result in masked_step_results
                    for check in result.checks
                ),
            )
        )
    return results


def _compute_run_fingerprint(config: CLIConfig, compiled_instances: list[CompiledInstance]) -> str:
    return _stable_hash(
        {
            "config": {
                "base_model": config.base_model,
                "model_path": config.model_path,
                "base_url": config.base_url,
                "renderer_name": config.renderer_name,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "seed": config.seed,
                "compile_only": config.compile_only,
            },
            "compiled_instance_fingerprints": [
                item.instance_fingerprint for item in compiled_instances
            ],
        }
    )


def _build_export_payload(compiled_instances: list[CompiledInstance]) -> dict[str, Any]:
    instances = []
    for instance in compiled_instances:
        instances.append(
            {
                "instance_id": instance.instance_id,
                "family": instance.family,
                "workload_metadata": instance.workload_metadata,
                "conversation": [step.to_dict() for step in instance.steps],
                "mask_annotations": [
                    {
                        "step_id": step.step_id,
                        "phase": step.phase,
                        "masked": step.masked,
                        "source": step.source,
                    }
                    for step in instance.steps
                ],
                "checks": [check.to_dict() for check in instance.checks],
            }
        )
    return {
        "schema_version": "inferencex_v1",
        "instances": instances,
    }


def _build_bundle(
    config: CLIConfig,
    authored_specs: list[dict[str, Any]],
    compiled_instances: list[CompiledInstance],
    run_results: list[InstanceRunResult] | None,
    started_at: str,
    finished_at: str,
    resolved_renderer_name: str | None,
) -> dict[str, Any]:
    run_fingerprint = _compute_run_fingerprint(config, compiled_instances)
    summary: dict[str, Any] = {
        "compile_only": config.compile_only,
        "authored_spec_count": len(authored_specs),
        "compiled_instance_count": len(compiled_instances),
        "masked_step_count": sum(
            1 for instance in compiled_instances for step in instance.steps if step.masked
        ),
    }

    if run_results is None:
        summary.update(
            {
                "run_result_count": 0,
                "passed_instances": None,
                "failed_instances": None,
                "total_checks": None,
                "passed_checks": None,
                "failed_checks": None,
                "score": None,
                "max_score": None,
                "score_pct": None,
                "total_latency_ms": None,
                "total_prompt_tokens": None,
                "total_completion_tokens": None,
            }
        )
    else:
        total_checks = sum(
            len(step_result.checks)
            for result in run_results
            for step_result in result.masked_step_results
        )
        passed_checks = sum(
            1
            for result in run_results
            for step_result in result.masked_step_results
            for check in step_result.checks
            if check.passed
        )
        total_score = sum(result.instance_score for result in run_results)
        total_max_score = sum(result.instance_max_score for result in run_results)
        total_latency_ms = sum(
            step_result.latency_ms
            for result in run_results
            for step_result in result.masked_step_results
        )
        total_prompt_tokens = sum(
            step_result.prompt_tokens or 0
            for result in run_results
            for step_result in result.masked_step_results
        )
        total_completion_tokens = sum(
            step_result.completion_tokens
            for result in run_results
            for step_result in result.masked_step_results
        )
        summary.update(
            {
                "run_result_count": len(run_results),
                "passed_instances": sum(1 for item in run_results if item.passed),
                "failed_instances": sum(1 for item in run_results if not item.passed),
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "score": total_score,
                "max_score": total_max_score,
                "score_pct": (
                    0.0 if total_max_score == 0 else round((total_score / total_max_score) * 100, 2)
                ),
                "total_latency_ms": round(total_latency_ms, 2),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
            }
        )

    meta = {
        "started_at": started_at,
        "finished_at": finished_at,
        "run_fingerprint": run_fingerprint,
        "bundle_id": f"{finished_at.replace(':', '').replace('-', '')}_{run_fingerprint[:8]}",
    }
    config_payload = config.to_dict()
    config_payload["resolved_renderer_name"] = resolved_renderer_name

    bundle = {
        "meta": meta,
        "config": config_payload,
        "authored_specs": authored_specs,
        "compiled_instances": [item.to_dict() for item in compiled_instances],
        "summary": summary,
        "export_payloads": {"inferencex_v1": _build_export_payload(compiled_instances)},
    }
    if run_results is not None:
        bundle["run_results"] = [item.to_dict() for item in run_results]
    return bundle


def _write_bundle(output_dir: Path, bundle: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    fingerprint = bundle["meta"]["run_fingerprint"][:8]
    output_path = output_dir / f"workload_bundle_{timestamp}_{fingerprint}.json"
    output_path.write_text(json.dumps(bundle, indent=2) + "\n")
    return output_path


def _print_patterns() -> None:
    for spec in _list_builtin_patterns():
        print(f"{spec.name}: {spec.description}")


def _print_rules() -> None:
    for rule_name, description in SUPPORTED_RULES:
        print(f"{rule_name}: {description}")


def _print_bundle_summary(bundle: dict[str, Any], output_path: Path) -> None:
    summary = bundle["summary"]
    print(f"Bundle ID: {bundle['meta']['bundle_id']}")
    print(f"Run fingerprint: {bundle['meta']['run_fingerprint']}")
    print(f"Compiled instances: {summary['compiled_instance_count']}")
    print(f"Masked steps: {summary['masked_step_count']}")
    if bundle["config"]["compile_only"]:
        print("Mode: compile-only")
    else:
        print("Mode: compile-and-run")
        print(
            "Overall: "
            f"{summary['score_pct']:.2f}% "
            f"({summary['score']:.1f}/{summary['max_score']:.1f})"
        )
        print(
            "Instances: "
            f"{summary['passed_instances']} passed, "
            f"{summary['failed_instances']} failed"
        )
        print(
            "Checks: "
            f"{summary['passed_checks']}/{summary['total_checks']} passed"
        )
    print(f"Bundle: {output_path}")


def _load_specs_from_file(path: Path) -> list[AuthoredSpec]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return [AuthoredSpec.from_dict(payload)]
    if isinstance(payload, list):
        return [AuthoredSpec.from_dict(item) for item in payload]
    raise ValueError("spec file must contain a JSON object or a list of objects.")


def _load_requested_specs(config: CLIConfig) -> list[AuthoredSpec]:
    if config.pattern_names and config.spec_file:
        raise ValueError("Use either --pattern or --spec-file, not both.")
    if config.spec_file:
        return _load_specs_from_file(Path(config.spec_file))
    if config.pattern_names:
        specs = []
        for name in config.pattern_names:
            if name not in FAMILY_EXPECTATIONS:
                raise ValueError(f"Unknown pattern: {name}")
            specs.append(_build_builtin_spec(name))
        return specs
    raise ValueError("Either --pattern NAME or --spec-file PATH is required.")


def _build_rule_case(
    rule: str,
    params: dict[str, Any],
    response_text: str,
    expected: bool,
) -> tuple[CompiledCheck, str, bool]:
    return (
        CompiledCheck(
            step_id="self_test_step",
            phase="self_test_phase",
            name=f"{rule}_test",
            rule=rule,
            params=params,
        ),
        response_text,
        expected,
    )


def _run_self_test() -> int:
    rule_cases = [
        _build_rule_case("contains_all", {"needles": ["alpha", "beta"]}, "alpha beta", True),
        _build_rule_case("contains_none", {"needles": ["gamma"]}, "alpha beta", True),
        _build_rule_case("contains_any", {"needles": ["beta", "gamma"]}, "alpha beta", True),
        _build_rule_case("regex", {"pattern": r"beta$"}, "alpha beta", True),
        _build_rule_case("regex_none", {"pattern": r"gamma"}, "alpha beta", True),
        _build_rule_case(
            "ordered_substrings",
            {"needles": ["alpha", "beta"]},
            "alpha something beta",
            True,
        ),
        _build_rule_case("line_count_at_most", {"max": 2}, "one\ntwo", True),
        _build_rule_case("bullet_count_equals", {"count": 2}, "- one\n- two", True),
        _build_rule_case("bullet_count_at_most", {"max": 2}, "- one\n- two", True),
        _build_rule_case("question_count_equals", {"count": 1}, "Really?", True),
        _build_rule_case(
            "exact_line_set",
            {"expected_lines": ["Ticket: X", "Region: us-west-2"]},
            "Region: us-west-2\nTicket: X",
            True,
        ),
        _build_rule_case(
            "field_equals",
            {"field": "Ticket", "value": "SUP-1"},
            "Ticket: SUP-1\nRegion: us-west-2",
            True,
        ),
        _build_rule_case(
            "required_omission",
            {"needles": ["secret"]},
            "public summary only",
            True,
        ),
        _build_rule_case(
            "state_preserved",
            {"expected_values": ["SUP-1", "us-west-2"]},
            "Ticket SUP-1 in us-west-2",
            True,
        ),
        _build_rule_case(
            "state_updated_correctly",
            {"current_values": ["resolved"], "forbidden_values": ["investigating"]},
            "The case is resolved.",
            True,
        ),
        _build_rule_case(
            "no_new_allowed_terms",
            {"monitored_terms": ["alpha", "beta"], "allowed_terms": ["alpha"]},
            "alpha only",
            True,
        ),
    ]

    for check, response_text, expected in rule_cases:
        result = _evaluate_check(response_text, check)
        if result.passed != expected:
            raise AssertionError(
                f"Rule self-test failed for {check.rule}: expected {expected}, got {result.passed}"
            )

    context = {"facts": {"ticket_id": "SUP-1"}}
    try:
        _resolve_templates("{{facts.missing}}", context, "self_test.missing")
    except ValueError:
        pass
    else:
        raise AssertionError("Unresolved template key should fail.")

    config = CLIConfig(
        base_model=DEFAULT_BASE_MODEL,
        model_path=None,
        base_url=None,
        renderer_name=None,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
        seed=DEFAULT_SEED,
        output_dir=DEFAULT_OUTPUT_DIR,
        pattern_names=["support_escalation"],
        spec_file=None,
        instance_count=1,
        compile_only=True,
        list_patterns=False,
        list_rules=False,
        self_test=True,
    )
    specs = [_build_builtin_spec("support_escalation")]
    authored_a, compiled_a = _compile_specs(config, specs)
    authored_b, compiled_b = _compile_specs(config, specs)
    if compiled_a[0].instance_fingerprint != compiled_b[0].instance_fingerprint:
        raise AssertionError("Same-seed compilation should produce identical instance fingerprints.")
    if authored_a != authored_b:
        raise AssertionError("Same-seed compilation should preserve authored specs exactly.")

    config_other_seed = CLIConfig(**{**config.to_dict(), "seed": DEFAULT_SEED + 1})
    _unused_authored_c, compiled_c = _compile_specs(config_other_seed, specs)
    if compiled_a[0].hidden_state == compiled_c[0].hidden_state:
        raise AssertionError("Different seeds should produce different hidden state.")
    phases_a = [step.phase for step in compiled_a[0].steps]
    phases_c = [step.phase for step in compiled_c[0].steps]
    if phases_a != phases_c:
        raise AssertionError("Different seeds should not change the conversation structure.")

    for family_name in FAMILY_EXPECTATIONS:
        family_spec = _build_builtin_spec(family_name)
        _validate_spec(family_spec)
        _compiled_authored, family_compiled = _compile_specs(
            CLIConfig(
                **{
                    **config.to_dict(),
                    "pattern_names": [family_name],
                }
            ),
            [family_spec],
        )
        masked_phases = {
            step.phase for step in family_compiled[0].steps if step.masked
        }
        if masked_phases != set(family_spec.mask_policy.masked_phases):
            raise AssertionError(
                f"Mask policy compilation mismatch for {family_name}: {masked_phases}"
            )

    fixed_started = "2026-03-26T12:00:00-07:00"
    fixed_finished = "2026-03-26T12:00:01-07:00"
    bundle_a = _build_bundle(
        config=config,
        authored_specs=authored_a,
        compiled_instances=compiled_a,
        run_results=None,
        started_at=fixed_started,
        finished_at=fixed_finished,
        resolved_renderer_name=None,
    )
    bundle_b = _build_bundle(
        config=config,
        authored_specs=authored_b,
        compiled_instances=compiled_b,
        run_results=None,
        started_at=fixed_started,
        finished_at=fixed_finished,
        resolved_renderer_name=None,
    )
    if bundle_a["meta"]["run_fingerprint"] != bundle_b["meta"]["run_fingerprint"]:
        raise AssertionError("Bundle fingerprints should be stable for identical inputs.")

    print("Self-test passed.")
    return 0


def _parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compile high-fidelity conversational workloads with hidden state, assistant-side masking, "
            "deterministic scoring rules, and optional Tinker execution."
        )
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--renderer", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Built-in workload pattern to compile. Repeat the flag to include multiple patterns.",
    )
    parser.add_argument("--spec-file", default=None, help="JSON authored spec file.")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--list-patterns", action="store_true")
    parser.add_argument("--list-rules", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    pattern_names: list[str] = []
    for item in args.pattern:
        pattern_names.extend(part.strip() for part in str(item).split(",") if part.strip())

    return CLIConfig(
        base_model=str(args.base_model),
        model_path=_clean_optional(args.model_path),
        base_url=_clean_optional(args.base_url),
        renderer_name=_clean_optional(args.renderer),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        pattern_names=pattern_names,
        spec_file=_clean_optional(args.spec_file),
        instance_count=int(args.instance_count),
        compile_only=bool(args.compile_only),
        list_patterns=bool(args.list_patterns),
        list_rules=bool(args.list_rules),
        self_test=bool(args.self_test),
    )


async def _async_main(config: CLIConfig) -> int:
    if config.self_test:
        return _run_self_test()

    if config.list_patterns:
        _print_patterns()
        if not config.list_rules:
            return 0
    if config.list_rules:
        _print_rules()
        if not config.pattern_names and not config.spec_file:
            return 0

    specs = _load_requested_specs(config)
    started_at = _now_iso()
    authored_specs, compiled_instances = _compile_specs(config, specs)

    run_results: list[InstanceRunResult] | None = None
    resolved_renderer_name: str | None = None
    if not config.compile_only:
        _validate_runtime_preconditions(config)
        runtime = _build_runtime(config)
        resolved_renderer_name = runtime.resolved_renderer_name
        run_results = await _run_compiled_instances(runtime, config, compiled_instances)

    finished_at = _now_iso()
    bundle = _build_bundle(
        config=config,
        authored_specs=authored_specs,
        compiled_instances=compiled_instances,
        run_results=run_results,
        started_at=started_at,
        finished_at=finished_at,
        resolved_renderer_name=resolved_renderer_name,
    )
    output_path = _write_bundle(Path(config.output_dir), bundle)
    _print_bundle_summary(bundle, output_path)

    if run_results is None:
        return 0
    return 0 if all(item.passed for item in run_results) else 1


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
