#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def check(
    name: str,
    rule: str,
    *,
    weight: float = 1.0,
    description: str = "",
    **params: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": name,
        "rule": rule,
        "weight": weight,
    }
    if description:
        payload["description"] = description
    payload.update(params)
    return payload


def turn(user: str, checks: list[dict[str, object]]) -> dict[str, object]:
    return {"user": user, "checks": checks}


def scenario(
    *,
    name: str,
    description: str,
    system_prompt: str,
    facts: dict[str, str],
    turns: list[dict[str, object]],
    tags: list[str],
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "facts": facts,
        "turns": turns,
        "tags": tags,
    }


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

FIRST_NAMES = [
    "Mina",
    "Jordan",
    "Avery",
    "Priya",
    "Noah",
    "Elena",
    "Marcus",
    "Riley",
    "Sofia",
    "Theo",
]

LAST_NAMES = [
    "Patel",
    "Lopez",
    "Nguyen",
    "Shah",
    "Bennett",
    "Khan",
    "Morris",
    "Garcia",
    "Kim",
    "Sullivan",
]

EXPENSE_TRANSPORTS = [
    "airport rideshare",
    "train fare",
    "taxi to client site",
    "parking receipt",
    "shuttle transfer",
    "subway fare",
    "ferry ticket",
    "bus pass",
    "commuter rail",
    "car share fee",
]

EXPENSE_FORBIDDEN = [
    "minibar",
    "alcohol",
    "spa service",
    "gift shop",
    "movie rental",
    "late checkout fee",
    "snack drawer",
    "room upgrade",
    "souvenir charge",
    "cocktail tab",
]

ISSUE_TRIGGERS = [
    "password reset",
    "billing profile update",
    "role change",
    "SSO migration",
    "API key rotation",
    "domain switch",
    "account merge",
    "seat reassignment",
    "email alias update",
    "plan downgrade",
]

ENVIRONMENTS = [
    "staging",
    "sandbox",
    "preprod",
    "qa",
    "dev-preview",
    "integration",
    "staging-eu",
    "staging-us",
    "uat",
    "pilot",
]

REGIONS = [
    "us-west-2",
    "us-east-1",
    "eu-west-1",
    "eu-central-1",
    "ap-southeast-1",
    "ap-northeast-1",
    "ca-central-1",
    "sa-east-1",
    "us-gov-west-1",
    "eu-north-1",
]

PRODUCT_NAMES = [
    "Northwind Copilot",
    "Harbor Assist",
    "Atlas Brief",
    "Signal Canvas",
    "Summit Notes",
    "Pioneer Desk",
    "Voyager Hub",
    "Beacon Studio",
    "Relay Board",
    "Crescent Flow",
]

COUNTERPARTIES = [
    "Acme Retail",
    "Blue Mesa Health",
    "Lighthouse Energy",
    "Summit Logistics",
    "Cedar Finance",
    "Brightline Schools",
    "Granite Telecom",
    "Pioneer Foods",
    "Meridian Labs",
    "Juniper Media",
]

RISK_NOTES = [
    "data residency addendum pending",
    "subprocessor list still under review",
    "insurance certificate expires this quarter",
    "security exhibit needs updated signatures",
    "redline on limitation of liability remains open",
    "access-control appendix not yet approved",
    "privacy schedule requires final markup",
    "audit-rights clause still disputed",
    "termination notice language needs revision",
    "governing-law fallback is unresolved",
]

SERVICES = [
    "billing API",
    "search indexer",
    "identity gateway",
    "notification fanout",
    "document renderer",
    "checkout service",
    "analytics ingest",
    "admin console",
    "catalog sync",
    "support inbox",
]

PRIMARY_CHANNELS = [
    "PagerDuty bridge",
    "on-call hotline",
    "priority incident line",
    "operations bridge",
    "war-room channel",
    "response hotline",
    "incident bridge",
    "escalation desk",
    "triage hotline",
    "command bridge",
]


def month_day(index: int) -> str:
    month = MONTHS[index % len(MONTHS)]
    day = 10 + index
    return f"{month} {day}"


def month_day_plus_one(index: int) -> str:
    month = MONTHS[index % len(MONTHS)]
    day = 11 + index
    return f"{month} {day}"


def time_utc(index: int) -> str:
    hour = 9 + index
    minute = 15 if index % 2 == 0 else 45
    return f"{hour:02d}:{minute:02d} UTC"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_expense_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for index in range(10):
        trip_code = f"TRIP-{4821 + index}"
        hotel_cap = f"${180 + (index * 5)}"
        meal_cap = f"${65 + index}"
        hotel_amount = f"${172 + index}"
        transport_label = EXPENSE_TRANSPORTS[index]
        transport_amount = f"${24 + index}"
        forbidden_label = EXPENSE_FORBIDDEN[index]
        forbidden_amount = f"${19 + index}"
        name = f"expense_policy_memory_{index + 1:02d}"
        scenarios.append(
            scenario(
                name=name,
                description="Memory-heavy travel reimbursement evaluation with one forbidden item.",
                system_prompt=(
                    "You are an internal operations assistant. Stay grounded in the conversation, "
                    "answer directly, and do not invent policy exceptions."
                ),
                facts={
                    "trip_code": trip_code,
                    "hotel_cap": hotel_cap,
                    "meal_cap": meal_cap,
                    "approved_transport": transport_label,
                    "forbidden_item": forbidden_label,
                },
                turns=[
                    turn(
                        (
                            f"I'm going to give you a reimbursement policy to use later. Hotel: up to {hotel_cap}/night "
                            f"with receipt. Meals: up to {meal_cap}/day. {transport_label.capitalize()} is reimbursable. "
                            f"{forbidden_label.capitalize()} is never reimbursable. My trip code is {trip_code}. "
                            "Reply with one sentence confirming the trip code and the hotel and meal caps."
                        ),
                        [
                            check(
                                "acknowledges_key_policy_facts",
                                "contains_all",
                                needles=["{{trip_code}}", "{{hotel_cap}}", "{{meal_cap}}"],
                            ),
                            check("ack_is_compact", "line_count_at_most", max=2),
                        ],
                    ),
                    turn(
                        (
                            "Now evaluate these expenses and answer in exactly three bullets: "
                            f"{hotel_amount} hotel, {transport_amount} {transport_label}, {forbidden_amount} {forbidden_label}."
                        ),
                        [
                            check("uses_three_bullets", "bullet_count_equals", count=3),
                            check(
                                "hotel_is_approved",
                                "regex",
                                pattern=rf"(?is)({re.escape(hotel_amount)}|hotel).*(reimburs|covered|eligible|approved)",
                            ),
                            check(
                                "transport_is_approved",
                                "regex",
                                pattern=rf"(?is)({re.escape(transport_amount)}|{re.escape(transport_label)}).*(reimburs|covered|eligible|approved)",
                            ),
                            check(
                                "forbidden_item_is_rejected",
                                "regex",
                                pattern=rf"(?is)({re.escape(forbidden_amount)}|{re.escape(forbidden_label)}).*((not|never|isn't|is not).*(reimburs|covered|eligible|approved)|excluded|ineligible)",
                            ),
                        ],
                    ),
                    turn(
                        "Give me a one-sentence summary of only the approved items and include the trip code exactly.",
                        [
                            check("summary_keeps_trip_code", "contains_all", needles=["{{trip_code}}"]),
                            check(
                                "summary_mentions_both_approved_items",
                                "regex",
                                pattern=rf"(?is)((hotel|{re.escape(hotel_amount)}).*(?:{re.escape(transport_label)}|{re.escape(transport_amount)}))|(((?:{re.escape(transport_label)}|{re.escape(transport_amount)})).*(hotel|{re.escape(hotel_amount)}))",
                            ),
                            check(
                                "summary_omits_rejected_item",
                                "regex_none",
                                pattern=rf"(?is)({re.escape(forbidden_label)}|{re.escape(forbidden_amount)})",
                            ),
                        ],
                    ),
                ],
                tags=["memory", "policy", "consistency", "travel"],
            )
        )
    return scenarios


def build_support_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for index in range(10):
        ticket_id = f"SUP-{7714 + index}"
        customer_name = f"{FIRST_NAMES[index]} {LAST_NAMES[index]}"
        environment = ENVIRONMENTS[index]
        region = REGIONS[index]
        trigger = ISSUE_TRIGGERS[index]
        name = f"support_handoff_precision_{index + 1:02d}"
        scenarios.append(
            scenario(
                name=name,
                description="Support handoff note with exact field formatting and retained facts.",
                system_prompt=(
                    "You are a support operations assistant. Follow formatting instructions exactly, "
                    "retain incident details across turns, and do not invent missing evidence."
                ),
                facts={
                    "ticket_id": ticket_id,
                    "customer_name": customer_name,
                    "environment": environment,
                    "region": region,
                    "trigger": trigger,
                },
                turns=[
                    turn(
                        (
                            f"I need a support handoff note later. Facts to remember: customer {customer_name}, ticket {ticket_id}, "
                            f"issue started right after a {trigger}, and it only reproduces in {environment}. "
                            "Before drafting anything, ask me exactly one clarifying question."
                        ),
                        [check("asks_one_clarifying_question", "question_count_equals", count=1)],
                    ),
                    turn(
                        (
                            f"The affected region is {region} and there were no 2FA errors. Now draft the handoff note using "
                            "exactly these labels on separate lines: Ticket, Customer, Problem, Environment, Region, Next Step."
                        ),
                        [
                            check("ticket_line_exact", "regex", pattern=rf"(?im)^Ticket:\s*{re.escape(ticket_id)}\s*$"),
                            check("customer_line_exact", "regex", pattern=rf"(?im)^Customer:\s*{re.escape(customer_name)}\s*$"),
                            check("problem_mentions_trigger", "regex", pattern=rf"(?im)^Problem:.*{re.escape(trigger)}.*$"),
                            check("environment_line_exact", "regex", pattern=rf"(?im)^Environment:\s*{re.escape(environment)}\s*$"),
                            check("region_line_exact", "regex", pattern=rf"(?im)^Region:\s*{re.escape(region)}\s*$"),
                            check("mentions_no_2fa_errors", "regex", pattern=r"(?is)(no|without).*(2FA|two-factor).*(errors?)"),
                            check("next_step_line_present", "regex", pattern=r"(?im)^Next Step:\s*\S.+$"),
                        ],
                    ),
                ],
                tags=["formatting", "memory", "handoff", "support"],
            )
        )
    return scenarios


def build_launch_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for index in range(10):
        product_name = PRODUCT_NAMES[index]
        launch_date = month_day(index)
        drift_date = month_day_plus_one(index)
        rollout_count = 12 + index
        rollout_group = f"{rollout_count} design partners"
        drift_rollout = f"{rollout_count + 1} design partners"
        url = f"{slug(product_name)}.example.com/launch"
        name = f"launch_brief_consistency_{index + 1:02d}"
        scenarios.append(
            scenario(
                name=name,
                description="Preserve exact launch facts across a draft and revision cycle.",
                system_prompt=(
                    "You are a product communications assistant. Keep every explicit fact stable "
                    "unless the user changes it."
                ),
                facts={
                    "product_name": product_name,
                    "launch_date": launch_date,
                    "rollout_group": rollout_group,
                    "launch_url": url,
                },
                turns=[
                    turn(
                        (
                            f"Use these launch facts later: product {product_name}, public launch {launch_date}, "
                            f"initial rollout to {rollout_group}, and URL {url}. Reply with a short acknowledgement "
                            "that repeats the product name and launch date."
                        ),
                        [check("ack_repeats_product_and_date", "contains_all", needles=["{{product_name}}", "{{launch_date}}"])],
                    ),
                    turn(
                        "Draft a two-sentence announcement paragraph.",
                        [
                            check(
                                "draft_keeps_all_launch_facts",
                                "contains_all",
                                needles=["{{product_name}}", "{{launch_date}}", "{{rollout_group}}", "{{launch_url}}"],
                            )
                        ],
                    ),
                    turn(
                        "Revise the announcement to be shorter, but keep every factual detail unchanged.",
                        [
                            check(
                                "revision_preserves_all_facts",
                                "contains_all",
                                needles=["{{product_name}}", "{{launch_date}}", "{{rollout_group}}", "{{launch_url}}"],
                            ),
                            check(
                                "revision_does_not_drift",
                                "contains_none",
                                needles=[drift_date, drift_rollout],
                            ),
                        ],
                    ),
                ],
                tags=["revision", "consistency", "factuality", "launch"],
            )
        )
    return scenarios


def build_contract_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for index in range(10):
        contract_id = f"CT-{2401 + index}"
        counterparty = COUNTERPARTIES[index]
        renewal_date = month_day(index + 2)
        owner_name = f"{FIRST_NAMES[(index + 3) % 10]} {LAST_NAMES[(index + 4) % 10]}"
        risk_note = RISK_NOTES[index]
        name = f"contract_review_ordering_{index + 1:02d}"
        scenarios.append(
            scenario(
                name=name,
                description="Contract review workflow with ordered fact recall and forbidden drift.",
                system_prompt=(
                    "You are a legal operations assistant. Keep contract facts stable, follow output "
                    "formatting exactly, and avoid inventing closed issues."
                ),
                facts={
                    "contract_id": contract_id,
                    "counterparty": counterparty,
                    "renewal_date": renewal_date,
                    "owner_name": owner_name,
                    "risk_note": risk_note,
                },
                turns=[
                    turn(
                        (
                            f"Remember these contract review details for later: contract {contract_id}, counterparty {counterparty}, "
                            f"renewal deadline {renewal_date}, owner {owner_name}, and note '{risk_note}'. "
                            "Reply with one short line repeating the contract id and renewal date."
                        ),
                        [
                            check("ack_repeats_contract_and_date", "contains_all", needles=["{{contract_id}}", "{{renewal_date}}"]),
                            check("ack_is_one_line", "line_count_at_most", max=1),
                        ],
                    ),
                    turn(
                        (
                            "Write exactly three bullets labeled Status, Risk, and Owner. Mention the counterparty, the risk note, "
                            "and the owner name. Do not add a signed date."
                        ),
                        [
                            check("uses_three_bullets", "bullet_count_equals", count=3),
                            check(
                                "draft_mentions_required_fields",
                                "contains_all",
                                needles=["{{counterparty}}", "{{risk_note}}", "{{owner_name}}"],
                            ),
                        ],
                    ),
                    turn(
                        "Give one sentence in this order: contract id, counterparty, renewal date. Do not mention pricing.",
                        [
                            check(
                                "final_sentence_preserves_order",
                                "ordered_substrings",
                                needles=["{{contract_id}}", "{{counterparty}}", "{{renewal_date}}"],
                            ),
                            check("final_sentence_omits_pricing", "contains_none", needles=["pricing"]),
                        ],
                    ),
                ],
                tags=["contract", "ordering", "memory", "legal"],
            )
        )
    return scenarios


def build_outage_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    severities = ["SEV-1", "SEV-2", "SEV-3", "SEV-2", "SEV-1", "SEV-3", "SEV-2", "SEV-1", "SEV-3", "SEV-2"]
    for index in range(10):
        incident_id = f"INC-{3201 + index}"
        service_name = SERVICES[index]
        severity = severities[index]
        eta = time_utc(index)
        name = f"outage_update_guardrail_{index + 1:02d}"
        scenarios.append(
            scenario(
                name=name,
                description="Outage status updates with explicit no-resolution guardrails.",
                system_prompt=(
                    "You are an incident communications assistant. Keep incident facts precise, do "
                    "not claim resolution unless the user says so, and preserve ordering when asked."
                ),
                facts={
                    "incident_id": incident_id,
                    "service_name": service_name,
                    "severity": severity,
                    "eta": eta,
                },
                turns=[
                    turn(
                        (
                            f"Remember this incident for later: {incident_id}, service {service_name}, severity {severity}, "
                            f"next mitigation update at {eta}. Reply with one sentence confirming the incident id and severity."
                        ),
                        [
                            check("ack_repeats_incident_and_severity", "contains_all", needles=["{{incident_id}}", "{{severity}}"]),
                            check("ack_is_single_line", "line_count_at_most", max=1),
                        ],
                    ),
                    turn(
                        (
                            f"Give exactly two bullets: impact and ETA. Mention {service_name} and {eta}. "
                            "Do not say the incident is resolved."
                        ),
                        [
                            check("uses_two_bullets", "bullet_count_equals", count=2),
                            check("mentions_service_and_eta", "contains_all", needles=["{{service_name}}", "{{eta}}"]),
                            check("omits_resolved_claim", "contains_none", needles=["resolved"]),
                        ],
                    ),
                    turn(
                        "Write one sentence in this order: incident id, service name, severity, ETA.",
                        [
                            check(
                                "final_sentence_preserves_order",
                                "ordered_substrings",
                                needles=["{{incident_id}}", "{{service_name}}", "{{severity}}", "{{eta}}"],
                            ),
                            check("final_sentence_is_single_line", "line_count_at_most", max=1),
                        ],
                    ),
                ],
                tags=["incident", "guardrail", "ordering", "ops"],
            )
        )
    return scenarios


def build_routing_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for index in range(10):
        request_id = f"ESC-{5101 + index}"
        primary_channel = PRIMARY_CHANNELS[index]
        root_slug = slug(primary_channel)
        backup_slack = f"#{root_slug}-backup"
        backup_email = f"{root_slug}@alerts.example.com"
        name = f"escalation_channel_routing_{index + 1:02d}"
        scenarios.append(
            scenario(
                name=name,
                description="Escalation routing with backup channel alternatives and forbidden mention checks.",
                system_prompt=(
                    "You are an escalation routing assistant. Keep channel names exact, follow "
                    "formatting requests precisely, and avoid introducing unsupported escalation paths."
                ),
                facts={
                    "request_id": request_id,
                    "primary_channel": primary_channel,
                    "backup_slack": backup_slack,
                    "backup_email": backup_email,
                },
                turns=[
                    turn(
                        (
                            f"Remember these escalation channels for later: request {request_id}, primary channel {primary_channel}, "
                            f"backup Slack {backup_slack}, and backup email {backup_email}. "
                            "Reply with one short acknowledgement that repeats the request id and primary channel."
                        ),
                        [
                            check("ack_repeats_request_and_primary", "contains_all", needles=["{{request_id}}", "{{primary_channel}}"]),
                            check("ack_is_single_line", "line_count_at_most", max=1),
                        ],
                    ),
                    turn(
                        (
                            f"Write exactly two bullets telling the operator what to do if {primary_channel} fails. "
                            f"Mention {request_id} and use at least one backup channel: {backup_slack} or {backup_email}."
                        ),
                        [
                            check("uses_two_bullets", "bullet_count_equals", count=2),
                            check("mentions_request_id", "contains_all", needles=["{{request_id}}"]),
                            check("uses_one_backup_channel", "contains_any", needles=["{{backup_slack}}", "{{backup_email}}"]),
                        ],
                    ),
                    turn(
                        "Give one sentence that includes the request id and the primary channel, and do not mention SMS.",
                        [
                            check("final_sentence_mentions_request_and_primary", "contains_all", needles=["{{request_id}}", "{{primary_channel}}"]),
                            check("final_sentence_omits_sms", "contains_none", needles=["SMS"]),
                        ],
                    ),
                ],
                tags=["routing", "alternatives", "ops", "channels"],
            )
        )
    return scenarios


def build_dataset() -> list[dict[str, object]]:
    scenarios = (
        build_expense_scenarios()
        + build_support_scenarios()
        + build_launch_scenarios()
        + build_contract_scenarios()
        + build_outage_scenarios()
        + build_routing_scenarios()
    )
    names = [item["name"] for item in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Scenario names must be unique.")
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a large multi-turn conversation eval dataset JSON file."
    )
    parser.add_argument(
        "--output",
        default="scenario_datasets/conversation_eval_large_60.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    dataset = build_dataset()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    total_checks = sum(len(turn["checks"]) for item in dataset for turn in item["turns"])
    print(f"Wrote {len(dataset)} scenarios with {total_checks} total checks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
