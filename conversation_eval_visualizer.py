
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_report_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            matched = sorted(path.glob("conversation_eval_*.json"))
            if not matched:
                matched = sorted(path.glob("*.json"))
            if matched:
                latest = max(matched, key=lambda candidate: (candidate.stat().st_mtime, candidate.name))
                paths.append(latest)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input not found: {path}")
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    if not deduped:
        raise FileNotFoundError("No conversation_eval JSON files found.")
    return deduped


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def pct(a: float, b: float) -> float:
    return round(safe_div(a, b) * 100.0, 2)


def truncate(text: str, limit: int = 140) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def truncate_block(text: str, char_limit: int = 420, line_limit: int = 8) -> str:
    raw = str(text).strip()
    if not raw:
        return ""
    lines = raw.splitlines()
    truncated = False
    if len(lines) > line_limit:
        raw = "\n".join(lines[:line_limit])
        truncated = True
    if len(raw) > char_limit:
        return raw[: char_limit - 1].rstrip() + "…"
    if truncated:
        return raw.rstrip() + "\n…"
    return raw


def format_ms(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.0f} ms"


def format_num(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def collect_run_summary(report: dict[str, Any], source_path: Path) -> dict[str, Any]:
    summary = report.get("summary", {})
    config = report.get("config", {})
    results = report.get("results", [])
    prompt_tokens_total = sum(
        trace.get("prompt_tokens") or 0
        for result in results
        for trace in result.get("traces", [])
    )
    traces = [trace for result in results for trace in result.get("traces", [])]
    scenario_scores = {result["name"]: result.get("score_pct", 0.0) for result in results}
    return {
        "source_path": str(source_path),
        "filename": source_path.name,
        "run_id": report.get("meta", {}).get("run_id", "—"),
        "model": config.get("base_model", "—"),
        "renderer": config.get("resolved_renderer_name") or config.get("renderer_name") or "—",
        "score_pct": summary.get("score_pct", 0.0),
        "passed_scenarios": summary.get("passed_scenarios", 0),
        "scenario_count": summary.get("scenario_count", 0),
        "passed_checks": summary.get("passed_checks", 0),
        "total_checks": summary.get("total_checks", 0),
        "total_latency_ms": summary.get("total_latency_ms", 0.0),
        "total_completion_tokens": summary.get("total_completion_tokens", 0),
        "total_prompt_tokens": prompt_tokens_total,
        "turn_count": len(traces),
        "scenario_scores": scenario_scores,
        "fingerprint": report.get("meta", {}).get("workload_fingerprint", ""),
        "started_at": report.get("meta", {}).get("started_at", "—"),
        "finished_at": report.get("meta", {}).get("finished_at", "—"),
    }


def rule_counts(report: dict[str, Any]) -> Counter:
    counts: Counter = Counter()
    for scenario in report.get("scenarios", []):
        for turn in scenario.get("turns", []):
            for check in turn.get("checks", []):
                counts[check.get("rule", "unknown")] += 1
    return counts


def scenario_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(scenario.get("name", "")): scenario
        for scenario in report.get("scenarios", [])
    }


def scenario_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios_by_name = scenario_lookup(report)
    rows: list[dict[str, Any]] = []
    for result in report.get("results", []):
        scenario = scenarios_by_name.get(str(result.get("name", "")), {})
        workload_metadata = dict(scenario.get("workload_metadata", {}))
        traces = result.get("traces", [])
        lats = [float(t.get("latency_ms", 0.0)) for t in traces]
        comp = [int(t.get("completion_tokens", 0)) for t in traces]
        prompt = [int(t.get("prompt_tokens") or 0) for t in traces]
        checks = [c for t in traces for c in t.get("checks", [])]
        latency_total_ms = sum(lats)
        completion_tokens_total = sum(comp)
        completion_tokens_max = max(comp, default=0)
        max_prompt_tokens = max(prompt, default=0)
        first_prompt_tokens = prompt[0] if prompt else 0
        final_prompt_tokens = prompt[-1] if prompt else 0
        rows.append(
            {
                "name": result["name"],
                "description": result.get("description", ""),
                "family": str(scenario.get("benchmark_family", "—")),
                "reasoning_class": str(workload_metadata.get("reasoning_class", "—")),
                "context_growth_profile": str(workload_metadata.get("context_growth_profile", "—")),
                "decode_stress": str(workload_metadata.get("decode_stress", "—")),
                "expected_response_size_class": str(
                    workload_metadata.get("expected_response_size_class", "—")
                ),
                "passed": result.get("passed", False),
                "score_pct": result.get("score_pct", 0.0),
                "turns": len(traces),
                "checks": len(checks),
                "failures": sum(1 for c in checks if not c.get("passed")),
                "latency_total_ms": latency_total_ms,
                "latency_avg_ms": safe_div(latency_total_ms, len(lats)),
                "completion_tokens_total": completion_tokens_total,
                "completion_tokens_max": completion_tokens_max,
                "completion_tokens_avg": safe_div(completion_tokens_total, len(comp)),
                "prompt_tokens_avg": safe_div(sum(prompt), len(prompt)),
                "prompt_tokens_first": first_prompt_tokens,
                "prompt_tokens_final": final_prompt_tokens,
                "prompt_tokens_max": max_prompt_tokens,
                "prompt_growth": max(0, final_prompt_tokens - first_prompt_tokens),
                "ms_per_completion_token": safe_div(latency_total_ms, completion_tokens_total),
                "completion_tokens_per_second": safe_div(completion_tokens_total * 1000.0, latency_total_ms),
            }
        )
    return rows


def family_rows(scen_rows: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_to_family = {str(row["name"]): str(row["family"]) for row in scen_rows}
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scenario_count": 0,
            "turns": 0,
            "latency_total_ms": 0.0,
            "completion_tokens_total": 0,
            "completion_tokens_max": 0,
            "completion_token_values": [],
            "prompt_tokens_total": 0,
            "prompt_tokens_max": 0,
            "prompt_token_values": [],
            "prompt_growth_max": 0,
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "checks": 0,
            "failures": 0,
            "decode_stress": Counter(),
            "context_growth_profile": Counter(),
            "expected_response_size_class": Counter(),
        }
    )
    for row in scen_rows:
        group = grouped[row["family"]]
        group["scenario_count"] += 1
        group["turns"] += row["turns"]
        group["latency_total_ms"] += row["latency_total_ms"]
        group["completion_tokens_total"] += row["completion_tokens_total"]
        group["completion_tokens_max"] = max(group["completion_tokens_max"], row["completion_tokens_max"])
        group["prompt_tokens_total"] += row["prompt_tokens_avg"] * row["turns"]
        group["prompt_tokens_max"] = max(group["prompt_tokens_max"], row["prompt_tokens_max"])
        group["prompt_growth_max"] = max(group["prompt_growth_max"], row["prompt_growth"])
        group["passed_scenarios"] += 1 if row["passed"] else 0
        group["failed_scenarios"] += 0 if row["passed"] else 1
        group["checks"] += row["checks"]
        group["failures"] += row["failures"]
        group["decode_stress"].update([row["decode_stress"]])
        group["context_growth_profile"].update([row["context_growth_profile"]])
        group["expected_response_size_class"].update([row["expected_response_size_class"]])

    for turn in turns:
        family = scenario_to_family.get(str(turn["scenario"]))
        if not family:
            continue
        group = grouped[family]
        group["completion_token_values"].append(int(turn.get("completion_tokens") or 0))
        group["prompt_token_values"].append(int(turn.get("prompt_tokens") or 0))

    rows: list[dict[str, Any]] = []
    for family, payload in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "scenario_count": payload["scenario_count"],
                "turns": payload["turns"],
                "latency_total_ms": payload["latency_total_ms"],
                "latency_avg_ms": safe_div(payload["latency_total_ms"], payload["turns"]),
                "completion_tokens_total": payload["completion_tokens_total"],
                "completion_tokens_avg": safe_div(payload["completion_tokens_total"], payload["turns"]),
                "completion_tokens_median": (
                    statistics.median(payload["completion_token_values"])
                    if payload["completion_token_values"]
                    else 0.0
                ),
                "completion_tokens_max": payload["completion_tokens_max"],
                "prompt_tokens_avg": safe_div(payload["prompt_tokens_total"], payload["turns"]),
                "prompt_tokens_median": (
                    statistics.median(payload["prompt_token_values"])
                    if payload["prompt_token_values"]
                    else 0.0
                ),
                "prompt_tokens_max": payload["prompt_tokens_max"],
                "prompt_growth_max": payload["prompt_growth_max"],
                "passed_scenarios": payload["passed_scenarios"],
                "failed_scenarios": payload["failed_scenarios"],
                "checks": payload["checks"],
                "failures": payload["failures"],
                "passed_checks": payload["checks"] - payload["failures"],
                "failed_checks": payload["failures"],
                "decode_stress": payload["decode_stress"].most_common(1)[0][0] if payload["decode_stress"] else "—",
                "context_growth_profile": (
                    payload["context_growth_profile"].most_common(1)[0][0]
                    if payload["context_growth_profile"]
                    else "—"
                ),
                "expected_response_size_class": (
                    payload["expected_response_size_class"].most_common(1)[0][0]
                    if payload["expected_response_size_class"]
                    else "—"
                ),
                "completion_tokens_per_second": safe_div(
                    payload["completion_tokens_total"] * 1000.0,
                    payload["latency_total_ms"],
                ),
            }
        )
    return rows


def turn_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report.get("results", []):
        for trace in result.get("traces", []):
            checks = trace.get("checks", [])
            rows.append(
                {
                    "scenario": result["name"],
                    "index": trace.get("index"),
                    "user_message": trace.get("user_message", ""),
                    "assistant_message": trace.get("assistant_message", ""),
                    "latency_ms": float(trace.get("latency_ms", 0.0)),
                    "prompt_tokens": trace.get("prompt_tokens"),
                    "completion_tokens": trace.get("completion_tokens"),
                    "checks_passed": sum(1 for c in checks if c.get("passed")),
                    "checks_total": len(checks),
                    "stop_reason": trace.get("stop_reason", ""),
                    "checks": checks,
                }
            )
    return rows


def escape(text: Any) -> str:
    return html.escape(str(text))


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{escape(text)}</span>'


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{escape(sub)}</div>' if sub else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        f"{sub_html}"
        "</div>"
    )


def bar_cell(value: float, max_value: float, text: str) -> str:
    width = 0 if max_value <= 0 else int((value / max_value) * 100)
    return (
        '<div class="bar-wrap">'
        f'<div class="bar" style="width:{width}%"></div>'
        f'<span class="bar-text">{escape(text)}</span>'
        "</div>"
    )


def expandable_message_cell(text: str, *, preview: str, kind: str) -> str:
    preview_box = f'<div class="table-message-box {kind}">{escape(preview)}</div>'
    if preview == text:
        return preview_box
    return (
        f'<details class="table-message-box table-message-detail {kind}">'
        '<summary class="table-message-summary">'
        f'<span class="table-message-preview">{escape(preview)}</span>'
        f'<span class="table-message-full">{escape(text)}</span>'
        '<span class="table-message-hint table-message-hint-expand">Click to expand</span>'
        '<span class="table-message-hint table-message-hint-collapse">Click to collapse</span>'
        "</summary>"
        "</details>"
    )


def build_single_run_html(report: dict[str, Any], source_path: Path) -> str:
    summary = report["summary"]
    config = report["config"]
    meta = report["meta"]
    rules = rule_counts(report)
    scen_rows = scenario_rows(report)
    turns = turn_rows(report)
    fam_rows = family_rows(scen_rows, turns)
    prompt_tokens_total = sum((row["prompt_tokens"] or 0) for row in turns)

    max_scen_lat = max((row["latency_total_ms"] for row in scen_rows), default=0.0)
    max_scen_avg_lat = max((row["latency_avg_ms"] for row in scen_rows), default=0.0)
    max_scen_comp = max((row["completion_tokens_total"] for row in scen_rows), default=0)
    max_scen_comp_avg = max((row["completion_tokens_avg"] for row in scen_rows), default=0.0)
    max_scen_comp_max = max((row["completion_tokens_max"] for row in scen_rows), default=0)
    max_scen_prompt_avg = max((row["prompt_tokens_avg"] for row in scen_rows), default=0.0)
    max_scen_prompt = max((row["prompt_tokens_max"] for row in scen_rows), default=0)
    max_scen_prompt_growth = max((row["prompt_growth"] for row in scen_rows), default=0)
    max_scen_ms_per_token = max((row["ms_per_completion_token"] for row in scen_rows), default=0.0)
    max_turn_lat = max((row["latency_ms"] for row in turns), default=0.0)
    max_turn_prompt = max(((row["prompt_tokens"] or 0) for row in turns), default=0)
    max_turn_completion = max((row["completion_tokens"] or 0 for row in turns), default=0)

    summary_cards = "".join(
        [
            metric_card("Model", escape(config.get("base_model", "—")), config.get("resolved_renderer_name", "—")),
            metric_card("Overall score", f"{summary.get('score_pct', 0):.2f}%", f"{summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} checks passed"),
            metric_card("Total latency", format_ms(summary.get("total_latency_ms")), "all turns"),
            metric_card("Input total", format_num(prompt_tokens_total), "prompt tokens across all turns"),
            metric_card("Output total", format_num(summary.get("total_completion_tokens")), "completion tokens across all turns"),
            metric_card("Fingerprint", f"<code>{escape(meta.get('workload_fingerprint', '')[:12])}…</code>", "workload id"),
        ]
    )

    family_table_rows = ""
    for row in fam_rows:
        smoke_summary = f"Checks: {row['passed_checks']}/{row['checks']} passed"
        family_table_rows += (
            "<tr>"
            f"<td><div class='name'>{escape(row['family'])}</div>"
            f"<div class='muted'>{escape(row['decode_stress'])} decode • {escape(row['context_growth_profile'])} context • {escape(row['expected_response_size_class'])} outputs</div></td>"
            f"<td>{format_num(row['scenario_count'])}</td>"
            f"<td>{format_num(row['turns'])}</td>"
            f"<td>{bar_cell(row['latency_total_ms'], max_scen_lat, format_ms(row['latency_total_ms']))}</td>"
            f"<td>{format_ms(row['latency_avg_ms'])}</td>"
            f"<td>{format_num(row['completion_tokens_total'])}</td>"
            f"<td>{format_num(row['completion_tokens_avg'])}</td>"
            f"<td>{format_num(row['completion_tokens_median'])}</td>"
            f"<td>{format_num(row['completion_tokens_max'])}</td>"
            f"<td>{format_num(row['prompt_tokens_avg'])}</td>"
            f"<td>{format_num(row['prompt_tokens_median'])}</td>"
            f"<td>{format_num(row['prompt_tokens_max'])}</td>"
            f"<td>{row['completion_tokens_per_second']:.2f} tok/s</td>"
            f"<td>{escape(smoke_summary)}</td>"
            "</tr>"
        )

    scenario_table_rows = ""
    for row in scen_rows:
        ms_per_token_text = f"{row['ms_per_completion_token']:.2f} ms/tok"
        scenario_table_rows += (
            "<tr>"
            f"<td><div class='name'>{escape(row['name'])}</div>"
            f"<div class='muted'>{escape(row['family'])} • {row['turns']} turns • {escape(row['decode_stress'])} decode • {escape(row['expected_response_size_class'])} outputs</div>"
            f"<div class='muted'>{escape(truncate(row['description'], 110))}</div></td>"
            f"<td>{bar_cell(row['latency_total_ms'], max_scen_lat, format_ms(row['latency_total_ms']))}</td>"
            f"<td>{bar_cell(row['latency_avg_ms'], max_scen_avg_lat, format_ms(row['latency_avg_ms']))}</td>"
            f"<td>{bar_cell(float(row['completion_tokens_total']), float(max_scen_comp or 1), format_num(row['completion_tokens_total']))}</td>"
            f"<td>{bar_cell(row['completion_tokens_avg'], max_scen_comp_avg or 1.0, format_num(row['completion_tokens_avg']))}</td>"
            f"<td>{bar_cell(float(row['completion_tokens_max']), float(max_scen_comp_max or 1), format_num(row['completion_tokens_max']))}</td>"
            f"<td>{bar_cell(row['prompt_tokens_avg'], max_scen_prompt_avg or 1.0, format_num(row['prompt_tokens_avg']))}</td>"
            f"<td>{bar_cell(float(row['prompt_tokens_max']), float(max_scen_prompt or 1), format_num(row['prompt_tokens_max']))}</td>"
            f"<td>{bar_cell(float(row['prompt_growth']), float(max_scen_prompt_growth or 1), format_num(row['prompt_growth']))}</td>"
            f"<td>{bar_cell(row['ms_per_completion_token'], max_scen_ms_per_token or 1.0, ms_per_token_text)}</td>"
            f"<td><div>{badge('PASS', 'pass') if row['passed'] else badge('FAIL', 'fail')}</div><div class='muted'>{row['score_pct']:.2f}% score</div></td>"
            "</tr>"
        )

    rule_rows = ""
    total_rule_count = sum(rules.values()) or 1
    max_rule_count = max(rules.values(), default=0)
    for rule, count in rules.most_common():
        rule_rows += (
            "<tr>"
            f"<td><code>{escape(rule)}</code></td>"
            f"<td>{bar_cell(count, max_rule_count, str(count))}</td>"
            f"<td>{pct(count, total_rule_count):.2f}%</td>"
            "</tr>"
        )

    turn_groups_html = ""
    turns_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in turns:
        turns_by_scenario.setdefault(str(row["scenario"]), []).append(row)

    for scenario_name, scenario_turns in turns_by_scenario.items():
        passed_checks = sum(int(item["checks_passed"]) for item in scenario_turns)
        total_checks = sum(int(item["checks_total"]) for item in scenario_turns)
        group_rows_html = ""
        for row in scenario_turns:
            user_preview = truncate_block(row["user_message"], char_limit=360, line_limit=6)
            assistant_preview = truncate_block(row["assistant_message"], char_limit=520, line_limit=10)
            user_message_html = expandable_message_cell(
                str(row["user_message"]),
                preview=user_preview,
                kind="user",
            )
            assistant_message_html = expandable_message_cell(
                str(row["assistant_message"]),
                preview=assistant_preview,
                kind="assistant",
            )
            checks_html = "".join(
                "<li>"
                f"{badge('pass', 'pass') if c.get('passed') else badge('fail', 'fail')} "
                f"<code>{escape(c.get('name', ''))}</code> — {escape(c.get('details', ''))}"
                "</li>"
                for c in row["checks"]
            )
            group_rows_html += (
                "<tr>"
                f"<td><div class='name'>Turn {escape(row['index'])}</div><div class='muted'>{escape(scenario_name)}</div></td>"
                f"<td>{bar_cell(row['latency_ms'], max_turn_lat, format_ms(row['latency_ms']))}</td>"
                f"<td>{bar_cell(float(row['prompt_tokens'] or 0), float(max_turn_prompt or 1), format_num(row['prompt_tokens']))}</td>"
                f"<td>{bar_cell(float(row['completion_tokens'] or 0), float(max_turn_completion or 1), format_num(row['completion_tokens']))}</td>"
                f"<td>{row['checks_passed']}/{row['checks_total']}</td>"
                f"<td>{escape(row['stop_reason'])}</td>"
                f"<td>{user_message_html}</td>"
                f"<td>{assistant_message_html}</td>"
                f"<td><details class='turn-checks'><summary>Checks</summary><ul class='check-list'>{checks_html}</ul></details></td>"
                "</tr>"
            )

        turn_groups_html += (
            "<details class='turn-scenario-detail'>"
            f"<summary><strong>{escape(scenario_name)}</strong> "
            f"<span class='scenario-summary-meta'>{len(scenario_turns)} turns • {passed_checks}/{total_checks} checks passed</span></summary>"
            "<table class='turn-group-table'>"
            "<thead><tr>"
            "<th>Turn</th>"
            "<th>Latency</th>"
            "<th>Prompt tokens</th>"
            "<th>Completion tokens</th>"
            "<th>Checks</th>"
            "<th>Stop</th>"
            "<th>User message</th>"
            "<th>Assistant message</th>"
            "<th>Check details</th>"
            "</tr></thead>"
            f"<tbody>{group_rows_html}</tbody>"
            "</table>"
            "</details>"
        )

    scenario_details = ""
    for result in report["results"]:
        traces = result.get("traces", [])
        passed_checks = sum(
            1 for trace in traces for check in trace.get("checks", []) if check.get("passed")
        )
        total_checks = sum(len(trace.get("checks", [])) for trace in traces)
        scenario_details += (
            "<details class='scenario-detail'>"
            f"<summary><strong>{escape(result['name'])}</strong> — {escape(result.get('description', ''))}"
            f" <span class='scenario-summary-meta'>{escape(str(len(traces)))} turns • {escape(str(passed_checks))}/{escape(str(total_checks))} checks passed</span></summary>"
            "<div class='chat-thread'>"
        )
        for trace in traces:
            passed = sum(1 for check in trace.get("checks", []) if check.get("passed"))
            total = len(trace.get("checks", []))
            user_full = str(trace.get("user_message", ""))
            assistant_full = str(trace.get("assistant_message", ""))
            checks_html = "".join(
                "<tr>"
                f"<td><code>{escape(check.get('name', ''))}</code></td>"
                f"<td><code>{escape(check.get('rule', ''))}</code></td>"
                f"<td>{badge('PASS', 'pass') if check.get('passed') else badge('FAIL', 'fail')}</td>"
                f"<td>{escape(check.get('details', ''))}</td>"
                "</tr>"
                for check in trace.get("checks", [])
            )
            scenario_details += (
                "<div class='chat-turn'>"
                "<div class='chat-message-row user'>"
                "<div class='chat-avatar user'>U</div>"
                "<div class='chat-message user'>"
                "<div class='chat-role'>User</div>"
                f"<div class='chat-content'>{escape(user_full)}</div>"
                "</div>"
                "</div>"
                "<div class='chat-message-row assistant'>"
                "<div class='chat-avatar assistant'>A</div>"
                "<div class='chat-message assistant'>"
                "<div class='chat-role'>Assistant</div>"
                f"<div class='chat-content'>{escape(assistant_full)}</div>"
                "</div>"
                "</div>"
                f"<div class='chat-checks-head'>Contract checks: {passed}/{total} passed</div>"
                "<table class='chat-checks-table'><thead><tr><th>Check</th><th>Rule</th><th>Status</th><th>Details</th></tr></thead>"
                f"<tbody>{checks_html}</tbody></table>"
                "</div>"
            )
        scenario_details += "</div></details>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conversation Eval Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --bg: #f6f8fc;
  --panel: #ffffff;
  --panel-2: #f5f8fd;
  --text: #182233;
  --muted: #5f6f85;
  --border: #d8e1ef;
  --pass: #e7f5ec;
  --pass-text: #1d5b41;
  --fail: #fff1ee;
  --fail-text: #8b433d;
  --bar: #d9e8fb;
  --accent: #c3daf8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  background:
    radial-gradient(circle at top left, #ffffff 0%, #f6f8fc 46%, #f3f6fb 100%);
  color: var(--text);
  line-height: 1.45;
  overflow-x: hidden;
}}
main {{ width: 100%; max-width: none; margin: 0; padding: 28px clamp(14px, 2.4vw, 32px); }}
h1, h2, h3 {{ margin: 0 0 12px 0; }}
p, li {{ color: var(--text); }}
small, .muted {{ color: var(--muted); }}
header {{
  display: grid;
  gap: 18px;
  margin-bottom: 32px;
}}
.subtitle {{ color: var(--muted); max-width: 1000px; }}
.section {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 30px;
  box-shadow: 0 10px 24px rgba(30, 48, 78, 0.04);
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 18px;
}}
.metric-card {{
  background: linear-gradient(180deg, #fbfcff 0%, var(--panel-2) 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px;
  min-height: 108px;
}}
.metric-label {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 8px; }}
.metric-value {{ font-size: 1.4rem; font-weight: 700; overflow-wrap: anywhere; }}
.metric-sub {{ color: var(--muted); font-size: 0.9rem; margin-top: 6px; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 14px;
}}
th, td {{
  text-align: left;
  padding: 16px 14px;
  border-top: 1px solid var(--border);
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 600;
  background: #f8fbff;
}}
.name {{ font-weight: 700; margin-bottom: 4px; }}
.badge {{
  display: inline-block;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}}
.pass {{ background: var(--pass); color: var(--pass-text); }}
.fail {{ background: var(--fail); color: var(--fail-text); }}
.bar-wrap {{
  position: relative;
  min-width: 150px;
  background: #fdfefe;
  border: 1px solid var(--border);
  border-radius: 999px;
  height: 28px;
  overflow: hidden;
}}
.bar {{
  height: 100%;
  background: linear-gradient(90deg, var(--bar), var(--accent));
  opacity: 1;
}}
.bar-text {{
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  padding: 0 10px;
  font-weight: 600;
  color: var(--text);
}}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  background: #f8fbff;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  margin: 8px 0 0 0;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  background: #f2f6fd;
  padding: 2px 4px;
  border-radius: 6px;
}}
details {{
  border-radius: 10px;
}}
summary {{
  cursor: pointer;
}}
.check-list {{
  margin: 10px 0 0 0;
  padding-left: 20px;
}}
.check-list li {{
  margin: 0 0 10px 0;
  line-height: 1.6;
}}
.check-list li:last-child {{
  margin-bottom: 0;
}}
.section-details summary {{
  list-style: none;
}}
.section-details summary::-webkit-details-marker {{
  display: none;
}}
.section-summary {{
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text);
}}
.section-details[open] summary {{
  margin-bottom: 12px;
}}
.scenario-detail {{
  display: block;
  margin-top: 28px;
  border: 1px solid var(--border);
  padding: 24px;
  border-radius: 14px;
  background: var(--panel-2);
}}
.scenario-detail > summary {{
  padding-bottom: 4px;
}}
.scenario-detail[open] > summary {{
  margin-bottom: 18px;
}}
.scenario-summary-meta {{
  color: var(--muted);
  font-weight: 500;
  font-size: 0.92rem;
}}
.section-controls {{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin: 14px 0 18px;
}}
.section-control-button {{
  appearance: none;
  border: 1px solid var(--border);
  background: #f8fbff;
  color: var(--text);
  border-radius: 999px;
  padding: 8px 14px;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}}
.section-control-button:hover {{
  background: #eef4fd;
}}
.chat-thread {{
  display: grid;
  gap: 24px;
  margin-top: 22px;
}}
.chat-turn {{
  display: grid;
  gap: 10px;
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
}}
.chat-turn + .chat-turn {{
  padding-top: 24px;
}}
.chat-message-row {{
  display: flex;
  gap: 16px;
  align-items: flex-start;
}}
.chat-message-row.user {{
  justify-content: flex-end;
  padding-left: 18%;
}}
.chat-message-row.assistant,
.chat-reply-summary {{
  justify-content: flex-start;
  padding-right: 18%;
}}
.chat-message-row.user .chat-avatar {{
  order: 2;
}}
.chat-message-row.user .chat-message {{
  order: 1;
}}
.chat-reply-detail {{
  display: block;
}}
.chat-reply-summary {{
  list-style: none;
}}
.chat-reply-summary::-webkit-details-marker {{
  display: none;
}}
.chat-reply-summary .chat-message {{
  transition: box-shadow 140ms ease, transform 140ms ease, background 140ms ease, border-color 140ms ease;
}}
.chat-reply-summary:hover .chat-message.assistant {{
  background: #f8fbff;
  border-color: #cbd9ed;
  box-shadow: 0 8px 18px rgba(31, 55, 88, 0.08);
}}
.chat-reply-detail[open] .chat-message.assistant {{
  background: #f8fbff;
  border-color: #cbd9ed;
  box-shadow: 0 8px 18px rgba(31, 55, 88, 0.08);
}}
.chat-avatar {{
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.82rem;
  font-weight: 700;
  flex: 0 0 auto;
}}
.chat-avatar.user {{
  display: none;
}}
.chat-avatar.assistant {{
  background: #eef4fd;
  color: #203758;
}}
.chat-message {{
  max-width: min(78%, 820px);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 1px 0 rgba(0, 0, 0, 0.04);
}}
.chat-message.user {{
  background: #203758;
  color: #ffffff;
  border-bottom-right-radius: 6px;
}}
.chat-message.assistant {{
  background: #ffffff;
  color: var(--text);
  border-bottom-left-radius: 6px;
  box-shadow: 0 4px 14px rgba(31, 55, 88, 0.05);
}}
.chat-message.user .chat-role {{
  color: rgba(255, 255, 255, 0.72);
}}
.chat-role {{
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 11px;
}}
.chat-content {{
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.6;
}}
.chat-checks-head {{
  margin: 10px 0 0 50px;
  color: var(--muted);
  font-weight: 600;
  font-size: 0.92rem;
}}
.chat-expanded-block {{
  margin: 12px 0 0 50px;
}}
.chat-expanded-label {{
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  margin-bottom: 8px;
}}
.chat-expanded-content {{
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.6;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #fcfdff;
  padding: 14px 16px;
}}
.chat-checks-table {{
  margin: 12px 0 0 50px;
}}
.table-message-box {{
  min-width: 240px;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.55;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  background: #fbfdff;
}}
.table-message-box.user {{
  background: #fdfefe;
}}
	.table-message-box.assistant {{
	  background: #f4f8ff;
	}}
	.table-message-detail {{
	  min-width: 240px;
	  padding: 0;
	}}
	.table-message-summary {{
	  display: block;
	  list-style: none;
	  cursor: pointer;
	  padding: 12px 14px;
	}}
	.table-message-summary::-webkit-details-marker {{
	  display: none;
	}}
	.table-message-preview,
	.table-message-full {{
	  display: block;
	  white-space: pre-wrap;
	  word-break: break-word;
	  line-height: 1.55;
	}}
	.table-message-full {{
	  display: none;
	}}
	.table-message-detail[open] .table-message-preview {{
	  display: none;
	}}
	.table-message-detail[open] .table-message-full {{
	  display: block;
	}}
	.table-message-hint {{
	  display: inline-block;
	  margin-top: 8px;
	  color: var(--muted);
	  font-size: 0.82rem;
	  font-weight: 600;
	}}
	.table-message-hint-collapse {{
	  display: none;
	}}
	.table-message-detail[open] .table-message-hint-expand {{
	  display: none;
	}}
	.table-message-detail[open] .table-message-hint-collapse {{
	  display: inline-block;
	}}
	.turn-scenario-detail {{
	  display: block;
	  margin-top: 18px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: #fcfdff;
  padding: 16px 18px;
}}
.turn-scenario-detail > summary {{
  font-size: 1rem;
  font-weight: 700;
}}
.turn-scenario-detail[open] > summary {{
  margin-bottom: 14px;
}}
.turn-group-table {{
  margin-top: 0;
}}
ul.tight {{ margin: 8px 0 0 0; padding-left: 18px; }}
.grid-2 {{
  display: grid;
  grid-template-columns: 1.3fr 0.7fr;
  gap: 24px;
}}
.benchmark-detail-grid {{
  grid-template-columns: 1fr;
}}
.turn-table-section {{
  max-width: 100%;
  overflow-x: auto;
}}
.turn-table-section .bar-wrap {{
  min-width: 0;
  width: auto;
  height: auto;
  border: 0;
  background: transparent;
  border-radius: 0;
  overflow: visible;
}}
.turn-table-section .bar {{
  display: none;
}}
.turn-table-section .bar-text {{
  position: static;
  display: inline;
  padding: 0;
}}
.footer {{ color: var(--muted); font-size: 0.9rem; margin-top: 20px; }}
@media (max-width: 980px) {{
  .grid-2 {{ grid-template-columns: 1fr; }}
  .chat-message {{
    max-width: 100%;
  }}
  .chat-message-row,
  .chat-message-row.user,
  .chat-message-row.assistant,
  .chat-reply-summary {{
    justify-content: flex-start;
  }}
  .chat-message-row.user,
  .chat-message-row.assistant,
  .chat-reply-summary {{
    padding-left: 0;
    padding-right: 0;
  }}
  .chat-message-row.user .chat-avatar,
  .chat-message-row.user .chat-message {{
    order: initial;
  }}
  .chat-checks-head,
  .chat-expanded-block,
  .chat-checks-table {{
    margin-left: 0;
    padding-left: 0;
  }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>Conversation evaluation dashboard</h1>
  <div class="subtitle">
    This view separates the fast benchmark summary from the deeper replay/debug trace. It is built from
    <code>{escape(source_path.name)}</code> and is meant to answer two questions quickly: “did this run look good?”
    and “why did it score that way?”
  </div>
</header>

<section class="section">
  <h2>Run summary</h2>
  <div class="metrics">{summary_cards}</div>
  <div class="footer">
    Run ID <code>{escape(meta.get('run_id', '—'))}</code> • started {escape(meta.get('started_at', '—'))} • finished {escape(meta.get('finished_at', '—'))}
  </div>
</section>

<div class="grid-2">
  <section class="section">
    <h2>Family performance</h2>
    <table>
      <thead>
        <tr>
          <th>Family</th>
          <th>Scenarios</th>
          <th>Turns</th>
          <th>Total latency</th>
          <th>Avg / turn</th>
          <th>Completion tokens</th>
          <th>Avg response tokens</th>
          <th>Median response tokens</th>
          <th>Max response tokens</th>
          <th>Avg prompt tokens</th>
          <th>Median prompt tokens</th>
          <th>Max prompt tokens</th>
          <th>Decode rate</th>
          <th>Contract pass rate</th>
        </tr>
      </thead>
      <tbody>{family_table_rows}</tbody>
    </table>
  </section>
</div>

<div class="grid-2 benchmark-detail-grid">
  <section class="section turn-table-section">
    <h2>Turn-level benchmark view</h2>
    {turn_groups_html}
  </section>
</div>

<section class="section">
  <h2>Replay trace</h2>
  <p class="muted">
    Use this section for failure analysis and benchmark authoring. Each scenario can be opened to inspect the
    exact user prompt, assistant reply, and deterministic check outcomes turn by turn.
  </p>
  <div class="section-controls">
    <button type="button" class="section-control-button" id="replay-expand-all">Expand all</button>
    <button type="button" class="section-control-button" id="replay-collapse-all">Collapse all</button>
  </div>
  {scenario_details}
</section>

<section class="section">
  <details class="section-details">
    <summary><span class="section-summary">Rule coverage</span></summary>
    <table>
      <thead>
        <tr><th>Rule</th><th>Count</th><th>Share</th></tr>
      </thead>
      <tbody>{rule_rows}</tbody>
    </table>
    <p class="muted" style="margin-top:12px;">
      Rule coverage matters because it shows whether the suite is testing multiple failure shapes or mostly
      repeating one scoring style.
    </p>
  </details>
</section>
</main>
<script>
document.querySelectorAll("details").forEach((element) => {{
  element.open = false;
}});
document.querySelectorAll(".turn-checks").forEach((element) => {{
  element.open = true;
}});
const firstTurnScenario = document.querySelector(".turn-scenario-detail");
if (firstTurnScenario) {{
  firstTurnScenario.open = true;
}}
const replayScenarioDetails = document.querySelectorAll(".scenario-detail");
const replayExpandAll = document.getElementById("replay-expand-all");
const replayCollapseAll = document.getElementById("replay-collapse-all");
if (replayExpandAll) {{
  replayExpandAll.addEventListener("click", () => {{
    replayScenarioDetails.forEach((element) => {{
      element.open = true;
    }});
  }});
}}
if (replayCollapseAll) {{
  replayCollapseAll.addEventListener("click", () => {{
    replayScenarioDetails.forEach((element) => {{
      element.open = false;
    }});
  }});
}}
</script>
</body>
</html>
"""


def build_multi_run_html(reports: list[tuple[Path, dict[str, Any]]]) -> str:
    summaries = [collect_run_summary(report, path) for path, report in reports]
    all_scenarios = sorted(
        {name for item in summaries for name in item["scenario_scores"].keys()}
    )
    max_score = max((item["score_pct"] for item in summaries), default=0.0)
    max_latency = max((item["total_latency_ms"] for item in summaries), default=0.0)
    max_prompt = max((item["total_prompt_tokens"] for item in summaries), default=0)
    max_completion = max((item["total_completion_tokens"] for item in summaries), default=0)

    rows = ""
    for item in summaries:
        rows += (
            "<tr>"
            f"<td><div class='name'>{escape(item['filename'])}</div><div class='muted'>{escape(item['run_id'])}</div></td>"
            f"<td>{escape(item['model'])}</td>"
            f"<td>{escape(item['renderer'])}</td>"
            f"<td>{bar_cell(item['score_pct'], max_score or 100, f'{item['score_pct']:.2f}%')}</td>"
            f"<td>{item['passed_scenarios']}/{item['scenario_count']}</td>"
            f"<td>{item['passed_checks']}/{item['total_checks']}</td>"
            f"<td>{bar_cell(item['total_latency_ms'], max_latency or 1, format_ms(item['total_latency_ms']))}</td>"
            f"<td>{bar_cell(float(item['total_prompt_tokens']), float(max_prompt or 1), format_num(item['total_prompt_tokens']))}</td>"
            f"<td>{bar_cell(float(item['total_completion_tokens']), float(max_completion or 1), format_num(item['total_completion_tokens']))}</td>"
            "</tr>"
        )

    matrix_head = "".join(f"<th>{escape(name)}</th>" for name in all_scenarios)
    matrix_rows = ""
    for item in summaries:
        cells = "".join(
            f"<td>{item['scenario_scores'].get(name, '—') if name in item['scenario_scores'] else '—'}</td>"
            for name in all_scenarios
        )
        matrix_rows += (
            "<tr>"
            f"<td><div class='name'>{escape(item['filename'])}</div><div class='muted'>{escape(item['model'])}</div></td>"
            f"{cells}</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conversation Eval Comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ margin: 0; background: #0b1020; color: #eef2ff; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
main {{ max-width: 1380px; margin: 0 auto; padding: 24px; }}
.section {{ background: #121933; border: 1px solid #2a3567; border-radius: 16px; padding: 18px; margin-bottom: 18px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px; border-top: 1px solid #2a3567; text-align: left; vertical-align: top; }}
th {{ color: #9aa6d1; }}
.name {{ font-weight: 700; margin-bottom: 4px; }}
.muted {{ color: #9aa6d1; }}
.bar-wrap {{ position: relative; min-width: 150px; background: rgba(255,255,255,0.04); border: 1px solid #2a3567; border-radius: 999px; height: 28px; overflow: hidden; }}
.bar {{ height: 100%; background: linear-gradient(90deg, #5f7cff, #8ca0ff); opacity: 0.78; }}
.bar-text {{ position: absolute; inset: 0; display: flex; align-items: center; padding: 0 10px; font-weight: 600; }}
code {{ background: rgba(255,255,255,0.06); padding: 2px 4px; border-radius: 6px; }}
</style>
</head>
<body>
<main>
<section class="section">
  <h1>Conversation evaluation comparison</h1>
  <p class="muted">
    This comparison view is meant for cross-run benchmarking. It works best when multiple reports share the same
    workload fingerprint and differ only by model, engine, or serving stack.
  </p>
</section>
<section class="section">
  <h2>Run overview</h2>
  <table>
    <thead>
      <tr>
        <th>Run</th>
        <th>Model</th>
        <th>Renderer</th>
        <th>Score</th>
        <th>Scenarios</th>
        <th>Checks</th>
        <th>Total latency</th>
        <th>Total prompt tokens</th>
        <th>Total completion tokens</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>
<section class="section">
  <h2>Scenario score matrix</h2>
  <table>
    <thead>
      <tr><th>Run</th>{matrix_head}</tr>
    </thead>
    <tbody>{matrix_rows}</tbody>
  </table>
</section>
</main>
</body>
</html>
"""


def build_html(reports: list[tuple[Path, dict[str, Any]]]) -> str:
    if len(reports) == 1:
        return build_single_run_html(reports[0][1], reports[0][0])
    return build_multi_run_html(reports)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render one or more conversation_eval JSON reports into an HTML dashboard."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more report files or directories. Directory inputs use the latest matching report.",
    )
    parser.add_argument(
        "--output",
        default="conversation_eval_dashboard.html",
        help="Output HTML file path.",
    )
    args = parser.parse_args()

    report_paths = iter_report_paths(args.inputs)
    reports = [(path, load_report(path)) for path in report_paths]
    html_text = build_html(reports)

    output_path = Path(args.output)
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
