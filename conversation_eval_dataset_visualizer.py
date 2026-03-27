#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark_contracts import CONTRACT_SCHEMA_VERSION, compile_contracts_to_dataset, load_contracts


def _looks_like_contract_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        return str(payload.get("schema_version", "")) == CONTRACT_SCHEMA_VERSION
    if isinstance(payload, list) and payload:
        first = payload[0]
        return isinstance(first, dict) and str(first.get("schema_version", "")) == CONTRACT_SCHEMA_VERSION
    return False


def load_view_source(raw: str) -> tuple[list[dict[str, Any]], Path, str]:
    path = Path(raw)
    if path.is_file():
        payload = json.loads(path.read_text())
        if _looks_like_contract_payload(payload):
            return compile_contracts_to_dataset(load_contracts(path)), path, "contracts"
        if not isinstance(payload, list):
            raise ValueError("Scenario dataset file must contain a top-level JSON list.")
        return payload, path, "dataset"
    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {path}")
        for candidate in json_files:
            payload = json.loads(candidate.read_text())
            if _looks_like_contract_payload(payload):
                return compile_contracts_to_dataset(load_contracts(path)), path, "contracts"
        matched = sorted(path.glob("conversation_eval*.json"))
        if not matched:
            matched = json_files
        dataset_path = max(matched, key=lambda candidate: (candidate.stat().st_mtime, candidate.name))
        payload = json.loads(dataset_path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Scenario dataset file must contain a top-level JSON list: {dataset_path}")
        return payload, dataset_path, "dataset"
    raise FileNotFoundError(f"Input not found: {path}")


def escape(text: Any) -> str:
    return html.escape(str(text))


def truncate(text: str, limit: int = 140) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def format_num(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def pct(a: float, b: float) -> float:
    return round(safe_div(a, b) * 100.0, 2)


def metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{escape(sub)}</div>' if sub else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        f"{sub_html}"
        "</div>"
    )


def family_name(scenario_name: str) -> str:
    parts = scenario_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scenario_name


def scenario_family(scenario: dict[str, Any]) -> str:
    explicit = str(scenario.get("benchmark_family", "")).strip()
    if explicit:
        return explicit
    return family_name(str(scenario.get("name", "")))


def summarize_check_params(check: dict[str, Any]) -> str:
    if "needles" in check:
        needles = check.get("needles") or []
        return f"{len(needles)} needles"
    if "pattern" in check:
        return truncate(str(check.get("pattern", "")), 72)
    if "count" in check:
        return f"count={check['count']}"
    if "max" in check:
        return f"max={check['max']}"
    return "custom rule params"


def summarize_input_contract(input_contract: dict[str, Any] | None) -> str:
    if not input_contract:
        return "—"
    parts: list[str] = []
    task_type = str(input_contract.get("task_type", "")).strip()
    if task_type:
        parts.append(f"task={task_type}")
    if "min_input_character_count" in input_contract:
        parts.append(f"min input chars={input_contract['min_input_character_count']}")
    return " • ".join(parts) if parts else "authored input contract"


def summarize_output_contract(output_contract: dict[str, Any] | None) -> str:
    if not output_contract:
        return "—"
    parts: list[str] = []
    keyword_counts = output_contract.get("keyword_counts") or []
    if keyword_counts:
        parts.append(f"keywords={len(keyword_counts)} exact")
    if "sentence_count" in output_contract:
        parts.append(f"sentences={output_contract['sentence_count']}")
    if "bullet_count" in output_contract:
        parts.append(f"bullets={output_contract['bullet_count']}")
    if "max_line_count" in output_contract:
        parts.append(f"max lines={output_contract['max_line_count']}")
    if "min_line_count" in output_contract:
        parts.append(f"min lines={output_contract['min_line_count']}")
    if "min_character_count" in output_contract:
        parts.append(f"min chars={output_contract['min_character_count']}")
    if "required_section_headers" in output_contract:
        parts.append(f"headers={len(output_contract['required_section_headers'])}")
    return " • ".join(parts) if parts else "authored output contract"


def rule_counts(dataset: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for scenario in dataset:
        for turn in scenario.get("turns", []):
            for check in turn.get("checks", []):
                counts[str(check.get("rule", "unknown"))] += 1
    return counts


def tag_counts(dataset: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for scenario in dataset:
        for tag in scenario.get("tags", []):
            counts[str(tag)] += 1
    return counts


def family_rows(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"scenarios": 0, "turns": 0, "checks": 0, "tags": Counter(), "rules": Counter()}
    )
    for scenario in dataset:
        family = scenario_family(scenario)
        row = grouped[family]
        row["scenarios"] += 1
        turns = scenario.get("turns", [])
        row["turns"] += len(turns)
        row["checks"] += sum(len(turn.get("checks", [])) for turn in turns)
        row["tags"].update(str(tag) for tag in scenario.get("tags", []))
        for turn in turns:
            row["rules"].update(str(check.get("rule", "unknown")) for check in turn.get("checks", []))

    rows: list[dict[str, Any]] = []
    for family, payload in sorted(grouped.items()):
        top_tags = ", ".join(tag for tag, _ in payload["tags"].most_common(3)) or "—"
        top_rules = ", ".join(rule for rule, _ in payload["rules"].most_common(3)) or "—"
        rows.append(
            {
                "family": family,
                "scenarios": payload["scenarios"],
                "turns": payload["turns"],
                "checks": payload["checks"],
                "top_tags": top_tags,
                "top_rules": top_rules,
            }
        )
    return rows


def turn_rows(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in dataset:
        scenario_name = str(scenario.get("name", ""))
        for index, turn in enumerate(scenario.get("turns", []), start=1):
            checks = turn.get("checks", [])
            unique_rules = list(dict.fromkeys(str(check.get("rule", "unknown")) for check in checks))
            rows.append(
                {
                    "family": scenario_family(scenario),
                    "scenario": scenario_name,
                    "index": index,
                    "prompt": str(turn.get("prompt", turn.get("user", ""))),
                    "user_message": str(turn.get("user", "")),
                    "input_summary": summarize_input_contract(turn.get("input_contract")),
                    "output_summary": summarize_output_contract(turn.get("output_contract")),
                    "checks_total": len(checks),
                    "rules_summary": ", ".join(unique_rules),
                    "checks": checks,
                }
            )
    return rows


def scenario_details_html(dataset: list[dict[str, Any]]) -> str:
    html_chunks: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in dataset:
        grouped[scenario_family(scenario)].append(scenario)

    for family, scenarios in sorted(grouped.items()):
        family_turns = sum(len(scenario.get("turns", [])) for scenario in scenarios)
        family_checks = sum(
            len(turn.get("checks", []))
            for scenario in scenarios
            for turn in scenario.get("turns", [])
        )
        family_tags = Counter(
            str(tag)
            for scenario in scenarios
            for tag in scenario.get("tags", [])
        )
        top_tags = "".join(
            f'<span class="pill">{escape(tag)}</span>'
            for tag, _ in family_tags.most_common(4)
        )

        html_chunks.append(
            "<details class='family-detail'>"
            f"<summary><strong>{escape(family)}</strong> — {len(scenarios)} scenarios • {family_turns} turns • {family_checks} checks</summary>"
            f"<div class='family-meta'><div class='pill-row'>{top_tags or '<span class=\"muted\">—</span>'}</div></div>"
        )

        for scenario in scenarios:
            tags_html = "".join(f'<span class="pill">{escape(tag)}</span>' for tag in scenario.get("tags", []))
            facts = scenario.get("facts", {})
            facts_text = "\n".join(f"{key}: {value}" for key, value in facts.items()) or "—"
            metadata_text = json.dumps(
                scenario.get("workload_metadata", {}),
                indent=2,
                ensure_ascii=True,
            ) if scenario.get("workload_metadata") else "—"
            contract_source = str(scenario.get("contract_source", "")).strip() or "—"

            html_chunks.append(
                "<details class='scenario-detail'>"
                f"<summary><strong>{escape(scenario.get('name', ''))}</strong> — {escape(scenario.get('description', ''))}</summary>"
                f"<div class='trace-block'><div class='trace-label'>System prompt</div><pre>{escape(scenario.get('system_prompt', ''))}</pre></div>"
                f"<div class='trace-block'><div class='trace-label'>Facts</div><pre>{escape(facts_text)}</pre></div>"
                f"<div class='trace-block'><div class='trace-label'>Workload metadata</div><pre>{escape(metadata_text)}</pre></div>"
                f"<div class='trace-block'><div class='trace-label'>Contract source</div><pre>{escape(contract_source)}</pre></div>"
                f"<div class='trace-block'><div class='trace-label'>Tags</div><div class='pill-row'>{tags_html or '<span class=\"muted\">—</span>'}</div></div>"
            )

            for index, turn in enumerate(scenario.get("turns", []), start=1):
                checks = turn.get("checks", [])
                input_contract = turn.get("input_contract")
                output_contract = turn.get("output_contract")
                correctness_checks = turn.get("correctness_checks", [])
                generated_checks = turn.get("generated_checks", [])
                check_rows = "".join(
                    "<tr>"
                    f"<td><code>{escape(check.get('name', ''))}</code></td>"
                    f"<td><code>{escape(check.get('rule', ''))}</code></td>"
                    f"<td>{escape(summarize_check_params(check))}</td>"
                    f"<td>{escape(check.get('description', '')) or '—'}</td>"
                    "</tr>"
                    for check in checks
                )
                rules_summary = ", ".join(
                    list(dict.fromkeys(str(check.get("rule", "unknown")) for check in checks))
                )
                html_chunks.append(
                    (
                        "<div class='trace-card'>"
                        f"<div class='trace-head'>Turn {index} • {len(checks)} checks • {escape(rules_summary)}</div>"
                        f"<div class='trace-block'><div class='trace-label'>Input shape</div><pre>{escape(summarize_input_contract(input_contract))}</pre></div>"
                        f"<div class='trace-block'><div class='trace-label'>Output shape</div><pre>{escape(summarize_output_contract(output_contract))}</pre></div>"
                        f"<div class='trace-block'><div class='trace-label'>Authored prompt</div><pre>{escape(turn.get('prompt', turn.get('user', '')))}</pre></div>"
                        f"<div class='trace-block'><div class='trace-label'>User</div><pre>{escape(turn.get('user', ''))}</pre></div>"
                    )
                    + (
                        f"<div class='trace-block'><div class='trace-label'>Input contract</div><pre>{escape(json.dumps(input_contract, indent=2, ensure_ascii=True))}</pre></div>"
                        if input_contract
                        else ""
                    )
                    + (
                        f"<div class='trace-block'><div class='trace-label'>Output contract</div><pre>{escape(json.dumps(output_contract, indent=2, ensure_ascii=True))}</pre></div>"
                        if output_contract
                        else ""
                    )
                    + (
                        f"<div class='trace-block'><div class='trace-label'>Authored correctness checks</div><pre>{escape(json.dumps(correctness_checks, indent=2, ensure_ascii=True))}</pre></div>"
                        if correctness_checks
                        else ""
                    )
                    + (
                        f"<div class='trace-block'><div class='trace-label'>Generated contract checks</div><pre>{escape(json.dumps(generated_checks, indent=2, ensure_ascii=True))}</pre></div>"
                        if generated_checks
                        else ""
                    )
                    + (
                        "<table><thead><tr><th>Check</th><th>Rule</th><th>Key params</th><th>Description</th></tr></thead>"
                        f"<tbody>{check_rows}</tbody></table>"
                        "</div>"
                    )
                )

            html_chunks.append("</details>")

        html_chunks.append("</details>")

    return "".join(html_chunks)


def grouped_turn_rows_html(turns: list[dict[str, Any]]) -> str:
    html_chunks: list[str] = []
    current_family = None
    for row in turns:
        if row["family"] != current_family:
            current_family = row["family"]
            html_chunks.append(
                "<tr class='group-row'>"
                f"<td colspan='6'>{escape(current_family)}</td>"
                "</tr>"
            )
        checks_html = "".join(
            "<li>"
            f"<code>{escape(check.get('name', ''))}</code> — "
            f"<code>{escape(check.get('rule', ''))}</code> — {escape(summarize_check_params(check))}"
            "</li>"
            for check in row["checks"]
        )
        html_chunks.append(
            "<tr>"
            f"<td><div class='name'>{escape(row['scenario'])}</div><div class='muted'>Turn {row['index']}</div></td>"
            f"<td>{format_num(row['checks_total'])}</td>"
            f"<td>{escape(truncate(row['rules_summary'], 80))}</td>"
            f"<td><div class='muted'>{escape(row['input_summary'])}</div><div style='margin-top:6px'>{escape(row['output_summary'])}</div></td>"
            f"<td><details><summary>{escape(truncate(row['user_message'], 110))}</summary><pre>{escape(row['user_message'])}</pre></details></td>"
            f"<td><details><summary>Checks</summary><ul class='check-list'>{checks_html}</ul></details></td>"
            "</tr>"
        )
    return "".join(html_chunks)


def build_html(dataset: list[dict[str, Any]], source_path: Path, source_kind: str) -> str:
    total_turns = sum(len(scenario.get("turns", [])) for scenario in dataset)
    total_checks = sum(len(turn.get("checks", [])) for scenario in dataset for turn in scenario.get("turns", []))
    rules = rule_counts(dataset)
    tags = tag_counts(dataset)
    families = family_rows(dataset)
    turns = turn_rows(dataset)

    summary_cards = "".join(
        [
            metric_card("Dataset file", escape(source_path.name), "scenario source"),
            metric_card("Source type", escape(source_kind), "authored contracts or derived dataset"),
            metric_card("Scenario families", format_num(len(families)), "unique families"),
            metric_card("Scenarios", format_num(len(dataset)), "total scenarios"),
            metric_card("Turns", format_num(total_turns), "all scenarios"),
            metric_card("Checks", format_num(total_checks), "all turns"),
            metric_card("Unique rules", format_num(len(rules)), "rule types"),
            metric_card("Unique tags", format_num(len(tags)), "tag labels"),
            metric_card("Avg checks / turn", f"{safe_div(total_checks, total_turns):.2f}", "dataset density"),
        ]
    )

    family_table_rows = ""
    for row in families:
        family_table_rows += (
            "<tr>"
            f"<td><div class='name'>{escape(row['family'])}</div><div class='muted'>{escape(row['top_tags'])}</div></td>"
            f"<td>{format_num(row['scenarios'])}</td>"
            f"<td>{format_num(row['turns'])}</td>"
            f"<td>{format_num(row['checks'])}</td>"
            f"<td>{escape(row['top_rules'])}</td>"
            "</tr>"
        )

    tag_rows = ""
    total_tag_count = sum(tags.values()) or 1
    for tag, count in tags.most_common():
        tag_rows += (
            "<tr>"
            f"<td><code>{escape(tag)}</code></td>"
            f"<td>{format_num(count)}</td>"
            f"<td>{pct(count, total_tag_count):.2f}%</td>"
            "</tr>"
        )

    turn_rows_html = grouped_turn_rows_html(turns)

    rule_rows = ""
    total_rule_count = sum(rules.values()) or 1
    for rule, count in rules.most_common():
        rule_rows += (
            "<tr>"
            f"<td><code>{escape(rule)}</code></td>"
            f"<td>{format_num(count)}</td>"
            f"<td>{pct(count, total_rule_count):.2f}%</td>"
            "</tr>"
        )

    scenario_library = scenario_details_html(dataset)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conversation Eval Dataset Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --bg: #ffffff;
  --panel: #ffffff;
  --panel-2: #fafafa;
  --text: #000000;
  --muted: #303030;
  --border: #000000;
  --pass: #000000;
  --pass-text: #ffffff;
  --fail: #ffffff;
  --fail-text: #000000;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
  overflow-x: hidden;
}}
main {{ width: 100%; max-width: none; margin: 0; padding: 24px clamp(12px, 2vw, 28px); }}
h1, h2, h3 {{ margin: 0 0 12px 0; }}
p, li {{ color: var(--text); }}
small, .muted {{ color: var(--muted); }}
header {{
  display: grid;
  gap: 16px;
  margin-bottom: 28px;
}}
.subtitle {{ color: var(--muted); max-width: 1000px; }}
.section {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px;
  margin-bottom: 26px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 16px;
}}
.metric-card {{
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
  min-height: 100px;
}}
.metric-label {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 8px; }}
.metric-value {{ font-size: 1.4rem; font-weight: 700; overflow-wrap: anywhere; }}
.metric-sub {{ color: var(--muted); font-size: 0.9rem; margin-top: 6px; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
}}
th, td {{
  text-align: left;
  padding: 14px 12px;
  border-top: 1px solid var(--border);
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 600;
  background: #fcfcfc;
}}
.name {{ font-weight: 700; margin-bottom: 4px; }}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  background: #fcfcfc;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  margin: 8px 0 0 0;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  background: #fcfcfc;
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
  margin: 8px 0 0 0;
  padding-left: 18px;
}}
.trace-card {{
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
  margin-top: 16px;
  background: var(--panel-2);
}}
.trace-head {{
  color: var(--muted);
  margin-bottom: 10px;
  font-size: 0.92rem;
}}
.trace-label {{
  font-weight: 700;
  margin: 4px 0 6px 0;
}}
.trace-checks {{
  margin-top: 16px;
}}
.trace-checks summary {{
  font-weight: 600;
  color: var(--muted);
}}
.trace-checks table {{
  margin-top: 12px;
}}
.family-detail {{
  display: block;
  margin-top: 20px;
  border: 1px solid var(--border);
  padding: 18px;
  border-radius: 14px;
  background: var(--panel);
}}
.family-detail > summary {{
  font-size: 1.05rem;
  font-weight: 700;
}}
.family-meta {{
  margin-top: 12px;
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
  margin-top: 20px;
  border: 1px solid var(--border);
  padding: 18px;
  border-radius: 14px;
  background: var(--panel-2);
}}
.pill-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}}
.pill {{
  display: inline-block;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.82rem;
  background: #fcfcfc;
}}
.grid-2 {{
  display: grid;
  grid-template-columns: 1.3fr 0.7fr;
  gap: 24px;
}}
.benchmark-detail-grid {{
  grid-template-columns: 1fr;
}}
.group-row td {{
  background: var(--panel-2);
  font-weight: 700;
  border-top: 1px solid var(--border);
}}
.turn-table-section {{
  max-width: 100%;
  overflow-x: auto;
}}
.footer {{ color: var(--muted); font-size: 0.9rem; margin-top: 18px; }}
@media (max-width: 980px) {{
  .grid-2 {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>Conversation evaluation dataset viewer</h1>
  <div class="subtitle">
    This view shows the benchmark contract surface area, not scored model output. It is built from
    <code>{escape(source_path.name)}</code> and is meant to answer two questions quickly: “what workload classes does this benchmark cover?”
    and “what exact contract and deterministic check logic will the harness run?”
  </div>
</header>

<section class="section">
  <h2>Dataset summary</h2>
  <div class="metrics">{summary_cards}</div>
  <div class="footer">
    Source <code>{escape(str(source_path))}</code>
  </div>
</section>

<div class="grid-2">
  <section class="section">
    <h2>Family coverage</h2>
    <table>
      <thead>
        <tr>
          <th>Family</th>
          <th>Scenarios</th>
          <th>Turns</th>
          <th>Checks</th>
          <th>Top rules</th>
        </tr>
      </thead>
      <tbody>{family_table_rows}</tbody>
    </table>
  </section>

  <section class="section">
    <h2>Tag coverage</h2>
    <table>
      <thead>
        <tr><th>Tag</th><th>Count</th><th>Share</th></tr>
      </thead>
      <tbody>{tag_rows}</tbody>
    </table>
  </section>
</div>

<div class="grid-2 benchmark-detail-grid">
  <section class="section turn-table-section">
    <h2>Turn-level dataset view</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario / turn</th>
          <th>Checks</th>
          <th>Rules</th>
          <th>Contracts</th>
          <th>User message</th>
          <th>Check details</th>
        </tr>
      </thead>
      <tbody>{turn_rows_html}</tbody>
    </table>
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
    </details>
  </section>
</div>

<section class="section">
  <h2>Scenario library</h2>
  <p class="muted">
    Use this section to inspect the exact system prompt, turns, input contracts, output contracts, and deterministic checks that define the benchmark.
  </p>
  {scenario_library}
</section>
</main>
<script>
document.querySelectorAll(".trace-card > table").forEach((table) => {{
  const details = document.createElement("details");
  details.className = "trace-checks";

  const summary = document.createElement("summary");
  summary.textContent = "Checks";

  table.before(details);
  details.appendChild(summary);
  details.appendChild(table);
}});

document.querySelectorAll("details").forEach((element) => {{
  element.open = false;
}});
document.querySelectorAll(".turn-table-section details").forEach((element) => {{
  element.open = true;
}});
const firstScenario = document.querySelector(".scenario-detail");
if (firstScenario) {{
  firstScenario.open = true;
}}
const firstFamily = document.querySelector(".family-detail");
if (firstFamily) {{
  firstFamily.open = true;
}}
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a conversation eval scenario dataset JSON file into an HTML viewer."
    )
    parser.add_argument("input", help="Scenario dataset JSON file or directory.")
    parser.add_argument(
        "--output",
        default="conversation_eval_dataset_dashboard.html",
        help="Output HTML file path.",
    )
    args = parser.parse_args()

    dataset, input_path, source_kind = load_view_source(args.input)
    html_text = build_html(dataset, input_path, source_kind)

    output_path = Path(args.output)
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
