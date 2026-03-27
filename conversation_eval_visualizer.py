
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
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


def scenario_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in report.get("results", []):
        traces = result.get("traces", [])
        lats = [float(t.get("latency_ms", 0.0)) for t in traces]
        comp = [int(t.get("completion_tokens", 0)) for t in traces]
        prompt = [int(t.get("prompt_tokens") or 0) for t in traces]
        checks = [c for t in traces for c in t.get("checks", [])]
        rows.append(
            {
                "name": result["name"],
                "description": result.get("description", ""),
                "passed": result.get("passed", False),
                "score_pct": result.get("score_pct", 0.0),
                "turns": len(traces),
                "checks": len(checks),
                "failures": sum(1 for c in checks if not c.get("passed")),
                "latency_total_ms": sum(lats),
                "latency_avg_ms": safe_div(sum(lats), len(lats)),
                "completion_tokens_total": sum(comp),
                "prompt_tokens_final": prompt[-1] if prompt else 0,
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


def build_single_run_html(report: dict[str, Any], source_path: Path) -> str:
    summary = report["summary"]
    config = report["config"]
    meta = report["meta"]
    rules = rule_counts(report)
    scen_rows = scenario_rows(report)
    turns = turn_rows(report)

    max_scen_lat = max((row["latency_total_ms"] for row in scen_rows), default=0.0)
    max_turn_lat = max((row["latency_ms"] for row in turns), default=0.0)
    max_turn_prompt = max(((row["prompt_tokens"] or 0) for row in turns), default=0)
    max_turn_completion = max((row["completion_tokens"] or 0 for row in turns), default=0)

    summary_cards = "".join(
        [
            metric_card("Model", escape(config.get("base_model", "—")), config.get("resolved_renderer_name", "—")),
            metric_card("Overall score", f"{summary.get('score_pct', 0):.2f}%", f"{summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} checks passed"),
            metric_card("Scenarios", f"{summary.get('passed_scenarios', 0)}/{summary.get('scenario_count', 0)}", "passed"),
            metric_card("Total latency", format_ms(summary.get("total_latency_ms")), "all turns"),
            metric_card("Completion tokens", format_num(summary.get("total_completion_tokens")), "all turns"),
            metric_card("Fingerprint", f"<code>{escape(meta.get('workload_fingerprint', '')[:12])}…</code>", "workload id"),
        ]
    )

    scenario_table_rows = ""
    for row in scen_rows:
        scenario_table_rows += (
            "<tr>"
            f"<td><div class='name'>{escape(row['name'])}</div><div class='muted'>{escape(row['description'])}</div></td>"
            f"<td>{badge('PASS', 'pass') if row['passed'] else badge('FAIL', 'fail')}</td>"
            f"<td>{row['score_pct']:.2f}%</td>"
            f"<td>{row['turns']}</td>"
            f"<td>{row['checks']}</td>"
            f"<td>{bar_cell(row['latency_total_ms'], max_scen_lat, format_ms(row['latency_total_ms']))}</td>"
            f"<td>{format_num(row['completion_tokens_total'])}</td>"
            f"<td>{format_num(row['prompt_tokens_final'])}</td>"
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

    turn_rows_html = ""
    for row in turns:
        checks_html = "".join(
            "<li>"
            f"{badge('pass', 'pass') if c.get('passed') else badge('fail', 'fail')} "
            f"<code>{escape(c.get('name', ''))}</code> — {escape(c.get('details', ''))}"
            "</li>"
            for c in row["checks"]
        )
        turn_rows_html += (
            "<tr>"
            f"<td><div class='name'>{escape(row['scenario'])}</div><div class='muted'>Turn {escape(row['index'])}</div></td>"
            f"<td>{bar_cell(row['latency_ms'], max_turn_lat, format_ms(row['latency_ms']))}</td>"
            f"<td>{bar_cell(float(row['prompt_tokens'] or 0), float(max_turn_prompt or 1), format_num(row['prompt_tokens']))}</td>"
            f"<td>{bar_cell(float(row['completion_tokens'] or 0), float(max_turn_completion or 1), format_num(row['completion_tokens']))}</td>"
            f"<td>{row['checks_passed']}/{row['checks_total']}</td>"
            f"<td>{escape(row['stop_reason'])}</td>"
            f"<td><details><summary>{escape(truncate(row['user_message'], 110))}</summary><pre>{escape(row['user_message'])}</pre></details></td>"
            f"<td><details><summary>{escape(truncate(row['assistant_message'], 110))}</summary><pre>{escape(row['assistant_message'])}</pre></details></td>"
            f"<td><details><summary>Checks</summary><ul class='check-list'>{checks_html}</ul></details></td>"
            "</tr>"
        )

    scenario_details = ""
    for result in report["results"]:
        scenario_details += (
            "<details class='scenario-detail'>"
            f"<summary><strong>{escape(result['name'])}</strong> — {escape(result.get('description', ''))}</summary>"
        )
        for trace in result.get("traces", []):
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
                "<div class='trace-card'>"
                f"<div class='trace-head'>Turn {escape(trace.get('index'))} • {format_ms(trace.get('latency_ms'))} • "
                f"prompt {format_num(trace.get('prompt_tokens'))} • completion {format_num(trace.get('completion_tokens'))}</div>"
                f"<div class='trace-block'><div class='trace-label'>User</div><pre>{escape(trace.get('user_message', ''))}</pre></div>"
                f"<div class='trace-block'><div class='trace-label'>Assistant</div><pre>{escape(trace.get('assistant_message', ''))}</pre></div>"
                "<table><thead><tr><th>Check</th><th>Rule</th><th>Status</th><th>Details</th></tr></thead>"
                f"<tbody>{checks_html}</tbody></table>"
                "</div>"
            )
        scenario_details += "</details>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conversation Eval Dashboard</title>
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
  --bar: #d2d2d2;
  --accent: #b8b8b8;
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
  background: #ffffff;
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
.footer {{ color: var(--muted); font-size: 0.9rem; margin-top: 18px; }}
@media (max-width: 980px) {{
  .grid-2 {{ grid-template-columns: 1fr; }}
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
    <h2>Scenario performance</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario</th>
          <th>Status</th>
          <th>Score</th>
          <th>Turns</th>
          <th>Checks</th>
          <th>Total latency</th>
          <th>Completion tokens</th>
          <th>Final prompt tokens</th>
        </tr>
      </thead>
      <tbody>{scenario_table_rows}</tbody>
    </table>
  </section>
</div>

<div class="grid-2 benchmark-detail-grid">
  <section class="section turn-table-section">
    <h2>Turn-level benchmark view</h2>
    <table>
      <thead>
        <tr>
          <th>Scenario / turn</th>
          <th>Latency</th>
          <th>Prompt tokens</th>
          <th>Completion tokens</th>
          <th>Checks</th>
          <th>Stop</th>
          <th>User message</th>
          <th>Assistant message</th>
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
      <p class="muted" style="margin-top:12px;">
        Rule coverage matters because it shows whether the suite is testing multiple failure shapes or mostly
        repeating one scoring style.
      </p>
    </details>
  </section>
</div>

<section class="section">
  <h2>Replay trace</h2>
  <p class="muted">
    Use this section for failure analysis and benchmark authoring. Each scenario can be opened to inspect the
    exact user prompt, assistant reply, and deterministic check outcomes turn by turn.
  </p>
  {scenario_details}
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
const firstReplayScenario = document.querySelector(".scenario-detail");
if (firstReplayScenario) {{
  firstReplayScenario.open = true;
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
