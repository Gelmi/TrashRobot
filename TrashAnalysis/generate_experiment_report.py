#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

CONDITIONS = ["justification", "no_justification"]
VALID_LABELS = {"recycle", "waste"}


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "recycle": "recycle",
        "recyclable": "recycle",
        "recycling": "recycle",
        "waste": "waste",
        "trash": "waste",
        "garbage": "waste",
        "non-recyclable": "waste",
        "non_recyclable": "waste",
        "not recyclable": "waste",
    }
    return mapping.get(text, text if text in VALID_LABELS else None)


def opposite_label(label: str | None) -> str | None:
    if label == "recycle":
        return "waste"
    if label == "waste":
        return "recycle"
    return None


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


@dataclass
class ConditionMetrics:
    condition: str
    n_trials: int
    correct_trials: int
    incorrect_trials: int
    follow_on_correct_numerator: int
    follow_on_correct_denominator: int
    follow_on_correct_rate: float | None
    reject_on_incorrect_numerator: int
    reject_on_incorrect_denominator: int
    reject_on_incorrect_rate: float | None
    first_error_index: int | None
    pre_error_correct_compliance_numerator: int
    pre_error_correct_compliance_denominator: int
    pre_error_correct_compliance_rate: float | None
    post_error_correct_compliance_numerator: int
    post_error_correct_compliance_denominator: int
    post_error_correct_compliance_rate: float | None
    trust_decay: float | None
    mean_time_arm: float | None
    mean_time_speaker: float | None
    zero_time_arm_count: int
    zero_time_speaker_count: int
    memory_correct_numerator: int
    memory_correct_denominator: int
    memory_accuracy_rate: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "n_trials": self.n_trials,
            "correct_trials": self.correct_trials,
            "incorrect_trials": self.incorrect_trials,
            "follow_on_correct": {
                "numerator": self.follow_on_correct_numerator,
                "denominator": self.follow_on_correct_denominator,
                "rate": round_or_none(self.follow_on_correct_rate),
            },
            "reject_on_incorrect": {
                "numerator": self.reject_on_incorrect_numerator,
                "denominator": self.reject_on_incorrect_denominator,
                "rate": round_or_none(self.reject_on_incorrect_rate),
            },
            "trust_decay": {
                "first_error_index": self.first_error_index,
                "pre_error_correct_compliance": {
                    "numerator": self.pre_error_correct_compliance_numerator,
                    "denominator": self.pre_error_correct_compliance_denominator,
                    "rate": round_or_none(self.pre_error_correct_compliance_rate),
                },
                "post_error_correct_compliance": {
                    "numerator": self.post_error_correct_compliance_numerator,
                    "denominator": self.post_error_correct_compliance_denominator,
                    "rate": round_or_none(self.post_error_correct_compliance_rate),
                },
                "decay": round_or_none(self.trust_decay),
            },
            "timing": {
                "mean_time_arm": round_or_none(self.mean_time_arm),
                "mean_time_speaker": round_or_none(self.mean_time_speaker),
                "zero_time_arm_count": self.zero_time_arm_count,
                "zero_time_speaker_count": self.zero_time_speaker_count,
            },
            "memory_test": {
                "numerator": self.memory_correct_numerator,
                "denominator": self.memory_correct_denominator,
                "rate": round_or_none(self.memory_accuracy_rate),
            },
        }


def robot_suggestion(trial: dict[str, Any]) -> str | None:
    ground_truth = normalize_label(trial.get("ground_truth"))
    is_correct = trial.get("is_correct")
    if ground_truth is None or is_correct is None:
        return None
    return ground_truth if bool(is_correct) else opposite_label(ground_truth)



def compute_condition_metrics(trials: list[dict[str, Any]], condition: str) -> ConditionMetrics:
    follow_correct_num = 0
    follow_correct_den = 0
    reject_incorrect_num = 0
    reject_incorrect_den = 0

    arm_times: list[float] = []
    speaker_times: list[float] = []
    zero_time_arm_count = 0
    zero_time_speaker_count = 0

    memory_correct_num = 0
    memory_correct_den = 0

    correct_trials = 0
    incorrect_trials = 0
    first_error_index: int | None = None

    cleaned: list[dict[str, Any]] = []

    for idx, trial in enumerate(trials):
        gt = normalize_label(trial.get("ground_truth"))
        result = normalize_label(trial.get("result"))
        mem = normalize_label(trial.get("memory_test"))
        suggestion = robot_suggestion(trial)
        is_correct = trial.get("is_correct")

        time_arm_raw = trial.get("time_arm", 0)
        time_speaker_raw = trial.get("time_speaker", 0)
        try:
            time_arm = float(time_arm_raw)
        except (TypeError, ValueError):
            time_arm = 0.0
        try:
            time_speaker = float(time_speaker_raw)
        except (TypeError, ValueError):
            time_speaker = 0.0

        arm_times.append(time_arm)
        speaker_times.append(time_speaker)
        if time_arm == 0:
            zero_time_arm_count += 1
        if time_speaker == 0:
            zero_time_speaker_count += 1

        if gt is not None and mem is not None:
            memory_correct_den += 1
            if mem == gt:
                memory_correct_num += 1

        if is_correct is True:
            correct_trials += 1
            if suggestion is not None and result is not None:
                follow_correct_den += 1
                if result == suggestion:
                    follow_correct_num += 1
        elif is_correct is False:
            incorrect_trials += 1
            if first_error_index is None:
                first_error_index = idx
            if suggestion is not None and result is not None:
                reject_incorrect_den += 1
                if result != suggestion:
                    reject_incorrect_num += 1

        cleaned.append(
            {
                "index": idx,
                "ground_truth": gt,
                "result": result,
                "memory_test": mem,
                "robot_suggestion": suggestion,
                "is_correct": is_correct,
            }
        )

    pre_num = pre_den = post_num = post_den = 0
    if first_error_index is not None:
        for row in cleaned:
            idx = row["index"]
            if row["is_correct"] is not True:
                continue
            if row["robot_suggestion"] is None or row["result"] is None:
                continue
            complied = row["result"] == row["robot_suggestion"]
            if idx < first_error_index:
                pre_den += 1
                if complied:
                    pre_num += 1
            elif idx > first_error_index:
                post_den += 1
                if complied:
                    post_num += 1

    pre_rate = pct(pre_num, pre_den)
    post_rate = pct(post_num, post_den)
    decay = None if pre_rate is None or post_rate is None else pre_rate - post_rate

    return ConditionMetrics(
        condition=condition,
        n_trials=len(trials),
        correct_trials=correct_trials,
        incorrect_trials=incorrect_trials,
        follow_on_correct_numerator=follow_correct_num,
        follow_on_correct_denominator=follow_correct_den,
        follow_on_correct_rate=pct(follow_correct_num, follow_correct_den),
        reject_on_incorrect_numerator=reject_incorrect_num,
        reject_on_incorrect_denominator=reject_incorrect_den,
        reject_on_incorrect_rate=pct(reject_incorrect_num, reject_incorrect_den),
        first_error_index=first_error_index,
        pre_error_correct_compliance_numerator=pre_num,
        pre_error_correct_compliance_denominator=pre_den,
        pre_error_correct_compliance_rate=pre_rate,
        post_error_correct_compliance_numerator=post_num,
        post_error_correct_compliance_denominator=post_den,
        post_error_correct_compliance_rate=post_rate,
        trust_decay=decay,
        mean_time_arm=safe_mean(arm_times),
        mean_time_speaker=safe_mean(speaker_times),
        zero_time_arm_count=zero_time_arm_count,
        zero_time_speaker_count=zero_time_speaker_count,
        memory_correct_numerator=memory_correct_num,
        memory_correct_denominator=memory_correct_den,
        memory_accuracy_rate=pct(memory_correct_num, memory_correct_den),
    )



def aggregate_condition_metrics(condition_metrics_list: list[ConditionMetrics], condition: str) -> dict[str, Any]:
    def summed(attr: str) -> int:
        return sum(getattr(m, attr) for m in condition_metrics_list)

    def avg_of(attr: str) -> float | None:
        values = [getattr(m, attr) for m in condition_metrics_list if getattr(m, attr) is not None]
        return safe_mean(values)

    follow_num = summed("follow_on_correct_numerator")
    follow_den = summed("follow_on_correct_denominator")
    reject_num = summed("reject_on_incorrect_numerator")
    reject_den = summed("reject_on_incorrect_denominator")
    mem_num = summed("memory_correct_numerator")
    mem_den = summed("memory_correct_denominator")
    pre_num = summed("pre_error_correct_compliance_numerator")
    pre_den = summed("pre_error_correct_compliance_denominator")
    post_num = summed("post_error_correct_compliance_numerator")
    post_den = summed("post_error_correct_compliance_denominator")

    pooled_pre = pct(pre_num, pre_den)
    pooled_post = pct(post_num, post_den)
    pooled_decay = None if pooled_pre is None or pooled_post is None else pooled_pre - pooled_post

    return {
        "condition": condition,
        "participants": len(condition_metrics_list),
        "pooled": {
            "n_trials": summed("n_trials"),
            "correct_trials": summed("correct_trials"),
            "incorrect_trials": summed("incorrect_trials"),
            "follow_on_correct": {
                "numerator": follow_num,
                "denominator": follow_den,
                "rate": round_or_none(pct(follow_num, follow_den)),
            },
            "reject_on_incorrect": {
                "numerator": reject_num,
                "denominator": reject_den,
                "rate": round_or_none(pct(reject_num, reject_den)),
            },
            "trust_decay": {
                "pre_error_correct_compliance": {
                    "numerator": pre_num,
                    "denominator": pre_den,
                    "rate": round_or_none(pooled_pre),
                },
                "post_error_correct_compliance": {
                    "numerator": post_num,
                    "denominator": post_den,
                    "rate": round_or_none(pooled_post),
                },
                "decay": round_or_none(pooled_decay),
            },
            "timing": {
                "mean_time_arm": round_or_none(avg_of("mean_time_arm")),
                "mean_time_speaker": round_or_none(avg_of("mean_time_speaker")),
                "zero_time_arm_count": summed("zero_time_arm_count"),
                "zero_time_speaker_count": summed("zero_time_speaker_count"),
            },
            "memory_test": {
                "numerator": mem_num,
                "denominator": mem_den,
                "rate": round_or_none(pct(mem_num, mem_den)),
            },
        },
        "participant_average": {
            "follow_on_correct_rate": round_or_none(avg_of("follow_on_correct_rate")),
            "reject_on_incorrect_rate": round_or_none(avg_of("reject_on_incorrect_rate")),
            "trust_decay": round_or_none(avg_of("trust_decay")),
            "mean_time_arm": round_or_none(avg_of("mean_time_arm")),
            "mean_time_speaker": round_or_none(avg_of("mean_time_speaker")),
            "memory_accuracy_rate": round_or_none(avg_of("memory_accuracy_rate")),
        },
    }



def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of participants.")
    return data



def analyze_dataset(participants: list[dict[str, Any]]) -> dict[str, Any]:
    participant_rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[ConditionMetrics]] = {c: [] for c in CONDITIONS}

    for participant in participants:
        pid = str(participant.get("participant_id", ""))
        name = participant.get("name", "")
        row = {
            "participant_id": pid,
            "name": name,
            "gender": participant.get("gender"),
            "age": participant.get("age"),
            "first_scenario": participant.get("first_scenario"),
            "conditions": {},
        }
        for condition in CONDITIONS:
            trials = participant.get(condition, []) or []
            if not isinstance(trials, list):
                trials = []
            metrics = compute_condition_metrics(trials, condition)
            by_condition[condition].append(metrics)
            row["conditions"][condition] = metrics.as_dict()
        participant_rows.append(row)

    overall = {
        condition: aggregate_condition_metrics(metrics_list, condition)
        for condition, metrics_list in by_condition.items()
    }

    return {
        "n_participants": len(participants),
        "overall": overall,
        "participants": participant_rows,
        "notes": {
            "robot_suggestion_rule": "robot suggestion = ground_truth when is_correct is true, otherwise the opposite label",
            "memory_accuracy_rule": "memory_test is counted as correct when it matches ground_truth",
            "trust_decay_rule": "trust decay = compliance on correct trials before first incorrect recommendation minus compliance on correct trials after that first incorrect recommendation; the incorrect trial itself is excluded",
        },
    }



def format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def format_num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"



def build_html(report: dict[str, Any]) -> str:
    overall = report["overall"]
    participant_rows_html = []

    for p in report["participants"]:
        j = p["conditions"]["justification"]
        n = p["conditions"]["no_justification"]
        participant_rows_html.append(
            f"""
            <tr>
              <td>{p['participant_id']}</td>
              <td>{p.get('name','') or ''}</td>
              <td>{p.get('first_scenario','') or ''}</td>
              <td>{format_pct(j['follow_on_correct']['rate'])}</td>
              <td>{format_pct(j['reject_on_incorrect']['rate'])}</td>
              <td>{format_pct(j['trust_decay']['decay'])}</td>
              <td>{format_num(j['timing']['mean_time_arm'])}</td>
              <td>{format_num(j['timing']['mean_time_speaker'])}</td>
              <td>{j['timing']['zero_time_arm_count']}</td>
              <td>{j['timing']['zero_time_speaker_count']}</td>
              <td>{format_pct(j['memory_test']['rate'])}</td>
              <td>{format_pct(n['follow_on_correct']['rate'])}</td>
              <td>{format_pct(n['reject_on_incorrect']['rate'])}</td>
              <td>{format_pct(n['trust_decay']['decay'])}</td>
              <td>{format_num(n['timing']['mean_time_arm'])}</td>
              <td>{format_num(n['timing']['mean_time_speaker'])}</td>
              <td>{n['timing']['zero_time_arm_count']}</td>
              <td>{n['timing']['zero_time_speaker_count']}</td>
              <td>{format_pct(n['memory_test']['rate'])}</td>
            </tr>
            """
        )

    def summary_card(title: str, cond: dict[str, Any]) -> str:
        pooled = cond["pooled"]
        return f"""
        <section class=\"card\">
          <h2>{title}</h2>
          <div class=\"grid\">
            <div><span class=\"label\">Participants</span><span class=\"value\">{cond['participants']}</span></div>
            <div><span class=\"label\">Trials</span><span class=\"value\">{pooled['n_trials']}</span></div>
            <div><span class=\"label\">Follow on correct</span><span class=\"value\">{format_pct(pooled['follow_on_correct']['rate'])}</span></div>
            <div><span class=\"label\">Reject on incorrect</span><span class=\"value\">{format_pct(pooled['reject_on_incorrect']['rate'])}</span></div>
            <div><span class=\"label\">Pre-error compliance</span><span class=\"value\">{format_pct(pooled['trust_decay']['pre_error_correct_compliance']['rate'])}</span></div>
            <div><span class=\"label\">Post-error compliance</span><span class=\"value\">{format_pct(pooled['trust_decay']['post_error_correct_compliance']['rate'])}</span></div>
            <div><span class=\"label\">Trust decay</span><span class=\"value\">{format_pct(pooled['trust_decay']['decay'])}</span></div>
            <div><span class=\"label\">Mean time arm</span><span class=\"value\">{format_num(pooled['timing']['mean_time_arm'])}</span></div>
            <div><span class=\"label\">Mean time speaker</span><span class=\"value\">{format_num(pooled['timing']['mean_time_speaker'])}</span></div>
            <div><span class=\"label\">Zero arm times</span><span class=\"value\">{pooled['timing']['zero_time_arm_count']}</span></div>
            <div><span class=\"label\">Zero speaker times</span><span class=\"value\">{pooled['timing']['zero_time_speaker_count']}</span></div>
            <div><span class=\"label\">Memory accuracy</span><span class=\"value\">{format_pct(pooled['memory_test']['rate'])}</span></div>
          </div>
        </section>
        """

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Experiment Report</title>
  <style>
    :root {{
      --bg: #0b1020;
      --card: #141b34;
      --muted: #aab4d6;
      --text: #eef2ff;
      --accent: #78a6ff;
      --line: #2a345d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p, li {{ color: var(--muted); line-height: 1.5; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; margin: 24px 0; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px; box-shadow: 0 10px 25px rgba(0,0,0,.20); }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(150px,1fr)); gap: 12px; }}
    .label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 20px; font-weight: bold; }}
    .search {{ width: 100%; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--line); background: #0f1530; color: var(--text); margin: 8px 0 16px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 16px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1400px; background: var(--card); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 14px; }}
    th {{ position: sticky; top: 0; background: #1a2346; z-index: 1; }}
    .subhead {{ color: var(--accent); font-size: 13px; }}
    .notes {{ margin-top: 24px; }}
    code {{ background: #0f1530; padding: 2px 6px; border-radius: 6px; color: #d8e3ff; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Trash Sorting Experiment Report</h1>
    <p>Participants sorted items with robot assistance under two conditions: <strong>justification</strong> and <strong>no_justification</strong>. This report computes per-participant and pooled metrics directly from the JSON dataset.</p>
    <p><strong>Total participants:</strong> {report['n_participants']}</p>

    <div class=\"cards\">
      {summary_card('Justification', overall['justification'])}
      {summary_card('No Justification', overall['no_justification'])}
    </div>

    <section class=\"card\">
      <h2>Participants</h2>
      <input id=\"search\" class=\"search\" type=\"text\" placeholder=\"Filter by participant id, name, or first scenario\">
      <div class=\"table-wrap\">
        <table id=\"participantsTable\">
          <thead>
            <tr>
              <th rowspan=\"2\">ID</th>
              <th rowspan=\"2\">Name</th>
              <th rowspan=\"2\">First scenario</th>
              <th colspan=\"8\" class=\"subhead\">Justification</th>
              <th colspan=\"8\" class=\"subhead\">No Justification</th>
            </tr>
            <tr>
              <th>Follow on correct</th>
              <th>Reject on incorrect</th>
              <th>Trust decay</th>
              <th>Mean arm</th>
              <th>Mean speaker</th>
              <th>Zero arm</th>
              <th>Zero speaker</th>
              <th>Memory acc.</th>
              <th>Follow on correct</th>
              <th>Reject on incorrect</th>
              <th>Trust decay</th>
              <th>Mean arm</th>
              <th>Mean speaker</th>
              <th>Zero arm</th>
              <th>Zero speaker</th>
              <th>Memory acc.</th>
            </tr>
          </thead>
          <tbody>
            {''.join(participant_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section class=\"card notes\">
      <h2>Metric definitions used</h2>
      <ul>
        <li><strong>Robot suggestion:</strong> <code>ground_truth</code> when <code>is_correct = true</code>, otherwise the opposite label.</li>
        <li><strong>Follow on correct:</strong> participant <code>result</code> equals robot suggestion on trials where <code>is_correct = true</code>.</li>
        <li><strong>Reject on incorrect:</strong> participant <code>result</code> differs from robot suggestion on trials where <code>is_correct = false</code>.</li>
        <li><strong>Trust decay:</strong> compliance on correct trials before the first incorrect recommendation minus compliance on correct trials after the first incorrect recommendation. The incorrect trial itself is excluded.</li>
        <li><strong>Memory accuracy:</strong> <code>memory_test</code> matches <code>ground_truth</code>.</li>
      </ul>
    </section>
  </div>

  <script>
    const search = document.getElementById('search');
    const rows = Array.from(document.querySelectorAll('#participantsTable tbody tr'));
    search.addEventListener('input', () => {{
      const q = search.value.toLowerCase().trim();
      rows.forEach(row => {{
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(q) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment metrics and an HTML report from participant JSON.")
    parser.add_argument("--input", required=True, help="Path to the participants JSON file")
    parser.add_argument("--output-html", default="experiment_report.html", help="Path to the output HTML report")
    parser.add_argument("--output-json", default="experiment_metrics.json", help="Path to the output metrics JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    participants = load_dataset(input_path)
    report = analyze_dataset(participants)

    output_json = Path(args.output_json)
    output_html = Path(args.output_html)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_html.write_text(build_html(report), encoding="utf-8")

    print(f"Loaded {report['n_participants']} participants from {input_path}")
    print(f"Metrics JSON written to {output_json}")
    print(f"HTML report written to {output_html}")


if __name__ == "__main__":
    main()
