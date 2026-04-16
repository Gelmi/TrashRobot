
import json
import math
import os
from pathlib import Path
from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def load_data(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(data: dict) -> pd.DataFrame:
    rows = []
    for p in data["participants"]:
        base = {"id": str(p["id"]), "gender": p["gender"], "age": p["age"]}
        for cond_key, cond_label in [
            ("justification", "Justification"),
            ("no_justification", "No Justification"),
        ]:
            row = dict(base)
            row["condition"] = cond_label
            row.update(p[cond_key])
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot_condition_metrics(df: pd.DataFrame, outpath: Path) -> None:
    summary = df.groupby("condition").agg(
        compliance_correct=("compliance_rate_when_correct", "mean"),
        override_incorrect=("override_rate_when_incorrect", "mean"),
        memory_accuracy=("memory_test_accuracy", "mean"),
    )
    metrics = ["compliance_correct", "override_incorrect", "memory_accuracy"]
    labels = [
        "Compliance when robot is correct",
        "Override when robot is incorrect",
        "Memory accuracy",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(metrics))
    width = 0.35
    conditions = list(summary.index)

    for i, cond in enumerate(conditions):
        values = [summary.loc[cond, m] for m in metrics]
        ax.bar([v + (i - 0.5) * width for v in x], values, width=width, label=cond)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Behavioral metrics by condition (participant-level means)")
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_plot_times(df: pd.DataFrame, outpath: Path) -> None:
    summary = df.groupby("condition").agg(
        avg_arm_time=("avg_time_arm", "mean"),
        avg_speaker_time=("avg_time_speaker", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary.index))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary["avg_arm_time"], width=width, label="Arm time")
    ax.bar([i + width / 2 for i in x], summary["avg_speaker_time"], width=width, label="Speaker time")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.index)
    ax.set_ylabel("Seconds per trial")
    ax.set_title("Average time per trial by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_plot_paired_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, outpath: Path, y_percent: bool = True) -> None:
    pivot = df.pivot(index="id", columns="condition", values=metric).sort_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for participant_id, row in pivot.iterrows():
        ax.plot([0, 1], [row["No Justification"], row["Justification"]], marker="o", alpha=0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Justification", "Justification"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_plot_trust_decay(df: pd.DataFrame, overall: dict, outpath: Path) -> None:
    participant_summary = df.groupby("condition").agg(
        pre=("compliance_rate_before_failure", "mean"),
        post=("compliance_rate_after_failure", "mean"),
    )

    pooled = pd.DataFrame(
        {
            "pre": [
                overall["total_compliance_rate_before_failure_no_justification"],
                overall["total_compliance_rate_before_failure_justification"],
            ],
            "post": [
                overall["total_compliance_rate_after_failure_no_justification"],
                overall["total_compliance_rate_after_failure_justification"],
            ],
        },
        index=["No Justification", "Justification"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, table, subtitle in [
        (axes[0], participant_summary.loc[["No Justification", "Justification"]], "Participant-level mean"),
        (axes[1], pooled, "Pooled over all valid trials"),
    ]:
        x = range(len(table.index))
        width = 0.35
        ax.bar([i - width / 2 for i in x], table["pre"], width=width, label="Pre-error")
        ax.bar([i + width / 2 for i in x], table["post"], width=width, label="Post-error")
        ax.set_xticks(list(x))
        ax.set_xticklabels(table.index, rotation=10)
        ax.set_title(subtitle)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Compliance on correct trials")
    axes[1].legend()
    fig.suptitle("Trust before vs after first robot error")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_plot_demographics(data: dict, outpath: Path) -> None:
    participants = pd.DataFrame(
        [{"id": str(p["id"]), "gender": p["gender"], "age": p["age"]} for p in data["participants"]]
    )
    gender_counts = participants["gender"].value_counts().sort_index()
    ages = participants["age"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(gender_counts.index, gender_counts.values)
    axes[0].set_title("Gender count")
    axes[0].set_ylabel("Participants")

    axes[1].hist(ages, bins=min(8, max(4, ages.nunique())))
    axes[1].set_title("Age distribution")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Participants")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def round_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def render_html(data: dict, df: pd.DataFrame, outdir: Path) -> str:
    participants = pd.DataFrame(
        [{"id": str(p["id"]), "gender": p["gender"], "age": p["age"]} for p in data["participants"]]
    )
    overall = data["overall_metrics"]

    condition_summary = df.groupby("condition").agg(
        participants=("id", "nunique"),
        mean_age=("age", "mean"),
        compliance_when_correct=("compliance_rate_when_correct", "mean"),
        override_when_incorrect=("override_rate_when_incorrect", "mean"),
        memory_accuracy=("memory_test_accuracy", "mean"),
        avg_arm_time_s=("avg_time_arm", "mean"),
        avg_speaker_time_s=("avg_time_speaker", "mean"),
        pre_error_compliance_mean=("compliance_rate_before_failure", "mean"),
        post_error_compliance_mean=("compliance_rate_after_failure", "mean"),
    ).reset_index()
    condition_summary["trust_decay_mean"] = (
        condition_summary["pre_error_compliance_mean"] - condition_summary["post_error_compliance_mean"]
    )

    pooled_summary = pd.DataFrame(
        [
            {
                "condition": "Justification",
                "compliance_when_correct": overall["total_compliant_rate_when_correct_justification"],
                "override_when_incorrect": overall["total_override_rate_when_incorrect_justification"],
                "memory_accuracy": overall["total_memory_test_accuracy_justification"],
                "pre_error_compliance_pooled": overall["total_compliance_rate_before_failure_justification"],
                "post_error_compliance_pooled": overall["total_compliance_rate_after_failure_justification"],
                "trust_decay_pooled": overall["total_compliance_rate_before_failure_justification"]
                - overall["total_compliance_rate_after_failure_justification"],
                "avg_arm_time_s": overall["avg_time_arm_justification"],
                "avg_speaker_time_s": overall["avg_time_speaker_justification"],
            },
            {
                "condition": "No Justification",
                "compliance_when_correct": overall["total_compliant_rate_when_correct_no_justification"],
                "override_when_incorrect": overall["total_override_rate_when_incorrect_no_justification"],
                "memory_accuracy": overall["total_memory_test_accuracy_no_justification"],
                "pre_error_compliance_pooled": overall["total_compliance_rate_before_failure_no_justification"],
                "post_error_compliance_pooled": overall["total_compliance_rate_after_failure_no_justification"],
                "trust_decay_pooled": overall["total_compliance_rate_before_failure_no_justification"]
                - overall["total_compliance_rate_after_failure_no_justification"],
                "avg_arm_time_s": overall["avg_time_arm_no_justification"],
                "avg_speaker_time_s": overall["avg_time_speaker_no_justification"],
            },
        ]
    )

    participant_table = df[
        [
            "id",
            "gender",
            "age",
            "condition",
            "compliance_rate_when_correct",
            "override_rate_when_incorrect",
            "memory_test_accuracy",
            "avg_time_arm",
            "avg_time_speaker",
            "compliance_rate_before_failure",
            "compliance_rate_after_failure",
        ]
    ].sort_values(["id", "condition"])

    def html_table(frame: pd.DataFrame) -> str:
        return round_df(frame).to_html(index=False, classes="data-table", border=0, na_rep="—")

    n = overall["n_participants"]
    key_points = f"""
    <ul>
      <li><strong>Sample:</strong> {n} participants, within-subject comparison of justification vs no justification.</li>
      <li><strong>Compliance when the robot is correct:</strong> pooled rate was
          {overall["total_compliant_rate_when_correct_justification"]:.1%} with justification vs
          {overall["total_compliant_rate_when_correct_no_justification"]:.1%} without justification.</li>
      <li><strong>Override when the robot is incorrect:</strong> pooled rate was
          {overall["total_override_rate_when_incorrect_justification"]:.1%} with justification vs
          {overall["total_override_rate_when_incorrect_no_justification"]:.1%} without justification.</li>
      <li><strong>Memory accuracy:</strong> nearly identical across conditions
          ({overall["total_memory_test_accuracy_justification"]:.1%} vs
          {overall["total_memory_test_accuracy_no_justification"]:.1%}).</li>
      <li><strong>Timing:</strong> average arm time was much lower with justification
          ({overall["avg_time_arm_justification"]:.2f}s vs {overall["avg_time_arm_no_justification"]:.2f}s),
          while average speaker time was higher
          ({overall["avg_time_speaker_justification"]:.2f}s vs {overall["avg_time_speaker_no_justification"]:.2f}s).</li>
      <li><strong>Trust decay:</strong> depending on aggregation, the pre/post pattern is mixed, so it is worth checking both participant-level means and pooled trial totals.</li>
    </ul>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RO-MAN 2026 Recycling Experiment Dashboard</title>
<style>
  body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 0;
    background: #f6f8fb;
    color: #1f2937;
  }}
  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
  }}
  .hero {{
    background: linear-gradient(135deg, #17324d, #325d88);
    color: white;
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 24px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .card {{
    background: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
  }}
  .card h3 {{
    margin-top: 0;
    font-size: 1rem;
  }}
  .metric {{
    font-size: 2rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
  }}
  .section {{
    margin: 24px 0;
  }}
  .section h2 {{
    margin-bottom: 12px;
  }}
  .plot {{
    background: white;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 16px;
  }}
  .plot img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    border-radius: 14px;
    overflow: hidden;
  }}
  .data-table th, .data-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
    font-size: 0.95rem;
  }}
  .data-table th {{
    background: #eaf0f6;
    position: sticky;
    top: 0;
  }}
  .small {{
    color: #4b5563;
    font-size: 0.95rem;
  }}
  ul {{ line-height: 1.6; }}
</style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>RO-MAN 2026 — Human-Robot Symbiosis with AI</h1>
      <p><strong>Experiment dashboard:</strong> justification vs no-justification in an embodied-LLM recycling task.</p>
      <p class="small">This page was generated automatically from the experiment JSON file.</p>
    </div>

    <div class="grid">
      <div class="card"><h3>Participants</h3><div class="metric">{n}</div><div class="small">Within-subject design</div></div>
      <div class="card"><h3>Trials per condition</h3><div class="metric">{overall["total_n_trials_justification"]}</div><div class="small">160 justification / 160 no-justification</div></div>
      <div class="card"><h3>Pooled compliance gain</h3><div class="metric">{(overall["total_compliant_rate_when_correct_justification"] - overall["total_compliant_rate_when_correct_no_justification"]):+.1%}</div><div class="small">Justification minus no-justification</div></div>
      <div class="card"><h3>Pooled override gap</h3><div class="metric">{(overall["total_override_rate_when_incorrect_justification"] - overall["total_override_rate_when_incorrect_no_justification"]):+.1%}</div><div class="small">Justification minus no-justification</div></div>
    </div>

    <div class="section card">
      <h2>Key takeaways</h2>
      {key_points}
    </div>

    <div class="section">
      <h2>Figures</h2>
      <div class="plot"><img src="fig_condition_metrics.png" alt="Condition metrics"></div>
      <div class="plot"><img src="fig_times.png" alt="Timing metrics"></div>
      <div class="plot"><img src="fig_paired_compliance.png" alt="Paired compliance plot"></div>
      <div class="plot"><img src="fig_paired_override.png" alt="Paired override plot"></div>
      <div class="plot"><img src="fig_trust_decay.png" alt="Trust decay plot"></div>
      <div class="plot"><img src="fig_demographics.png" alt="Demographics plot"></div>
    </div>

    <div class="section">
      <h2>Condition summary (participant-level means)</h2>
      {html_table(condition_summary)}
    </div>

    <div class="section">
      <h2>Condition summary (pooled from overall totals)</h2>
      {html_table(pooled_summary)}
    </div>

    <div class="section">
      <h2>Participant demographics</h2>
      {html_table(participants.sort_values("id"))}
    </div>

    <div class="section">
      <h2>Participant x condition metrics</h2>
      {html_table(participant_table)}
    </div>
  </div>
</body>
</html>"""
    return html


def generate_dashboard(json_path: str = "experiment_metrics.json", output_dir: str = "experiment_dashboard") -> None:
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    data = load_data(json_path)
    df = build_dataframe(data)

    save_plot_condition_metrics(df, output_dir / "fig_condition_metrics.png")
    save_plot_times(df, output_dir / "fig_times.png")
    save_plot_paired_metric(
        df,
        metric="compliance_rate_when_correct",
        title="Participant-level paired comparison: compliance on correct trials",
        ylabel="Compliance rate",
        outpath=output_dir / "fig_paired_compliance.png",
        y_percent=True,
    )
    save_plot_paired_metric(
        df,
        metric="override_rate_when_incorrect",
        title="Participant-level paired comparison: override on incorrect trials",
        ylabel="Override rate",
        outpath=output_dir / "fig_paired_override.png",
        y_percent=True,
    )
    save_plot_trust_decay(df, data["overall_metrics"], output_dir / "fig_trust_decay.png")
    save_plot_demographics(data, output_dir / "fig_demographics.png")

    html = render_html(data, df, output_dir)
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard created at: {output_dir / 'index.html'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate an HTML dashboard from the experiment JSON file.")
    parser.add_argument("--input", default="experiment_metrics.json", help="Path to the input JSON file")
    parser.add_argument("--output-dir", default="experiment_dashboard", help="Directory where the webpage will be written")
    args = parser.parse_args()

    generate_dashboard(args.input, args.output_dir)
