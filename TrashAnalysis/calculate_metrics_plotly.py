import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon


METRIC_SPECS = [
    {
        "key": "compliance_when_correct",
        "title": "Compliance When Robot Is Correct",
        "ylabel": "Compliance Rate",
        "j_path": ["justification", "compliance_rate_when_correct"],
        "n_path": ["no_justification", "compliance_rate_when_correct"],
        "aggregate": "identity",
        "hypothesis": "H1",
        "description": "Participant-level compliance rate on correct robot recommendations.",
    },
    {
        "key": "easy_compliance_when_correct",
        "title": "Easy Compliance When Robot Is Correct",
        "ylabel": "Compliance Rate",
        "j_path": ["justification", "easy_compliance_rate_when_correct"],
        "n_path": ["no_justification", "easy_compliance_rate_when_correct"],
        "aggregate": "identity",
        "hypothesis": "H1",
        "description": "Participant-level compliance rate on easy correct trials.",
    },
    {
        "key": "hard_compliance_when_correct",
        "title": "Hard Compliance When Robot Is Correct",
        "ylabel": "Compliance Rate",
        "j_path": ["justification", "hard_compliance_rate_when_correct"],
        "n_path": ["no_justification", "hard_compliance_rate_when_correct"],
        "aggregate": "identity",
        "hypothesis": "H1",
        "description": "Participant-level compliance rate on hard correct trials.",
    },
    {
        "key": "override_when_incorrect",
        "title": "Override When Robot Is Incorrect",
        "ylabel": "Override Rate",
        "j_path": ["justification", "override_rate_when_incorrect"],
        "n_path": ["no_justification", "override_rate_when_incorrect"],
        "aggregate": "identity",
        "hypothesis": "H2",
        "description": "Participant-level override rate on incorrect robot recommendations.",
    },
    {
        "key": "easy_override_when_incorrect",
        "title": "Easy Override When Robot Is Incorrect",
        "ylabel": "Override Rate",
        "j_path": ["justification", "easy_override_rate_when_incorrect"],
        "n_path": ["no_justification", "easy_override_rate_when_incorrect"],
        "aggregate": "identity",
        "hypothesis": "H2",
        "description": "Participant-level override rate on easy incorrect trials.",
    },
    {
        "key": "hard_override_when_incorrect",
        "title": "Hard Override When Robot Is Incorrect",
        "ylabel": "Override Rate",
        "j_path": ["justification", "hard_override_rate_when_incorrect"],
        "n_path": ["no_justification", "hard_override_rate_when_incorrect"],
        "aggregate": "identity",
        "hypothesis": "H2",
        "description": "Participant-level override rate on hard incorrect trials.",
    },
    {
        "key": "trust_decay",
        "title": "Trust Decay",
        "ylabel": "Decay Rate",
        "j_path": ["justification", "compliance_decay_rate"],
        "n_path": ["no_justification", "compliance_decay_rate"],
        "aggregate": "identity",
        "hypothesis": "H3",
        "description": "Participant-level compliance decay before vs after first robot error.",
    },
    {
        "key": "memory_test",
        "title": "Memory Test Accuracy",
        "ylabel": "Accuracy",
        "j_path": ["justification", "memory_test_accuracy"],
        "n_path": ["no_justification", "memory_test_accuracy"],
        "aggregate": "identity",
        "hypothesis": "Memory",
        "description": "Participant-level overall memory test accuracy.",
    },
    {
        "key": "easy_memory_test",
        "title": "Easy Memory Test Accuracy",
        "ylabel": "Accuracy",
        "j_path": ["justification", "easy_memory_test_accuracy"],
        "n_path": ["no_justification", "easy_memory_test_accuracy"],
        "aggregate": "identity",
        "hypothesis": "Memory",
        "description": "Participant-level memory accuracy for easy items.",
    },
    {
        "key": "hard_memory_test",
        "title": "Hard Memory Test Accuracy",
        "ylabel": "Accuracy",
        "j_path": ["justification", "hard_memory_test_accuracy"],
        "n_path": ["no_justification", "hard_memory_test_accuracy"],
        "aggregate": "identity",
        "hypothesis": "Memory",
        "description": "Participant-level memory accuracy for hard items.",
    },
    {
        "key": "time_speaker",
        "title": "Reaction Time After Speech",
        "ylabel": "Median Time (s)",
        "j_path": ["justification", "times_speaker"],
        "n_path": ["no_justification", "times_speaker"],
        "aggregate": "median",
        "hypothesis": "Timing",
        "description": "Median participant reaction time measured from the end of the spoken suggestion.",
    },
    {
        "key": "easy_time_speaker",
        "title": "Reaction Time After Speech (Easy)",
        "ylabel": "Median Time (s)",
        "j_path": ["justification", "easy_times_speaker"],
        "n_path": ["no_justification", "easy_times_speaker"],
        "aggregate": "median",
        "hypothesis": "Timing",
        "description": "Median participant reaction time after speech on easy trials.",
    },
    {
        "key": "hard_time_speaker",
        "title": "Reaction Time After Speech (Hard)",
        "ylabel": "Median Time (s)",
        "j_path": ["justification", "hard_times_speaker"],
        "n_path": ["no_justification", "hard_times_speaker"],
        "aggregate": "median",
        "hypothesis": "Timing",
        "description": "Median participant reaction time after speech on hard trials.",
    },
    {
        "key": "time_arm",
        "title": "Reaction Time After Pointing",
        "ylabel": "Median Time (s)",
        "j_path": ["justification", "times_arm"],
        "n_path": ["no_justification", "times_arm"],
        "aggregate": "median",
        "hypothesis": "Timing",
        "description": "Median participant reaction time measured from the end of the pointing movement.",
    },
    {
        "key": "easy_time_arm",
        "title": "Reaction Time After Pointing (Easy)",
        "ylabel": "Median Time (s)",
        "j_path": ["justification", "easy_times_arm"],
        "n_path": ["no_justification", "easy_times_arm"],
        "aggregate": "median",
        "hypothesis": "Timing",
        "description": "Median participant reaction time after pointing on easy trials.",
    },
    {
        "key": "hard_time_arm",
        "title": "Reaction Time After Pointing (Hard)",
        "ylabel": "Median Time (s)",
        "j_path": ["justification", "hard_times_arm"],
        "n_path": ["no_justification", "hard_times_arm"],
        "aggregate": "median",
        "hypothesis": "Timing",
        "description": "Median participant reaction time after pointing on hard trials.",
    },
]


HTML_STYLE = """
<style>
body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
h1, h2, h3 { margin-bottom: 0.3rem; }
p, li { line-height: 1.45; }
.section { margin-top: 30px; }
.metric-card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.stats { background: #f7f7f9; padding: 10px 14px; border-radius: 8px; margin: 10px 0 16px 0; }
.summary-table { border-collapse: collapse; width: 100%; margin-top: 10px; }
.summary-table th, .summary-table td { border: 1px solid #ddd; padding: 8px 10px; text-align: left; }
.summary-table th { background: #f1f3f5; }
.small { color: #555; font-size: 0.92rem; }
</style>
"""


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of participants.")
    return data


def analyze_dataset(participants: list[dict[str, Any]]) -> dict[str, Any]:
    participants_metrics = [analyze_participant(participant) for participant in participants]
    return {"participants": participants_metrics}


def analyze_participant(participant: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": participant["participant_id"],
        "gender": participant["gender"],
        "age": participant["age"],
        "justification": compute_metrics(participant.get("justification", [])),
        "no_justification": compute_metrics(participant.get("no_justification", [])),
    }


def compute_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {"n_trials": 0, "avg_response_time": None, "correct_responses": None}

    n_trials = len(trials)
    total_time_arm = sum(trial.get("time_arm", 0) for trial in trials)
    times_arm = [trial.get("time_arm", 0) for trial in trials if trial.get("time_arm") is not None]
    easy_times_arm = [trial.get("time_arm", 0) for trial in trials if trial.get("time_arm") is not None and trial.get("difficulty") == "easy"]
    hard_times_arm = [trial.get("time_arm", 0) for trial in trials if trial.get("time_arm") is not None and trial.get("difficulty") == "hard"]
    total_time_speaker = sum(trial.get("time_speaker", 0) for trial in trials)
    times_speaker = [trial.get("time_speaker", 0) for trial in trials if trial.get("time_speaker") is not None]
    easy_times_speaker = [trial.get("time_speaker", 0) for trial in trials if trial.get("time_speaker") is not None and trial.get("difficulty") == "easy"]
    hard_times_speaker = [trial.get("time_speaker", 0) for trial in trials if trial.get("time_speaker") is not None and trial.get("difficulty") == "hard"]
    n_correct_trials = sum(1 for trial in trials if trial.get("is_correct") is True)
    n_easy_correct_trials = sum(1 for trial in trials if trial.get("is_correct") is True and trial.get("difficulty") == "easy")
    n_hard_correct_trials = sum(1 for trial in trials if trial.get("is_correct") is True and trial.get("difficulty") == "hard")
    n_incorrect_trials = sum(1 for trial in trials if trial.get("is_correct") is False)
    n_easy_incorrect_trials = sum(1 for trial in trials if trial.get("is_correct") is False and trial.get("difficulty") == "easy")
    n_hard_incorrect_trials = sum(1 for trial in trials if trial.get("is_correct") is False and trial.get("difficulty") == "hard")
    compliant_responses_when_correct = 0
    easy_compliant_responses_when_correct = 0
    hard_compliant_responses_when_correct = 0
    override_responses_when_incorrect = 0
    easy_override_responses_when_incorrect = 0
    hard_override_responses_when_incorrect = 0
    correct_memory_test_reponses = 0
    correct_easy_memory_test_reponses = 0
    correct_hard_memory_test_reponses = 0
    compliant_responses_before_failure = 0
    n_trials_before_failure = 0
    compliant_responses_after_failure = 0
    n_trials_after_failure = 0
    failure = False

    for trial in trials:
        if trial.get("is_correct") is True and trial.get("ground_truth") == trial.get("result"):
            compliant_responses_when_correct += 1
            if trial.get("difficulty") == "easy":
                easy_compliant_responses_when_correct += 1
            elif trial.get("difficulty") == "hard":
                hard_compliant_responses_when_correct += 1
        elif trial.get("is_correct") is False and trial.get("ground_truth") == trial.get("result"):
            override_responses_when_incorrect += 1
            if trial.get("difficulty") == "easy":
                easy_override_responses_when_incorrect += 1
            elif trial.get("difficulty") == "hard":
                hard_override_responses_when_incorrect += 1

        if trial.get("memory_test") is not None and trial.get("memory_test") == trial.get("result"):
            correct_memory_test_reponses += 1
            if trial.get("difficulty") == "easy":
                correct_easy_memory_test_reponses += 1
            elif trial.get("difficulty") == "hard":
                correct_hard_memory_test_reponses += 1

        if not failure and trial.get("is_correct") is False:
            failure = True
        else:
            if failure:
                n_trials_after_failure += 1
                if trial.get("is_correct") is True and trial.get("ground_truth") == trial.get("result"):
                    compliant_responses_after_failure += 1
            else:
                n_trials_before_failure += 1
                if trial.get("is_correct") is True and trial.get("ground_truth") == trial.get("result"):
                    compliant_responses_before_failure += 1

    return {
        "n_trials": n_trials,
        "n_correct_trials": n_correct_trials,
        "n_incorrect_trials": n_incorrect_trials,
        "total_time_arm": total_time_arm,
        "times_arm": times_arm,
        "easy_times_arm": easy_times_arm,
        "hard_times_arm": hard_times_arm,
        "total_time_speaker": total_time_speaker,
        "times_speaker": times_speaker,
        "easy_times_speaker": easy_times_speaker,
        "hard_times_speaker": hard_times_speaker,
        "compliant_responses_when_correct": compliant_responses_when_correct,
        "override_responses_when_incorrect": override_responses_when_incorrect,
        "correct_memory_test_reponses": correct_memory_test_reponses,
        "compliant_responses_before_failure": compliant_responses_before_failure,
        "compliant_responses_after_failure": compliant_responses_after_failure,
        "n_trials_before_failure": n_trials_before_failure,
        "n_trials_after_failure": n_trials_after_failure,
        "avg_time_arm": total_time_arm / n_trials if n_trials > 0 else None,
        "avg_time_speaker": total_time_speaker / n_trials if n_trials > 0 else None,
        "compliance_rate_when_correct": compliant_responses_when_correct / n_correct_trials if n_correct_trials > 0 else None,
        "easy_compliance_rate_when_correct": easy_compliant_responses_when_correct / n_easy_correct_trials if n_easy_correct_trials > 0 else None,
        "hard_compliance_rate_when_correct": hard_compliant_responses_when_correct / n_hard_correct_trials if n_hard_correct_trials > 0 else None,
        "override_rate_when_incorrect": override_responses_when_incorrect / n_incorrect_trials if n_incorrect_trials > 0 else None,
        "easy_override_rate_when_incorrect": easy_override_responses_when_incorrect / n_easy_incorrect_trials if n_easy_incorrect_trials > 0 else None,
        "hard_override_rate_when_incorrect": hard_override_responses_when_incorrect / n_hard_incorrect_trials if n_hard_incorrect_trials > 0 else None,
        "memory_test_accuracy": correct_memory_test_reponses / n_trials if n_trials > 0 else None,
        "easy_memory_test_accuracy": correct_easy_memory_test_reponses / (n_trials / 2) if n_trials > 0 else None,
        "hard_memory_test_accuracy": correct_hard_memory_test_reponses / (n_trials / 2) if n_trials > 0 else None,
        "compliance_rate_before_failure": compliant_responses_before_failure / n_trials_before_failure if n_trials_before_failure > 0 else None,
        "compliance_rate_after_failure": compliant_responses_after_failure / n_trials_after_failure if n_trials_after_failure > 0 else None,
        "compliance_decay_rate": (compliant_responses_before_failure / n_trials_before_failure) - (compliant_responses_after_failure / n_trials_after_failure)
        if n_trials_before_failure > 0 and n_trials_after_failure > 0 else None,
    }


def compute_overall_metrics(report: dict[str, Any]) -> dict[str, Any]:
    participants = report.get("participants", [])
    overall_metrics = {"n_participants": len(participants)}
    overall_metrics["all_times_arm_justification"] = [time for p in participants for time in p["justification"]["times_arm"] if time is not None]
    overall_metrics["all_times_arm_no_justification"] = [time for p in participants for time in p["no_justification"]["times_arm"] if time is not None]
    overall_metrics["all_times_speaker_justification"] = [time for p in participants for time in p["justification"]["times_speaker"] if time is not None]
    overall_metrics["all_times_speaker_no_justification"] = [time for p in participants for time in p["no_justification"]["times_speaker"] if time is not None]
    overall_metrics["all_trust_decay_rates_justification"] = [p["justification"]["compliance_decay_rate"] for p in participants if p["justification"]["compliance_decay_rate"] is not None]
    overall_metrics["all_trust_decay_rates_no_justification"] = [p["no_justification"]["compliance_decay_rate"] for p in participants if p["no_justification"]["compliance_decay_rate"] is not None]
    report["overall_metrics"] = overall_metrics
    return report


def compute_overall_questionnaire_metrics(participants: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    question_keys = [
        "i_believe_that_there_could_be_negative_consequences_when_using_the_recycling_assistant_robot",
        "i_believe_that_the_recycling_assistant_robot_will_act_in_my_best_interest",
        "i_think_that_the_recycling_assistant_robot_is_competent_and_effective_in_sorting_the_trash_items",
        "when_sharing_something_with_the_recycling_assistant_robot_i_expect_to_get_back_a_meaningful_and_knowledgeable_response",
        "the_way_the_robot_moved_made_me_uncomfortable",
        "i_felt_i_could_rely_on_the_robot_to_do_what_it_was_supposed_to_do",
        "the_speed_at_which_the_gripper_picked_up_and_released_the_components_made_me_uneasy",
        "i_felt_safe_interacting_with_the_robot",
        "i_knew_the_gripper_would_not_drop_the_components",
        "the_size_of_the_robot_did_not_intimidate_me",
        "the_robot_gripper_did_not_look_reliable",
        "i_was_comfortable_the_robot_would_not_hurt_me",
        "i_trusted_that_the_robot_was_safe_to_cooperate_with",
        "the_gripper_seemed_like_it_could_be_trusted",
    ]
    justification_questionnaire = {key: [] for key in question_keys}
    no_justification_questionnaire = {key: [] for key in question_keys}

    for participant in participants:
        for question in question_keys:
            if question in participant.get("justification_questionnaire", {}):
                justification_questionnaire[question].append(participant["justification_questionnaire"][question])
            if question in participant.get("no_justification_questionnaire", {}):
                no_justification_questionnaire[question].append(participant["no_justification_questionnaire"][question])

    report["justification_questionnaire"] = justification_questionnaire
    report["no_justification_questionnaire"] = no_justification_questionnaire
    return report


def _pretty_question_label(question_key: str) -> str:
    label = question_key.replace("_", " ").strip()
    return label[:1].upper() + label[1:] if label else question_key


def _count_likert(values, scale=(1, 2, 3, 4, 5)):
    counts = {k: 0 for k in scale}
    for v in values:
        if v in counts:
            counts[v] += 1
    return [counts[k] for k in scale]


def _p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _sanitize_filename(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def wilcoxon_effect_size(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff_no_zero = diff[diff != 0]
    n_non_zero = len(diff_no_zero)

    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return None, None, None, None, 0

    if n_non_zero == 0:
        return 0.0, 1.0, 0.0, 0.0, 0

    stat, p = wilcoxon(x, y)
    mean_w = n_non_zero * (n_non_zero + 1) / 4
    std_w = np.sqrt(n_non_zero * (n_non_zero + 1) * (2 * n_non_zero + 1) / 24)
    z = (stat - mean_w) / std_w
    r = z / np.sqrt(n_non_zero)
    return float(stat), float(p), float(z), float(r), int(n_non_zero)


def extract_metric_series(participants: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
    ids, just_values, nojust_values = [], [], []
    for participant in participants:
        j_val = participant
        n_val = participant
        for key in spec["j_path"]:
            j_val = j_val.get(key) if isinstance(j_val, dict) else None
        for key in spec["n_path"]:
            n_val = n_val.get(key) if isinstance(n_val, dict) else None

        if j_val is None or n_val is None:
            continue

        if spec.get("aggregate") == "median":
            if not j_val or not n_val:
                continue
            j_num = float(np.median(j_val))
            n_num = float(np.median(n_val))
        else:
            j_num = float(j_val)
            n_num = float(n_val)

        ids.append(str(participant.get("id", len(ids) + 1)))
        just_values.append(j_num)
        nojust_values.append(n_num)
    return ids, just_values, nojust_values


def paired_plot(just, nojust, title, ylabel, output_dir: Path | None = None):
    x = [0, 1]
    plt.figure()
    plt.boxplot([nojust, just], positions=x)
    for i in range(len(just)):
        plt.plot(x, [nojust[i], just[i]], marker="o", alpha=0.5)
    plt.xticks(x, ["No Justification", "Justification"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / f"{_sanitize_filename(title)}_paired.png", dpi=300, bbox_inches="tight")
    plt.close()


def mean_error_plot(just, nojust, title, ylabel, output_dir: Path | None = None, use_sem=True):
    data = [nojust, just]
    labels = ["No Justification", "Justification"]
    means = [np.mean(d) for d in data]
    stds = [np.std(d, ddof=1) if len(d) > 1 else 0.0 for d in data]
    errors = [s / np.sqrt(len(d)) if use_sem and len(d) > 0 else s for s, d in zip(stds, data)]

    x = np.arange(len(labels))
    plt.figure()
    plt.errorbar(x, means, yerr=errors, fmt="o", capsize=5)
    plt.plot(x, means, linestyle="--", alpha=0.6)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / f"{_sanitize_filename(title)}_mean_error.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_metric_plotly_figure(metric_name: str, title: str, ylabel: str, participant_ids: list[str], just: list[float], nojust: list[float], p: float | None, r: float | None):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Paired participant values", "Mean ± SEM"), horizontal_spacing=0.16)

    for pid, j, n in zip(participant_ids, just, nojust):
        fig.add_trace(
            go.Scatter(
                x=["No Justification", "Justification"],
                y=[n, j],
                mode="lines+markers",
                line=dict(width=1),
                marker=dict(size=8),
                name=f"P{pid}",
                legendgroup=f"P{pid}",
                showlegend=False,
                hovertemplate=f"Participant {pid}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(go.Box(y=nojust, name="No Justification", boxpoints="all", jitter=0.25, pointpos=0, showlegend=False), row=1, col=1)
    fig.add_trace(go.Box(y=just, name="Justification", boxpoints="all", jitter=0.25, pointpos=0, showlegend=False), row=1, col=1)

    means = [float(np.mean(nojust)) if nojust else np.nan, float(np.mean(just)) if just else np.nan]
    sems = [float(np.std(nojust, ddof=1) / np.sqrt(len(nojust))) if len(nojust) > 1 else 0.0,
            float(np.std(just, ddof=1) / np.sqrt(len(just))) if len(just) > 1 else 0.0]
    fig.add_trace(
        go.Scatter(
            x=["No Justification", "Justification"],
            y=means,
            mode="markers+lines",
            error_y=dict(type="data", array=sems, visible=True),
            marker=dict(size=11),
            line=dict(dash="dash"),
            name="Mean ± SEM",
            showlegend=False,
            hovertemplate="%{x}<br>Mean: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    stats_text = f"p = {p:.4g}, r = {r:.3f}" if p is not None and r is not None else "Wilcoxon unavailable"
    fig.update_layout(title=f"{title}<br><sup>{metric_name} · {stats_text}</sup>", height=520, template="plotly_white")
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=2)
    return fig


def create_questionnaire_plotly_figure(question_label: str, counts_just: list[int], counts_nojust: list[int], p_value: float | None, statistic: float | None, mean_j: float | None, mean_nj: float | None, n_pairs: int):
    likert = [1, 2, 3, 4, 5]
    title_suffix = f"p = {p_value:.4g}, W = {statistic:.3f}, n = {n_pairs}" if p_value is not None and statistic is not None else f"n = {n_pairs}"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=likert, y=counts_just, name="Justification"))
    fig.add_trace(go.Bar(x=likert, y=counts_nojust, name="No Justification"))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title=f"{question_label}<br><sup>{title_suffix}; mean_j = {mean_j if mean_j is not None else 'NA'}, mean_nj = {mean_nj if mean_nj is not None else 'NA'}</sup>",
        xaxis_title="Likert score",
        yaxis_title="Count",
        height=420,
    )
    return fig


def save_questionnaire_grouped_barplots(participants: list[dict], report: dict, output_dir: str = "questionnaire_grouped_barplots") -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    justification_questionnaire = report.get("justification_questionnaire", {})
    no_justification_questionnaire = report.get("no_justification_questionnaire", {})
    all_questions = sorted(set(justification_questionnaire.keys()) | set(no_justification_questionnaire.keys()))
    likert_scale = [1, 2, 3, 4, 5]
    wilcoxon_results = {}

    for question in all_questions:
        question_label = _pretty_question_label(question)
        just_values = [v for v in justification_questionnaire.get(question, []) if v is not None]
        nojust_values = [v for v in no_justification_questionnaire.get(question, []) if v is not None]
        if not just_values and not nojust_values:
            continue

        counts_just = _count_likert(just_values, likert_scale)
        counts_nojust = _count_likert(nojust_values, likert_scale)

        paired_just, paired_nojust = [], []
        for participant in participants:
            j_q = participant.get("justification_questionnaire", {})
            nj_q = participant.get("no_justification_questionnaire", {})
            if j_q.get(question) is not None and nj_q.get(question) is not None:
                paired_just.append(float(j_q[question]))
                paired_nojust.append(float(nj_q[question]))

        statistic = p_value = None
        stars = "ns"
        if paired_just:
            diffs = np.asarray(paired_just) - np.asarray(paired_nojust)
            if np.any(diffs != 0):
                statistic, p_value = wilcoxon(np.asarray(paired_just), np.asarray(paired_nojust))
                statistic = float(statistic)
                p_value = float(p_value)
                stars = _p_to_stars(p_value)
            else:
                statistic, p_value = 0.0, 1.0

        plt.figure(figsize=(8, 5))
        x = np.arange(len(likert_scale))
        width = 0.35
        plt.bar(x - width / 2, counts_just, width=width, label="Justification")
        plt.bar(x + width / 2, counts_nojust, width=width, label="No Justification")
        plt.title(question_label)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.xticks(x, likert_scale)
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(output_path / f"{question}_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

        wilcoxon_results[question] = {
            "question_label": question_label,
            "n_pairs": len(paired_just),
            "statistic": statistic,
            "p_value": p_value,
            "significance": stars,
            "mean_justification": float(np.mean(paired_just)) if paired_just else None,
            "mean_no_justification": float(np.mean(paired_nojust)) if paired_nojust else None,
            "counts_justification": counts_just,
            "counts_no_justification": counts_nojust,
        }
    return wilcoxon_results


def run_metric_analysis(participants: list[dict], spec: dict[str, Any], static_output_dir: Path | None = None) -> dict[str, Any]:
    participant_ids, just, nojust = extract_metric_series(participants, spec)
    stat, p, z, r, n_non_zero = wilcoxon_effect_size(just, nojust)

    if static_output_dir is not None and just and nojust:
        paired_plot(just, nojust, f"{spec['title']} (p = {p:.3g}, r = {r:.3g})", spec["ylabel"], static_output_dir)
        mean_error_plot(just, nojust, f"Mean {spec['title']} (p = {p:.3g}, r = {r:.3g})", spec["ylabel"], static_output_dir)

    plotly_fig = create_metric_plotly_figure(spec["key"], spec["title"], spec["ylabel"], participant_ids, just, nojust, p, r) if just and nojust else None
    return {
        "key": spec["key"],
        "title": spec["title"],
        "hypothesis": spec["hypothesis"],
        "description": spec["description"],
        "ylabel": spec["ylabel"],
        "participant_ids": participant_ids,
        "justification_values": just,
        "no_justification_values": nojust,
        "wilcoxon": {
            "statistic": stat,
            "p_value": p,
            "z_value": z,
            "effect_size_r": r,
            "n_non_zero_differences": n_non_zero,
            "significance": _p_to_stars(p) if p is not None else None,
        },
        "plotly_fig": plotly_fig,
    }


def build_html(report: dict[str, Any], metric_results: list[dict[str, Any]], questionnaire_results: dict[str, Any]) -> str:
    participants = report.get("participants", [])
    n_participants = len(participants)

    summary_rows = []
    for result in metric_results:
        w = result["wilcoxon"]
        summary_rows.append(
            f"<tr><td>{result['key']}</td><td>{result['hypothesis']}</td><td>{w['statistic'] if w['statistic'] is not None else 'NA'}</td><td>{w['p_value'] if w['p_value'] is not None else 'NA'}</td><td>{w['effect_size_r'] if w['effect_size_r'] is not None else 'NA'}</td><td>{w['significance'] if w['significance'] is not None else 'NA'}</td></tr>"
        )

    sections = []
    for result in metric_results:
        w = result["wilcoxon"]
        fig_html = result["plotly_fig"].to_html(full_html=False, include_plotlyjs=False) if result["plotly_fig"] is not None else "<p>No paired values available.</p>"
        sections.append(
            f"""
            <div class='metric-card'>
              <h3>{result['title']}</h3>
              <p class='small'><strong>{result['hypothesis']}</strong> — {result['description']}</p>
              <div class='stats'>
                <strong>Wilcoxon signed-rank:</strong>
                W = {w['statistic'] if w['statistic'] is not None else 'NA'},
                p = {w['p_value'] if w['p_value'] is not None else 'NA'},
                z = {w['z_value'] if w['z_value'] is not None else 'NA'},
                r = {w['effect_size_r'] if w['effect_size_r'] is not None else 'NA'},
                significance = {w['significance'] if w['significance'] is not None else 'NA'},
                n(non-zero diffs) = {w['n_non_zero_differences']}
              </div>
              {fig_html}
            </div>
            """
        )

    questionnaire_sections = []
    for key, res in questionnaire_results.items():
        fig = create_questionnaire_plotly_figure(
            res["question_label"],
            res["counts_justification"],
            res["counts_no_justification"],
            res["p_value"],
            res["statistic"],
            res["mean_justification"],
            res["mean_no_justification"],
            res["n_pairs"],
        )
        questionnaire_sections.append(
            f"<div class='metric-card'><h3>{res['question_label']}</h3>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
        )

    return f"""
    <html>
      <head>
        <meta charset='utf-8'>
        <title>Experiment Metrics Report</title>
        <script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
        {HTML_STYLE}
      </head>
      <body>
        <h1>Experiment Metrics Report</h1>
        <p>Participants analyzed: <strong>{n_participants}</strong></p>

        <div class='section'>
          <h2>Metric Summary</h2>
          <table class='summary-table'>
            <thead>
              <tr><th>Metric</th><th>Hypothesis</th><th>W</th><th>p</th><th>r</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {''.join(summary_rows)}
            </tbody>
          </table>
        </div>

        <div class='section'>
          <h2>Metric Figures</h2>
          {''.join(sections)}
        </div>

        <div class='section'>
          <h2>Questionnaire Figures</h2>
          {''.join(questionnaire_sections)}
        </div>
      </body>
    </html>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment metrics and an HTML report from participant JSON.")
    parser.add_argument("--input", required=True, help="Path to the participants JSON file")
    parser.add_argument("--output-html", default="experiment_report.html", help="Path to the output HTML report")
    parser.add_argument("--output-json", default="experiment_metrics.json", help="Path to the output metrics JSON")
    parser.add_argument("--static-dir", default="metric_plots", help="Directory for PNG outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    static_dir = Path(args.static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)

    participants = load_dataset(input_path)
    report = analyze_dataset(participants)
    report = compute_overall_metrics(report)
    report = compute_overall_questionnaire_metrics(participants, report)

    metric_results = [run_metric_analysis(report["participants"], spec, static_dir) for spec in METRIC_SPECS]
    questionnaire_results = save_questionnaire_grouped_barplots(participants, report, output_dir=str(static_dir / "questionnaire_grouped_barplots"))

    report["metric_results"] = [
        {
            k: v for k, v in result.items() if k != "plotly_fig"
        }
        for result in metric_results
    ]
    report["questionnaire_results"] = questionnaire_results

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_html.write_text(build_html(report, metric_results, questionnaire_results), encoding="utf-8")

    print(f"Loaded participants from {input_path}")
    print(f"Metrics JSON written to {output_json}")
    print(f"HTML report written to {output_html}")
    print(f"Static PNGs written to {static_dir}")


if __name__ == "__main__":
    main()
