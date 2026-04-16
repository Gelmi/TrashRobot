import argparse
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import shapiro
from scipy.stats import ttest_ind

def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of participants.")
    return data

def analyze_dataset(participants: list[dict[str, Any]]) -> dict[str, Any]:
    participants_metrics = [{} for _ in participants]
    for idx, participant in enumerate(participants):
        participants_metrics[idx] = analyze_participant(participant)
    return {"participants": participants_metrics}

def analyze_participant(participant: dict[str, Any]) -> dict[str, Any]:
    participant_metrics = {}
    participant_metrics["id"] = participant["participant_id"]
    participant_metrics["gender"] = participant["gender"]
    participant_metrics["age"] = participant["age"]
    participant_metrics["justification"] = compute_metrics(participant.get("justification", []))
    participant_metrics["no_justification"] = compute_metrics(participant.get("no_justification", []))  
    return participant_metrics

def compute_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {"n_trials": 0, "avg_response_time": None, "correct_responses": None}

    n_trials = len(trials)
    total_time_arm = sum(trial.get("time_arm", 0) for trial in trials)
    time_arm_zeros = sum(1 for trial in trials if trial.get("time_arm") == 0)
    easy_time_arm_zeros = sum(1 for trial in trials if trial.get("time_arm") == 0 and trial.get("difficulty") == "easy")
    hard_time_arm_zeros = sum(1 for trial in trials if trial.get("time_arm") == 0 and trial.get("difficulty") == "hard")
    times_arm = [trial.get("time_arm", 0) for trial in trials if trial.get("time_arm") is not None and trial.get("time_arm") != 0]
    easy_times_arm = [trial.get("time_arm", 0) for trial in trials if trial.get("time_arm") is not None and trial.get("difficulty") == "easy" and trial.get("time_arm") != 0]
    hard_times_arm = [trial.get("time_arm", 0) for trial in trials if trial.get("time_arm") is not None and trial.get("difficulty") == "hard" and trial.get("time_arm") != 0]
    total_time_speaker = sum(trial.get("time_speaker", 0) for trial in trials)
    time_speaker_zeros = sum(1 for trial in trials if trial.get("time_speaker") == 0)
    time_speaker_zeros_arm_nonzeros = sum(1 for trial in trials if trial.get("time_speaker") == 0 and trial.get("time_arm") != 0)
    easy_time_speaker_zeros_arm_nonzeros = sum(1 for trial in trials if trial.get("time_speaker") == 0 and trial.get("time_arm") != 0 and trial.get("difficulty") == "easy")
    hard_time_speaker_zeros_arm_nonzeros = sum(1 for trial in trials if trial.get("time_speaker") == 0 and trial.get("time_arm") != 0 and trial.get("difficulty") == "hard")
    easy_time_speaker_zeros = sum(1 for trial in trials if trial.get("time_speaker") == 0 and trial.get("difficulty") == "easy")
    hard_time_speaker_zeros = sum(1 for trial in trials if trial.get("time_speaker") == 0 and trial.get("difficulty") == "hard")
    times_speaker = [trial.get("time_speaker", 0) for trial in trials if trial.get("time_speaker") is not None and trial.get("time_arm") != 0] 
    easy_times_speaker = [trial.get("time_speaker", 0) for trial in trials if trial.get("time_speaker") is not None and trial.get("difficulty") == "easy" and trial.get("time_arm") != 0]
    hard_times_speaker = [trial.get("time_speaker", 0) for trial in trials if trial.get("time_speaker") is not None and trial.get("difficulty") == "hard" and trial.get("time_arm") != 0]
    n_correct_trials = sum(1 for trial in trials if trial.get("is_correct") == True)
    n_easy_correct_trials = sum(1 for trial in trials if trial.get("is_correct") == True and trial.get("difficulty") == "easy")
    n_hard_correct_trials = sum(1 for trial in trials if trial.get("is_correct") == True and trial.get("difficulty") == "hard")
    n_incorrect_trials = sum(1 for trial in trials if trial.get("is_correct") == False)
    n_easy_incorrect_trials = sum(1 for trial in trials if trial.get("is_correct") == False and trial.get("difficulty") == "easy")
    n_hard_incorrect_trials = sum(1 for trial in trials if trial.get("is_correct") == False and trial.get("difficulty") == "hard")
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
    easy_compliant_responses_before_failure = 0
    n_easy_trials_before_failure = 0
    hard_compliant_responses_before_failure = 0
    n_hard_trials_before_failure = 0
    compliant_responses_after_failure = 0
    n_trials_after_failure = 0
    easy_compliant_responses_after_failure = 0
    n_easy_trials_after_failure = 0
    hard_compliant_responses_after_failure = 0
    n_hard_trials_after_failure = 0
    failure=False
    for trial in trials:
        if trial.get("is_correct") == True and trial.get("ground_truth") == trial.get("result"):
            compliant_responses_when_correct += 1
            if trial.get("difficulty") == "easy":
                easy_compliant_responses_when_correct += 1
            elif trial.get("difficulty") == "hard":
                hard_compliant_responses_when_correct += 1
        elif trial.get("is_correct") == False and trial.get("ground_truth") == trial.get("result"):
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
        if failure == False and trial.get("is_correct") == False:
            failure = True
        else:
            if failure:
                n_trials_after_failure += 1
                if trial.get("difficulty") == "easy":
                    n_easy_trials_after_failure += 1
                elif trial.get("difficulty") == "hard":
                    n_hard_trials_after_failure += 1
                if trial.get("is_correct") == True and trial.get("ground_truth") == trial.get("result"):
                    compliant_responses_after_failure += 1
                    if trial.get("difficulty") == "easy":
                        easy_compliant_responses_after_failure += 1
                    elif trial.get("difficulty") == "hard":
                        hard_compliant_responses_after_failure += 1

            else:
                n_trials_before_failure += 1
                if trial.get("difficulty") == "easy":
                    n_easy_trials_before_failure += 1
                elif trial.get("difficulty") == "hard":
                    n_hard_trials_before_failure += 1
                if trial.get("is_correct") == True and trial.get("ground_truth") == trial.get("result"):
                    compliant_responses_before_failure += 1
                    if trial.get("difficulty") == "easy":
                        easy_compliant_responses_before_failure += 1
                    elif trial.get("difficulty") == "hard":
                        hard_compliant_responses_before_failure += 1
    return {
        "n_trials": n_trials,
        "n_correct_trials": n_correct_trials,
        "n_incorrect_trials": n_incorrect_trials,
        "total_time_arm": total_time_arm,
        "time_arm_zeros": time_arm_zeros,
        "time_arm_zeros_rate": time_arm_zeros / n_trials if n_trials > 0 else None,
        "easy_time_arm_zeros": easy_time_arm_zeros,
        "easy_time_arm_zeros_rate": easy_time_arm_zeros / n_trials if n_trials > 0 else None,
        "hard_time_arm_zeros": hard_time_arm_zeros,
        "hard_time_arm_zeros_rate": hard_time_arm_zeros / n_trials if n_trials > 0 else None,
        "times_arm": times_arm,
        "easy_times_arm": easy_times_arm,
        "hard_times_arm": hard_times_arm,
        "total_time_speaker": total_time_speaker,
        "times_speaker": times_speaker,
        "time_speaker_zeros": time_speaker_zeros,
        "time_speaker_zeros_rate": time_speaker_zeros / n_trials if n_trials > 0 else None,
        "easy_time_speaker_zeros": easy_time_speaker_zeros,
        "easy_time_speaker_zeros_rate": easy_time_speaker_zeros / n_trials if n_trials > 0 else None,
        "hard_time_speaker_zeros": hard_time_speaker_zeros,
        "hard_time_speaker_zeros_rate": hard_time_speaker_zeros / n_trials if n_trials > 0 else None,
        "time_speaker_zeros_arm_nonzeros": time_speaker_zeros_arm_nonzeros,
        "time_speaker_zeros_arm_nonzeros_rate": time_speaker_zeros_arm_nonzeros / n_trials if n_trials > 0 else None,
        "time_speaker_zeros_arm_nonzeros_ratio": time_speaker_zeros_arm_nonzeros/time_speaker_zeros if time_speaker_zeros > 0 else None,
        "easy_time_speaker_zeros_arm_nonzeros": easy_time_speaker_zeros_arm_nonzeros,
        "easy_time_speaker_zeros_arm_nonzeros_rate": easy_time_speaker_zeros_arm_nonzeros / n_trials if n_trials > 0 else None,
        "hard_time_speaker_zeros_arm_nonzeros": hard_time_speaker_zeros_arm_nonzeros,
        "hard_time_speaker_zeros_arm_nonzeros_rate": hard_time_speaker_zeros_arm_nonzeros / n_trials if n_trials > 0 else None,
        "easy_times_speaker": easy_times_speaker,
        "hard_times_speaker": hard_times_speaker,
        "compliant_responses_when_correct": compliant_responses_when_correct,
        "override_responses_when_incorrect": override_responses_when_incorrect,
        "correct_memory_test_reponses": correct_memory_test_reponses,
        "compliant_responses_before_failure": compliant_responses_before_failure,
        "easy_compliant_responses_before_failure": easy_compliant_responses_before_failure,
        "hard_compliant_responses_before_failure": hard_compliant_responses_before_failure,
        "compliant_responses_after_failure": compliant_responses_after_failure,
        "easy_compliant_responses_after_failure": easy_compliant_responses_after_failure,
        "hard_compliant_responses_after_failure": hard_compliant_responses_after_failure,
        "n_trials_before_failure": n_trials_before_failure,
        "n_trials_after_failure": n_trials_after_failure,
        "n_easy_trials_before_failure": n_easy_trials_before_failure,
        "n_hard_trials_before_failure": n_hard_trials_before_failure,
        "n_easy_trials_after_failure": n_easy_trials_after_failure,
        "n_hard_trials_after_failure": n_hard_trials_after_failure,
        "avg_time_arm": total_time_arm / n_trials if n_trials > 0 else None,
        "avg_time_speaker": total_time_speaker / n_trials if n_trials > 0 else None,
        "compliance_rate_when_correct": compliant_responses_when_correct / n_correct_trials if n_correct_trials > 0 else None,
        "easy_compliance_rate_when_correct": easy_compliant_responses_when_correct / n_easy_correct_trials if n_easy_correct_trials > 0 else None,
        "hard_compliance_rate_when_correct": hard_compliant_responses_when_correct / n_hard_correct_trials if n_hard_correct_trials > 0 else None,
        "override_rate_when_incorrect": override_responses_when_incorrect / n_incorrect_trials if n_incorrect_trials > 0 else None,
        "easy_override_rate_when_incorrect": easy_override_responses_when_incorrect / n_easy_incorrect_trials if n_easy_incorrect_trials > 0 else None,
        "hard_override_rate_when_incorrect": hard_override_responses_when_incorrect / n_hard_incorrect_trials if n_hard_incorrect_trials > 0 else None,
        "memory_test_accuracy": correct_memory_test_reponses / n_trials if n_trials > 0 else None,
        "easy_memory_test_accuracy": correct_easy_memory_test_reponses / (n_trials/2) if n_trials > 0 else None,
        "hard_memory_test_accuracy": correct_hard_memory_test_reponses / (n_trials/2) if n_trials > 0 else None,
        "compliance_rate_before_failure": compliant_responses_before_failure / n_trials_before_failure if n_trials_before_failure > 0 else None,
        "compliance_rate_after_failure": compliant_responses_after_failure / n_trials_after_failure if n_trials_after_failure > 0 else None,
        "compliance_decay_rate": (compliant_responses_before_failure / n_trials_before_failure) - (compliant_responses_after_failure / n_trials_after_failure) if n_trials_before_failure > 0 and n_trials_after_failure > 0 else None,
        "easy_compliance_decay_rate": (easy_compliant_responses_before_failure / n_easy_trials_before_failure) - (easy_compliant_responses_after_failure / n_easy_trials_after_failure) if n_easy_trials_before_failure > 0 and n_easy_trials_after_failure > 0 else None,
        "hard_compliance_decay_rate": (hard_compliant_responses_before_failure / n_hard_trials_before_failure) - (hard_compliant_responses_after_failure / n_hard_trials_after_failure) if n_hard_trials_before_failure > 0 and n_hard_trials_after_failure > 0 else None
    }

def compute_overall_metrics(report: dict[str, Any]) -> dict[str, Any]:
    participants = report.get("participants", [])
    n_participants = len(participants)
    overall_metrics = {
        "n_participants": n_participants,
    }
    times_arm_justification = [time for p in participants for time in p["justification"]["times_arm"] if time is not None]
    times_arm_no_justification = [time for p in participants for time in p["no_justification"]["times_arm"] if time is not None]
    times_speaker_justification = [time for p in participants for time in p["justification"]["times_speaker"] if time is not None]
    times_speaker_no_justification = [time for p in participants for time in p["no_justification"]["times_speaker"] if time is not None]
    trust_decay_rates_justification = [p["justification"]["compliance_decay_rate"] for p in participants if p["justification"]["compliance_decay_rate"] is not None]
    trust_decay_rates_no_justification = [p["no_justification"]["compliance_decay_rate"] for p in participants if p["no_justification"]["compliance_decay_rate"] is not None]
    overall_metrics["all_times_arm_justification"] = times_arm_justification
    overall_metrics["all_times_arm_no_justification"] = times_arm_no_justification
    overall_metrics["all_times_speaker_justification"] = times_speaker_justification
    overall_metrics["all_times_speaker_no_justification"] = times_speaker_no_justification
    overall_metrics["all_trust_decay_rates_justification"] = trust_decay_rates_justification
    overall_metrics["all_trust_decay_rates_no_justification"] = trust_decay_rates_no_justification

    if n_participants > 0:
        overall_metrics["total_n_trials_justification"] = sum(p["justification"]["n_trials"] for p in participants)
        overall_metrics["total_n_trials_no_justification"] = sum(p["no_justification"]["n_trials"] for p in participants)
        overall_metrics["total_n_correct_trials_justification"] = sum(p["justification"]["n_correct_trials"] for p in participants)
        overall_metrics["total_n_correct_trials_no_justification"] = sum(p["no_justification"]["n_correct_trials"] for p in participants)
        overall_metrics["total_n_incorrect_trials_justification"] = sum(p["justification"]["n_incorrect_trials"] for p in participants)
        overall_metrics["total_n_incorrect_trials_no_justification"] = sum(p["no_justification"]["n_incorrect_trials"] for p in participants)

        overall_metrics["total_time_arm_justification"] = sum(p["justification"]["total_time_arm"] for p in participants if p["justification"]["total_time_arm"] is not None)
        overall_metrics["total_time_arm_no_justification"] = sum(p["no_justification"]["total_time_arm"] for p in participants if p["no_justification"]["total_time_arm"] is not None)
        overall_metrics["total_time_speaker_justification"] = sum(p["justification"]["total_time_speaker"] for p in participants if p["justification"]["total_time_speaker"] is not None)
        overall_metrics["total_time_speaker_no_justification"] = sum(p["no_justification"]["total_time_speaker"] for p in participants if p["no_justification"]["total_time_speaker"] is not None)
        overall_metrics["total_compliant_responses_when_correct_justification"] = sum(p["justification"]["compliant_responses_when_correct"] for p in participants if p["justification"]["compliant_responses_when_correct"] is not None)
        overall_metrics["total_compliant_responses_when_correct_no_justification"] = sum(p["no_justification"]["compliant_responses_when_correct"] for p in participants if p["no_justification"]["compliant_responses_when_correct"] is not None)
        overall_metrics["total_override_responses_when_incorrect_justification"] = sum(p["justification"]["override_responses_when_incorrect"] for p in participants if p["justification"]["override_responses_when_incorrect"] is not None)
        overall_metrics["total_override_responses_when_incorrect_no_justification"] = sum(p["no_justification"]["override_responses_when_incorrect"] for p in participants if p["no_justification"]["override_responses_when_incorrect"] is not None)
        overall_metrics["total_correct_memory_test_reponses_justification"] = sum(p["justification"]["correct_memory_test_reponses"] for p in participants if p["justification"]["correct_memory_test_reponses"] is not None)
        overall_metrics["total_correct_memory_test_reponses_no_justification"] = sum(p["no_justification"]["correct_memory_test_reponses"] for p in participants if p["no_justification"]["correct_memory_test_reponses"] is not None)
        overall_metrics["total_compliant_responses_before_failure_justification"] = sum(p["justification"]["compliant_responses_before_failure"] for p in participants if p["justification"]["compliant_responses_before_failure"] is not None)
        overall_metrics["total_compliant_responses_before_failure_no_justification"] = sum(p["no_justification"]["compliant_responses_before_failure"] for p in participants if p["no_justification"]["compliant_responses_before_failure"] is not None)
        overall_metrics["total_n_trials_before_failure_justification"] = sum(p["justification"]["n_trials_before_failure"] for p in participants if p["justification"]["n_trials_before_failure"] is not None)
        overall_metrics["total_n_trials_before_failure_no_justification"] = sum(p["no_justification"]["n_trials_before_failure"] for p in participants if p["no_justification"]["n_trials_before_failure"] is not None)
        overall_metrics["total_compliant_responses_after_failure_justification"] = sum(p["justification"]["compliant_responses_after_failure"] for p in participants if p["justification"]["compliant_responses_after_failure"] is not None)
        overall_metrics["total_compliant_responses_after_failure_no_justification"] = sum(p["no_justification"]["compliant_responses_after_failure"] for p in participants if p["no_justification"]["compliant_responses_after_failure"] is not None)
        overall_metrics["total_n_trials_after_failure_justification"] = sum(p["justification"]["n_trials_after_failure"] for p in participants if p["justification"]["n_trials_after_failure"] is not None)
        overall_metrics["total_n_trials_after_failure_no_justification"] = sum(p["no_justification"]["n_trials_after_failure"] for p in participants if p["no_justification"]["n_trials_after_failure"] is not None)

        overall_metrics["total_avg_time_arm_justification"] = overall_metrics["total_time_arm_justification"] / overall_metrics["total_n_trials_justification"] if overall_metrics["total_n_trials_justification"] > 0 else None
        overall_metrics["total_avg_time_arm_no_justification"] = overall_metrics["total_time_arm_no_justification"] / overall_metrics["total_n_trials_no_justification"] if overall_metrics["total_n_trials_no_justification"] > 0 else None
        overall_metrics["total_compliant_rate_when_correct_justification"] = overall_metrics["total_compliant_responses_when_correct_justification"] / overall_metrics["total_n_correct_trials_justification"] if overall_metrics["total_n_correct_trials_justification"] > 0 else None
        overall_metrics["total_compliant_rate_when_correct_no_justification"] = overall_metrics["total_compliant_responses_when_correct_no_justification"] / overall_metrics["total_n_correct_trials_no_justification"] if overall_metrics["total_n_correct_trials_no_justification"] > 0 else None
        overall_metrics["total_override_rate_when_incorrect_justification"] = overall_metrics["total_override_responses_when_incorrect_justification"] / overall_metrics["total_n_incorrect_trials_justification"] if overall_metrics["total_n_incorrect_trials_justification"] > 0 else None
        overall_metrics["total_override_rate_when_incorrect_no_justification"] = overall_metrics["total_override_responses_when_incorrect_no_justification"] / overall_metrics["total_n_incorrect_trials_no_justification"] if overall_metrics["total_n_incorrect_trials_no_justification"] > 0 else None
        overall_metrics["total_memory_test_accuracy_justification"] = overall_metrics["total_correct_memory_test_reponses_justification"] / overall_metrics["total_n_trials_justification"] if overall_metrics["total_n_trials_justification"] > 0 else None
        overall_metrics["total_memory_test_accuracy_no_justification"] = overall_metrics["total_correct_memory_test_reponses_no_justification"] / overall_metrics["total_n_trials_no_justification"] if overall_metrics["total_n_trials_no_justification"] > 0 else None
        overall_metrics["total_compliance_rate_before_failure_justification"] = overall_metrics["total_compliant_responses_before_failure_justification"] / overall_metrics["total_n_trials_before_failure_justification"] if overall_metrics["total_n_trials_before_failure_justification"] > 0 else None
        overall_metrics["total_compliance_rate_before_failure_no_justification"] = overall_metrics["total_compliant_responses_before_failure_no_justification"] / overall_metrics["total_n_trials_before_failure_no_justification"] if overall_metrics["total_n_trials_before_failure_no_justification"] > 0 else None
        overall_metrics["total_compliance_rate_after_failure_justification"] = overall_metrics["total_compliant_responses_after_failure_justification"] / overall_metrics["total_n_trials_after_failure_justification"] if overall_metrics["total_n_trials_after_failure_justification"] > 0 else None
        overall_metrics["total_compliance_rate_after_failure_no_justification"] = overall_metrics["total_compliant_responses_after_failure_no_justification"] / overall_metrics["total_n_trials_after_failure_no_justification"] if overall_metrics["total_n_trials_after_failure_no_justification"] > 0 else None

        overall_metrics["avg_time_arm_justification"] = sum(p["justification"]["avg_time_arm"] for p in participants if p["justification"]["avg_time_arm"] is not None) / n_participants
        overall_metrics["avg_time_arm_no_justification"] = sum(p["no_justification"]["avg_time_arm"] for p in participants if p["no_justification"]["avg_time_arm"] is not None) / n_participants
        overall_metrics["avg_time_speaker_justification"] = sum(p["justification"]["avg_time_speaker"] for p in participants if p["justification"]["avg_time_speaker"] is not None) / n_participants
        overall_metrics["avg_time_speaker_no_justification"] = sum(p["no_justification"]["avg_time_speaker"] for p in participants if p["no_justification"]["avg_time_speaker"] is not None) / n_participants
        overall_metrics["avg_compliance_rate_when_correct_justification"] = sum(p["justification"]["compliance_rate_when_correct"] for p in participants if p["justification"]["compliance_rate_when_correct"] is not None) / n_participants
        overall_metrics["avg_compliance_rate_when_correct_no_justification"] = sum(p["no_justification"]["compliance_rate_when_correct"] for p in participants if p["no_justification"]["compliance_rate_when_correct"] is not None) / n_participants
        overall_metrics["avg_override_rate_when_incorrect_justification"] = sum(p["justification"]["override_rate_when_incorrect"] for p in participants if p["justification"]["override_rate_when_incorrect"] is not None) / n_participants
        overall_metrics["avg_override_rate_when_incorrect_no_justification"] = sum(p["no_justification"]["override_rate_when_incorrect"] for p in participants if p["no_justification"]["override_rate_when_incorrect"] is not None) / n_participants
        overall_metrics["avg_memory_test_accuracy_justification"] = sum(p["justification"]["memory_test_accuracy"] for p in participants if p["justification"]["memory_test_accuracy"] is not None) / n_participants
        overall_metrics["avg_memory_test_accuracy_no_justification"] = sum(p["no_justification"]["memory_test_accuracy"] for p in participants if p["no_justification"]["memory_test_accuracy"] is not None) / n_participants
        overall_metrics["avg_compliance_rate_before_failure_justification"] = sum(p["justification"]["compliance_rate_before_failure"] for p in participants if p["justification"]["compliance_rate_before_failure"] is not None) / n_participants
        overall_metrics["avg_compliance_rate_before_failure_no_justification"] = sum(p["no_justification"]["compliance_rate_before_failure"] for p in participants if p["no_justification"]["compliance_rate_before_failure"] is not None) / n_participants
        overall_metrics["avg_compliance_rate_after_failure_justification"] = sum(p["justification"]["compliance_rate_after_failure"] for p in participants if p["justification"]["compliance_rate_after_failure"] is not None) / n_participants
        overall_metrics["avg_compliance_rate_after_failure_no_justification"] = sum(p["no_justification"]["compliance_rate_after_failure"] for p in participants if p["no_justification"]["compliance_rate_after_failure"] is not None) / n_participants
    report["overall_metrics"] = overall_metrics

    return report

def compute_overall_questionnaire_metrics(participants: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:

    justification_questionnaire = {
      "i_believe_that_there_could_be_negative_consequences_when_using_the_recycling_assistant_robot": [],
      "i_believe_that_the_recycling_assistant_robot_will_act_in_my_best_interest": [],
      "i_think_that_the_recycling_assistant_robot_is_competent_and_effective_in_sorting_the_trash_items": [],
      "when_sharing_something_with_the_recycling_assistant_robot_i_expect_to_get_back_a_meaningful_and_knowledgeable_response": [],
      "the_way_the_robot_moved_made_me_uncomfortable": [],
      "i_felt_i_could_rely_on_the_robot_to_do_what_it_was_supposed_to_do": [],
      "the_speed_at_which_the_gripper_picked_up_and_released_the_components_made_me_uneasy": [],
      "i_felt_safe_interacting_with_the_robot": [],
      "i_knew_the_gripper_would_not_drop_the_components": [],
      "the_size_of_the_robot_did_not_intimidate_me": [],
      "the_robot_gripper_did_not_look_reliable": [],
      "i_was_comfortable_the_robot_would_not_hurt_me": [],
      "i_trusted_that_the_robot_was_safe_to_cooperate_with": [],
      "the_gripper_seemed_like_it_could_be_trusted": []
    }
    no_justification_questionnaire = {
      "i_believe_that_there_could_be_negative_consequences_when_using_the_recycling_assistant_robot": [],
      "i_believe_that_the_recycling_assistant_robot_will_act_in_my_best_interest": [],
      "i_think_that_the_recycling_assistant_robot_is_competent_and_effective_in_sorting_the_trash_items": [],
      "when_sharing_something_with_the_recycling_assistant_robot_i_expect_to_get_back_a_meaningful_and_knowledgeable_response": [],
      "the_way_the_robot_moved_made_me_uncomfortable": [],
      "i_felt_i_could_rely_on_the_robot_to_do_what_it_was_supposed_to_do": [],
      "the_speed_at_which_the_gripper_picked_up_and_released_the_components_made_me_uneasy": [],
      "i_felt_safe_interacting_with_the_robot": [],
      "i_knew_the_gripper_would_not_drop_the_components": [],
      "the_size_of_the_robot_did_not_intimidate_me": [],
      "the_robot_gripper_did_not_look_reliable": [],
      "i_was_comfortable_the_robot_would_not_hurt_me": [],
      "i_trusted_that_the_robot_was_safe_to_cooperate_with": [],
      "the_gripper_seemed_like_it_could_be_trusted": []
    }

    for participant in participants:
        for question in justification_questionnaire.keys():
            print(participant.get("justification_questionnaire", {}))
            if question in participant.get("justification_questionnaire", {}):
                justification_questionnaire[question].append(participant["justification_questionnaire"][question])
        for question in no_justification_questionnaire.keys():
            if question in participant.get("no_justification_questionnaire", {}):
                no_justification_questionnaire[question].append(participant["no_justification_questionnaire"][question])
    report["justification_questionnaire"] = justification_questionnaire
    report["no_justification_questionnaire"] = no_justification_questionnaire
    return report

def plot_time_boxplots(overall_metrics):
    times_arm_justification = overall_metrics["all_times_arm_justification"]
    times_arm_no_justification = overall_metrics["all_times_arm_no_justification"]
    times_speaker_justification = overall_metrics["all_times_speaker_justification"]
    times_speaker_no_justification = overall_metrics["all_times_speaker_no_justification"]

    # ---- Plot 1: Arm times ----
    plt.figure()
    plt.boxplot(
        [times_arm_justification, times_arm_no_justification],
        labels=["Arm (Justification)", "Arm (No Justification)"]
    )
    plt.title("Arm Time Distribution")
    plt.ylabel("Time (s)")
    plt.xlabel("Condition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("arm_time_boxplot.png")  # Save the arm time boxplot as an image

    # ---- Plot 2: Speaker times ----
    plt.figure()
    plt.boxplot(
        [times_speaker_justification, times_speaker_no_justification],
        labels=["Speaker (Justification)", "Speaker (No Justification)"]
    )
    plt.title("Speaker Time Distribution")
    plt.ylabel("Time (s)")
    plt.xlabel("Condition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speaker_time_boxplot.png")  # Save the speaker time boxplot as an image

def plot_decay_boxplots(overall_metrics):
    total_decay_rates_justification = overall_metrics["all_trust_decay_rates_justification"]
    total_decay_rates_no_justification = overall_metrics["all_trust_decay_rates_no_justification"]

    # ---- Plot 1: Arm times ----
    plt.figure()
    plt.boxplot(
        [total_decay_rates_justification, total_decay_rates_no_justification],
        labels=["Trust Decay (Justification)", "Trust Decay (No Justification)"]
    )
    plt.title("Trust Decay Rate Distribution")
    plt.ylabel("Decay Rate")
    plt.xlabel("Condition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trust_decay_boxplot.png")  # Save the trust decay rate boxplot as an image
    
    means = [
        np.mean(total_decay_rates_justification),
        np.mean(total_decay_rates_no_justification)
    ]

    stds = [
        np.std(total_decay_rates_justification),
        np.std(total_decay_rates_no_justification)
    ]

    labels = ["Justification", "No Justification"]
    x = np.arange(len(labels))

    plt.figure()

    # Plot mean as points with vertical error bars
    plt.errorbar(
        x,
        means,
        yerr=stds,
        fmt='o',      # 👈 point marker
        capsize=5
    )

    plt.xticks(x, labels)
    plt.ylabel("Trust Decay Rate")
    plt.title("Mean Trust Decay Rate ± Std")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("trust_decay_mean_std.png")
    plt.close()

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


def save_questionnaire_grouped_barplots(
    participants: list[dict],
    report: dict,
    output_dir: str = "questionnaire_grouped_barplots"
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    justification_questionnaire = report.get("justification_questionnaire", {})
    no_justification_questionnaire = report.get("no_justification_questionnaire", {})

    all_questions = sorted(
        set(justification_questionnaire.keys()) | set(no_justification_questionnaire.keys())
    )

    likert_scale = [1, 2, 3, 4, 5]
    wilcoxon_results = {}

    for question in all_questions:
        question_label = _pretty_question_label(question)

        # Aggregated values for plotting counts
        just_values = [
            v for v in justification_questionnaire.get(question, [])
            if v is not None
        ]
        nojust_values = [
            v for v in no_justification_questionnaire.get(question, [])
            if v is not None
        ]

        if not just_values and not nojust_values:
            continue

        counts_just = _count_likert(just_values, likert_scale)
        counts_nojust = _count_likert(nojust_values, likert_scale)

        # Paired participant-level values for Wilcoxon
        paired_just = []
        paired_nojust = []

        for participant in participants:
            j_q = participant.get("justification_questionnaire", {})
            nj_q = participant.get("no_justification_questionnaire", {})

            j_val = j_q.get(question)
            nj_val = nj_q.get(question)

            if j_val is not None and nj_val is not None:
                paired_just.append(j_val)
                paired_nojust.append(nj_val)

        p_value = None
        statistic = None
        stars = "ns"

        if len(paired_just) > 0:
            arr_just = np.asarray(paired_just, dtype=float)
            arr_nojust = np.asarray(paired_nojust, dtype=float)

            # Wilcoxon can fail if all paired differences are zero
            diffs = arr_just - arr_nojust
            if np.any(diffs != 0):
                statistic, p_value = wilcoxon(arr_just, arr_nojust)
                stars = _p_to_stars(p_value)
            else:
                statistic, p_value, stars = 0.0, 1.0, "ns"

        wilcoxon_results[question] = {
            "question_label": question_label,
            "n_pairs": len(paired_just),
            "statistic": float(statistic) if statistic is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "significance": stars,
            "mean_justification": float(np.mean(paired_just)) if paired_just else None,
            "mean_no_justification": float(np.mean(paired_nojust)) if paired_nojust else None,
        }

        x = np.arange(len(likert_scale))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, counts_just, width=width, label="Justification")
        plt.bar(x + width / 2, counts_nojust, width=width, label="No Justification")

        plt.title(question_label)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.xticks(x, likert_scale)
        plt.legend()
        plt.grid(axis="y")

        # Add significance annotation
        y_max = max(max(counts_just, default=0), max(counts_nojust, default=0))
        y_range = max(y_max, 1)

        bracket_y = y_max + 0.10 * y_range
        bracket_h = 0.05 * y_range
        text_y = bracket_y + 0.03 * y_range

        plt.plot(
            [x[0] - width / 2, x[0] - width / 2, x[-1] + width / 2, x[-1] + width / 2],
            [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
            lw=1.5
        )

        if p_value is not None:
            annotation = f"{stars}\n(p = {p_value:.3g})"
        else:
            annotation = "ns"

        plt.text(
            (x[0] + x[-1]) / 2,
            text_y,
            annotation,
            ha="center",
            va="bottom"
        )

        plt.ylim(top=text_y + 0.10 * y_range)
        plt.tight_layout()
        plt.savefig(output_path / f"{question}_grouped.png", dpi=300, bbox_inches="tight")
        plt.close()

    return wilcoxon_results

def paired_plot(just, nojust, title, ylabel):
    x = [0, 1]

    plt.figure()

    # Boxplot
    plt.boxplot([nojust, just], positions=x)

    # Paired lines
    for i in range(len(just)):
        plt.plot(x, [nojust[i], just[i]], marker='o', alpha=0.5)

    plt.xticks(x, ["No Justification", "Justification"])
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(f"{title.replace(' ', '_').lower()}_paired.png", dpi=300, bbox_inches="tight")
    plt.close()

    medians = [np.median(d) for d in [just, nojust]]
    iqr = [np.percentile(d, 75) - np.percentile(d, 25) for d in [just, nojust]]
    return medians, iqr

def mean_error_plot(just, nojust, title, ylabel, use_sem=False):
    data = [nojust, just]
    labels = ["No Justification", "Justification"]

    means = [np.mean(d) for d in data]
    stds = [np.std(d, ddof=1) for d in data]

    if use_sem:
        errors = [s / np.sqrt(len(d)) for s, d in zip(stds, data)]
    else:
        errors = stds

    x = np.arange(len(labels))

    plt.figure()

    # Error bars
    plt.errorbar(x, means, yerr=errors, fmt='o', capsize=5)

    # Optional: connect means (nice touch)
    plt.plot(x, means, linestyle='--', alpha=0.6)

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(f"{title.replace(' ', '_').lower()}_mean_error.png", dpi=300, bbox_inches="tight")
    plt.close()
    return means, errors

def wilcoxon_effect_size(x, y):
    x = np.array(x)
    y = np.array(y)

    diff = x - y
    diff = diff[diff != 0]
    N = len(diff)

    #stat, p = wilcoxon(x, y)

    s = ttest_ind(x, y, equal_var=False)
    stat, p = s.statistic, s.pvalue

    mean_W = N * (N + 1) / 4
    std_W = np.sqrt(N * (N + 1) * (2*N + 1) / 24)

    z = (stat - mean_W) / std_W
    r = z / np.sqrt(N)

    return stat, p, z, r

def histograma(x, title, xlabel):
    plt.figure()
    plt.hist(x, bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

def compliance_when_correct_analysis(participants: list[dict]) -> dict[str, Any]:
    compliance_rates_justification = []
    compliance_rates_no_justification = []
    for participant in participants:
        just_compliance = participant.get("justification", {}).get("compliance_rate_when_correct")
        nojust_compliance = participant.get("no_justification", {}).get("compliance_rate_when_correct")

        if just_compliance is not None and nojust_compliance is not None:
            compliance_rates_justification.append(just_compliance)
            compliance_rates_no_justification.append(nojust_compliance)

    stat, p, z, r = wilcoxon_effect_size(compliance_rates_justification, compliance_rates_no_justification)
    shap_s, shap_p = shapiro(compliance_rates_justification)
    shap_s2, shap_p2 = shapiro(compliance_rates_no_justification)
    medians, iqr =paired_plot(compliance_rates_justification, compliance_rates_no_justification, f"Compliance Rate When Correct (p = {p:.3g}, r = {r:.3g})", "Compliance Rate")
    mean_error_plot(compliance_rates_justification, compliance_rates_no_justification, f"Mean Compliance Rate When Correct (p = {p:.3g}, r = {r:.3g})", "Compliance Rate")
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_compliance_when_correct_analysis(participants: list[dict]) -> dict[str, Any]:
    easy_compliance_rates_justification = []
    easy_compliance_rates_no_justification = []
    for participant in participants:
        just_compliance = participant.get("justification", {}).get("easy_compliance_rate_when_correct")
        nojust_compliance = participant.get("no_justification", {}).get("easy_compliance_rate_when_correct")

        if just_compliance is not None and nojust_compliance is not None:
            easy_compliance_rates_justification.append(just_compliance)
            easy_compliance_rates_no_justification.append(nojust_compliance)

    stat, p, z, r = wilcoxon_effect_size(easy_compliance_rates_justification, easy_compliance_rates_no_justification)
    medians, iqr = paired_plot(easy_compliance_rates_justification, easy_compliance_rates_no_justification, f"Compliance Rate When Correct For Easy Items (p = {p:.3g}, r = {r:.3g})", "Compliance Rate")
    mean_error_plot(easy_compliance_rates_justification, easy_compliance_rates_no_justification, f"Mean Easy Compliance Rate When Correct (p = {p:.3g}, r = {r:.3g})", "Compliance Rate")
    shap_s, shap_p = shapiro(easy_compliance_rates_justification)
    shap_s2, shap_p2 = shapiro(easy_compliance_rates_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_compliance_when_correct_analysis(participants: list[dict]) -> dict[str, Any]:
    hard_compliance_rates_justification = []
    hard_compliance_rates_no_justification = []
    for participant in participants:
        just_compliance = participant.get("justification", {}).get("hard_compliance_rate_when_correct")
        nojust_compliance = participant.get("no_justification", {}).get("hard_compliance_rate_when_correct")

        if just_compliance is not None and nojust_compliance is not None:
            hard_compliance_rates_justification.append(just_compliance)
            hard_compliance_rates_no_justification.append(nojust_compliance)

    stat, p, z, r = wilcoxon_effect_size(hard_compliance_rates_justification, hard_compliance_rates_no_justification)
    medians, iqr = paired_plot(hard_compliance_rates_justification, hard_compliance_rates_no_justification, f"Compliance Rate When Correct for Hard Items (p = {p:.3g}, r = {r:.3g})", "Compliance Rate")
    mean_error_plot(hard_compliance_rates_justification, hard_compliance_rates_no_justification, f"Mean Hard Compliance Rate When Correct (p = {p:.3g}, r = {r:.3g})", "Hard Compliance Rate")
    shap_s, shap_p = shapiro(hard_compliance_rates_justification)
    shap_s2, shap_p2 = shapiro(hard_compliance_rates_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def override_when_incorrect_analysis(participants: list[dict]) -> dict[str, Any]:
    override_rates_justification = []
    override_rates_no_justification = []
    for participant in participants:
        just_override = participant.get("justification", {}).get("override_rate_when_incorrect")
        nojust_override = participant.get("no_justification", {}).get("override_rate_when_incorrect")

        if just_override is not None and nojust_override is not None:
            override_rates_justification.append(just_override)
            override_rates_no_justification.append(nojust_override)

    stat, p, z, r = wilcoxon_effect_size(override_rates_justification, override_rates_no_justification)
    medians, iqr = paired_plot(override_rates_justification, override_rates_no_justification, f"Override Rate When Incorrect (p = {p:.3g}, r = {r:.3g})", "Override Rate")
    mean_error_plot(override_rates_justification, override_rates_no_justification, f"Mean Override Rate When Incorrect (p = {p:.3g}, r = {r:.3g})", "Override Rate")
    shap_s, shap_p = shapiro(override_rates_justification)
    shap_s2, shap_p2 = shapiro(override_rates_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_override_when_incorrect_analysis(participants: list[dict]) -> dict[str, Any]:
    easy_override_rates_justification = []
    easy_override_rates_no_justification = []
    for participant in participants:
        just_override = participant.get("justification", {}).get("easy_override_rate_when_incorrect")
        nojust_override = participant.get("no_justification", {}).get("easy_override_rate_when_incorrect")

        if just_override is not None and nojust_override is not None:
            easy_override_rates_justification.append(just_override)
            easy_override_rates_no_justification.append(nojust_override)

    stat, p, z, r = wilcoxon_effect_size(easy_override_rates_justification, easy_override_rates_no_justification)
    medians, iqr = paired_plot(easy_override_rates_justification, easy_override_rates_no_justification, f"Override Rate When Incorrect for Easy Items(p = {p:.3g}, r = {r:.3g})", "Override Rate")
    mean_error_plot(easy_override_rates_justification, easy_override_rates_no_justification, f"Mean Easy Override Rate When Incorrect (p = {p:.3g}, r = {r:.3g})", "Easy Override Rate")
    shap_s, shap_p = shapiro(easy_override_rates_justification)
    shap_s2, shap_p2 = shapiro(easy_override_rates_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_override_when_incorrect_analysis(participants: list[dict]) -> dict[str, Any]:
    hard_override_rates_justification = []
    hard_override_rates_no_justification = []
    for participant in participants:
        just_override = participant.get("justification", {}).get("hard_override_rate_when_incorrect")
        nojust_override = participant.get("no_justification", {}).get("hard_override_rate_when_incorrect")

        if just_override is not None and nojust_override is not None:
            hard_override_rates_justification.append(just_override)
            hard_override_rates_no_justification.append(nojust_override)

    stat, p, z, r = wilcoxon_effect_size(hard_override_rates_justification, hard_override_rates_no_justification)   
    medians, iqr = paired_plot(hard_override_rates_justification, hard_override_rates_no_justification, f"Hard Override Rate When Incorrect (p = {p:.3g}, r = {r:.3g})", "Hard Override Rate")
    mean_error_plot(hard_override_rates_justification, hard_override_rates_no_justification, f"Mean Hard Override Rate When Incorrect (p = {p:.3g}, r = {r:.3g})", "Hard Override Rate")
    shap_s, shap_p = shapiro(hard_override_rates_justification)
    shap_s2, shap_p2 = shapiro(hard_override_rates_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def trust_decay_analysis(participants: list[dict]) -> dict[str, Any]:
    trust_decay_justification = []
    trust_decay_no_justification = []
    for participant in participants:
        just_decay = participant.get("justification", {}).get("compliance_decay_rate")
        nojust_decay = participant.get("no_justification", {}).get("compliance_decay_rate")

        if just_decay is not None and nojust_decay is not None:
            trust_decay_justification.append(just_decay)
            trust_decay_no_justification.append(nojust_decay)
            

    stat, p, z, r = wilcoxon_effect_size(trust_decay_justification, trust_decay_no_justification)
    medians, iqr = paired_plot(trust_decay_justification, trust_decay_no_justification, f"Trust Decay Rate (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    mean, stds = mean_error_plot(trust_decay_justification, trust_decay_no_justification, f"Mean Trust Decay Rate (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    shap_s, shap_p = shapiro(trust_decay_justification)
    shap_s2, shap_p2 = shapiro(trust_decay_no_justification)
    histograma(trust_decay_justification, f"Trust Decay Rate with Justification (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    histograma(trust_decay_no_justification, f"Trust Decay Rate without Justification (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2), "justifi_size": len(trust_decay_justification), "no_justifi_size": len(trust_decay_no_justification), "means": mean, "stds": stds}

def easy_trust_decay_analysis(participants: list[dict]) -> dict[str, Any]:
    trust_decay_justification = []
    trust_decay_no_justification = []
    for participant in participants:
        just_decay = participant.get("justification", {}).get("easy_compliance_decay_rate")
        nojust_decay = participant.get("no_justification", {}).get("easy_compliance_decay_rate")

        if just_decay is not None and nojust_decay is not None:
            trust_decay_justification.append(just_decay)
            trust_decay_no_justification.append(nojust_decay)

    stat, p, z, r = wilcoxon_effect_size(trust_decay_justification, trust_decay_no_justification)
    medians, iqr = paired_plot(trust_decay_justification, trust_decay_no_justification, f"Trust Decay Rate for Easy Items (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    mean, stds =mean_error_plot(trust_decay_justification, trust_decay_no_justification, f"Mean Easy Trust Decay Rate (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    shap_s, shap_p = shapiro(trust_decay_justification)
    shap_s2, shap_p2 = shapiro(trust_decay_no_justification)
    histograma(trust_decay_justification, f"Easy Trust Decay Rate with Justification (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    histograma(trust_decay_no_justification, f"Easy Trust Decay Rate without Justification (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2), "justifi_size": len(trust_decay_justification), "no_justifi_size": len(trust_decay_no_justification), "means": mean, "stds": stds}

def hard_trust_decay_analysis(participants: list[dict]) -> dict[str, Any]:
    trust_decay_justification = []
    trust_decay_no_justification = []
    for participant in participants:
        just_decay = participant.get("justification", {}).get("hard_compliance_decay_rate")
        nojust_decay = participant.get("no_justification", {}).get("hard_compliance_decay_rate")

        if just_decay is not None and nojust_decay is not None:
            trust_decay_justification.append(just_decay)
            trust_decay_no_justification.append(nojust_decay)

    stat, p, z, r = wilcoxon_effect_size(trust_decay_justification, trust_decay_no_justification)
    medians, iqr = paired_plot(trust_decay_justification, trust_decay_no_justification, f"Hard Trust Decay Rate (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    mean, stds = mean_error_plot(trust_decay_justification, trust_decay_no_justification, f"Mean Hard Trust Decay Rate (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    shap_s, shap_p = shapiro(trust_decay_justification)
    shap_s2, shap_p2 = shapiro(trust_decay_no_justification)
    histograma(trust_decay_justification, f"Hard Trust Decay Rate with Justification (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    histograma(trust_decay_no_justification, f"Hard Trust Decay Rate without Justification (p = {p:.3g}, r = {r:.3g})", "Trust Decay Rate")
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2), "justifi_size": len(trust_decay_justification), "no_justifi_size": len(trust_decay_no_justification), "means": mean, "stds": stds}

def memory_test_analysis(participants: list[dict]) -> dict[str, Any]:
    memory_test_justification = []
    memory_test_no_justification = []
    for participant in participants:
        just_memory = participant.get("justification", {}).get("memory_test_accuracy")
        nojust_memory = participant.get("no_justification", {}).get("memory_test_accuracy")

        if just_memory is not None and nojust_memory is not None:
            memory_test_justification.append(just_memory)
            memory_test_no_justification.append(nojust_memory)

    stat, p, z, r = wilcoxon_effect_size(memory_test_justification, memory_test_no_justification)
    medians, iqr = paired_plot(memory_test_justification, memory_test_no_justification, f"Memory Test Accuracy (p = {p:.3g}, r = {r:.3g})", "Memory Test Accuracy")
    means, errors = mean_error_plot(memory_test_justification, memory_test_no_justification, f"Mean Memory Test Accuracy (p = {p:.3g}, r = {r:.3g})", "Memory Test Accuracy")
    shap_s, shap_p = shapiro(memory_test_justification)
    shap_s2, shap_p2 = shapiro(memory_test_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_memory_test_analysis(participants: list[dict]) -> dict[str, Any]:
    easy_memory_test_justification = []
    easy_memory_test_no_justification = []
    for participant in participants:
        just_memory = participant.get("justification", {}).get("easy_memory_test_accuracy")
        nojust_memory = participant.get("no_justification", {}).get("easy_memory_test_accuracy")

        if just_memory is not None and nojust_memory is not None:
            easy_memory_test_justification.append(just_memory)
            easy_memory_test_no_justification.append(nojust_memory)

    stat, p, z, r = wilcoxon_effect_size(easy_memory_test_justification, easy_memory_test_no_justification)
    medians, iqr = paired_plot(easy_memory_test_justification, easy_memory_test_no_justification, f"Easy Memory Test Accuracy (p = {p:.3g}, r = {r:.3g})", "Easy Memory Test Accuracy")
    means, errors = mean_error_plot(easy_memory_test_justification, easy_memory_test_no_justification, f"Mean Easy Memory Test Accuracy (p = {p:.3g}, r = {r:.3g})", "Easy Memory Test Accuracy")
    shap_s, shap_p = shapiro(easy_memory_test_justification)
    shap_s2, shap_p2 = shapiro(easy_memory_test_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_memory_test_analysis(participants: list[dict]) -> dict[str, Any]:
    hard_memory_test_justification = []
    hard_memory_test_no_justification = []
    for participant in participants:
        just_memory = participant.get("justification", {}).get("hard_memory_test_accuracy")
        nojust_memory = participant.get("no_justification", {}).get("hard_memory_test_accuracy")

        if just_memory is not None and nojust_memory is not None:
            hard_memory_test_justification.append(just_memory)
            hard_memory_test_no_justification.append(nojust_memory)

    stat, p, z, r = wilcoxon_effect_size(hard_memory_test_justification, hard_memory_test_no_justification)
    paired_plot(hard_memory_test_justification, hard_memory_test_no_justification, f"Hard Memory Test Accuracy (p = {p:.3g}, r = {r:.3g})", "Hard Memory Test Accuracy")
    means, stds = mean_error_plot(hard_memory_test_justification, hard_memory_test_no_justification, f"Mean Hard Memory Test Accuracy (p = {p:.3g}, r = {r:.3g})", "Hard Memory Test Accuracy")
    shap_s, shap_p = shapiro(hard_memory_test_justification)
    shap_s2, shap_p2 = shapiro(hard_memory_test_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def time_speaker_analysis(participants: list[dict]) -> dict[str, Any]:
    import statistics 
    times_speaker_justification = []
    times_speaker_no_justification = []
    for participant in participants:
        just_times = participant.get("justification", {}).get("times_speaker")
        nojust_times = participant.get("no_justification", {}).get("times_speaker")
    
        if just_times is not None and nojust_times is not None:
            times_speaker_justification.append(statistics.median(just_times))
            times_speaker_no_justification.append(statistics.median(nojust_times))

    stat, p, z, r = wilcoxon_effect_size(times_speaker_justification, times_speaker_no_justification)
    medians, iqr = paired_plot(times_speaker_justification, times_speaker_no_justification, f"Median Time to Respond to Speaker (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    mean_error_plot(times_speaker_justification, times_speaker_no_justification, f"Mean Time to Respond to Speaker (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    shap_s, shap_p = shapiro(times_speaker_justification)
    shap_s2, shap_p2 = shapiro(times_speaker_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_time_speaker_analysis(participants: list[dict]) -> dict[str, Any]:
    import statistics 
    easy_times_speaker_justification = []
    easy_times_speaker_no_justification = []
    for participant in participants:
        just_times = participant.get("justification", {}).get("easy_times_speaker")
        nojust_times = participant.get("no_justification", {}).get("easy_times_speaker")
    
        if just_times is not None and nojust_times is not None:
            easy_times_speaker_justification.append(statistics.median(just_times))
            easy_times_speaker_no_justification.append(statistics.median(nojust_times))

    stat, p, z, r = wilcoxon_effect_size(easy_times_speaker_justification, easy_times_speaker_no_justification)
    medians, iqr = paired_plot(easy_times_speaker_justification, easy_times_speaker_no_justification, f"Median Time to Respond to Speaker (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    mean_error_plot(easy_times_speaker_justification, easy_times_speaker_no_justification, f"Mean Time to Respond to Speaker (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    shap_s, shap_p = shapiro(easy_times_speaker_justification)
    shap_s2, shap_p2 = shapiro(easy_times_speaker_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_time_speaker_analysis(participants: list[dict]) -> dict[str, Any]:
    import statistics 
    hard_times_speaker_justification = []
    hard_times_speaker_no_justification = []
    for participant in participants:
        just_times = participant.get("justification", {}).get("hard_times_speaker")
        nojust_times = participant.get("no_justification", {}).get("hard_times_speaker")
    
        if just_times is not None and nojust_times is not None:
            hard_times_speaker_justification.append(statistics.median(just_times))
            hard_times_speaker_no_justification.append(statistics.median(nojust_times))

    stat, p, z, r = wilcoxon_effect_size(hard_times_speaker_justification, hard_times_speaker_no_justification)
    medians, iqr = paired_plot(hard_times_speaker_justification, hard_times_speaker_no_justification, f"Median Time to Respond to Speaker (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    mean_error_plot(hard_times_speaker_justification, hard_times_speaker_no_justification, f"Mean Time to Respond to Speaker (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    shap_s, shap_p = shapiro(hard_times_speaker_justification)
    shap_s2, shap_p2 = shapiro(hard_times_speaker_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def time_arm_analysis(participants: list[dict]) -> dict[str, Any]:
    import statistics 
    times_arm_justification = []
    times_arm_no_justification = []
    for participant in participants:
        just_times = participant.get("justification", {}).get("times_arm")
        nojust_times = participant.get("no_justification", {}).get("times_arm")
    
        if just_times is not None and nojust_times is not None:
            times_arm_justification.append(statistics.median(just_times))
            times_arm_no_justification.append(statistics.median(nojust_times))

    stat, p, z, r = wilcoxon_effect_size(times_arm_justification, times_arm_no_justification)
    medians, iqr = paired_plot(times_arm_justification, times_arm_no_justification, f"Median Time to Respond to Arm (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    mean_error_plot(times_arm_justification, times_arm_no_justification, f"Mean Time to Respond to Arm (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    shap_s, shap_p = shapiro(times_arm_justification)
    shap_s2, shap_p2 = shapiro(times_arm_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_time_arm_analysis(participants: list[dict]) -> dict[str, Any]:
    import statistics 
    easy_times_arm_justification = []
    easy_times_arm_no_justification = []
    for participant in participants:
        just_times = participant.get("justification", {}).get("easy_times_arm")
        nojust_times = participant.get("no_justification", {}).get("easy_times_arm")
    
        if just_times is not None and nojust_times is not None:
            easy_times_arm_justification.append(statistics.median(just_times))
            easy_times_arm_no_justification.append(statistics.median(nojust_times))

    stat, p, z, r = wilcoxon_effect_size(easy_times_arm_justification, easy_times_arm_no_justification)
    medians, iqr = paired_plot(easy_times_arm_justification, easy_times_arm_no_justification, f"Median Time to Respond to Arm (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    mean_error_plot(easy_times_arm_justification, easy_times_arm_no_justification, f"Mean Time to Respond to Arm (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    shap_s, shap_p = shapiro(easy_times_arm_justification)
    shap_s2, shap_p2 = shapiro(easy_times_arm_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_time_arm_analysis(participants: list[dict]) -> dict[str, Any]:
    import statistics 
    hard_times_arm_justification = []
    hard_times_arm_no_justification = []
    for participant in participants:
        just_times = participant.get("justification", {}).get("hard_times_arm")
        nojust_times = participant.get("no_justification", {}).get("hard_times_arm")
    
        if just_times is not None and nojust_times is not None:
            hard_times_arm_justification.append(statistics.median(just_times))
            hard_times_arm_no_justification.append(statistics.median(nojust_times))

    stat, p, z, r = wilcoxon_effect_size(hard_times_arm_justification, hard_times_arm_no_justification)
    medians, iqr = paired_plot(hard_times_arm_justification, hard_times_arm_no_justification, f"Median Time to Respond to Arm (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    mean_error_plot(hard_times_arm_justification, hard_times_arm_no_justification, f"Mean Time to Respond to Arm (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Time (s)")
    shap_s, shap_p = shapiro(hard_times_arm_justification)
    shap_s2, shap_p2 = shapiro(hard_times_arm_no_justification)
    return {"median": medians, "iqr": iqr, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def analyze_quetionnaire_responses(participants: list[dict[str, Any]]) -> dict[str, Any]:
    robot_motion_questions = ["the_way_the_robot_moved_made_me_uncomfortable", "the_speed_at_which_the_gripper_picked_up_and_released_the_components_made_me_uneasy"]
    safe_cooperation_questions = ["i_trusted_that_the_robot_was_safe_to_cooperate_with", "i_was_comfortable_the_robot_would_not_hurt_me", "the_size_of_the_robot_did_not_intimidate_me", "i_felt_safe_interacting_with_the_robot"]
    robot_reliability_questions = ["i_knew_the_gripper_would_not_drop_the_components", "the_robot_gripper_did_not_look_reliable", "the_gripper_seemed_like_it_could_be_trusted", "i_felt_i_could_rely_on_the_robot_to_do_what_it_was_supposed_to_do"]
    inverted_score_questions = ["the_way_the_robot_moved_made_me_uncomfortable", "the_speed_at_which_the_gripper_picked_up_and_released_the_components_made_me_uneasy", "the_robot_gripper_did_not_look_reliable"]
    for participant in participants:
        participant["justification_questionnaire_analysis"] = {
            "robot_motion_discomfort": np.sum([participant.get("justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("justification_questionnaire", {}).get(q) for q in robot_motion_questions if participant.get("justification_questionnaire", {}).get(q) is not None]),
            "safe_cooperation_trust": np.sum([participant.get("justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("justification_questionnaire", {}).get(q) for q in safe_cooperation_questions if participant.get("justification_questionnaire", {}).get(q) is not None]),
            "robot_reliability_trust": np.sum([participant.get("justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("justification_questionnaire", {}).get(q) for q in robot_reliability_questions if participant.get("justification_questionnaire", {}).get(q) is not None]),
            "total_trust": np.sum([participant.get("justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("justification_questionnaire", {}).get(q) for q in robot_motion_questions + safe_cooperation_questions + robot_reliability_questions if participant.get("justification_questionnaire", {}).get(q) is not None])
        }
        participant["no_justification_questionnaire_analysis"] = {
            "robot_motion_discomfort": np.sum([participant.get("no_justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("no_justification_questionnaire", {}).get(q) for q in robot_motion_questions if participant.get("no_justification_questionnaire", {}).get(q) is not None]),
            "safe_cooperation_trust": np.sum([participant.get("no_justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("no_justification_questionnaire", {}).get(q) for q in safe_cooperation_questions if participant.get("no_justification_questionnaire", {}).get(q) is not None]),
            "robot_reliability_trust": np.sum([participant.get("no_justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("no_justification_questionnaire", {}).get(q) for q in robot_reliability_questions if participant.get("no_justification_questionnaire", {}).get(q) is not None]),
            "total_trust": np.sum([participant.get("no_justification_questionnaire", {}).get(q) if q not in inverted_score_questions else 5 - participant.get("no_justification_questionnaire", {}).get(q) for q in robot_motion_questions + safe_cooperation_questions + robot_reliability_questions if participant.get("no_justification_questionnaire", {}).get(q) is not None])
        }

    return participants

def robot_motion_question_analysis(participants: list[dict]) -> dict[str, Any]:
    robot_motion_justification = []
    robot_motion_no_justification = []
    for participant in participants:
        just_motion = participant.get("justification_questionnaire_analysis", {}).get("robot_motion_discomfort")
        nojust_motion = participant.get("no_justification_questionnaire_analysis", {}).get("robot_motion_discomfort")

        if just_motion is not None and nojust_motion is not None:
            robot_motion_justification.append(just_motion)
            robot_motion_no_justification.append(nojust_motion)

    stat, p, z, r = wilcoxon_effect_size(robot_motion_justification, robot_motion_no_justification)
    paired_plot(robot_motion_justification, robot_motion_no_justification, f"Robot Motion Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(robot_motion_justification, robot_motion_no_justification, f"Mean Robot Motion Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(robot_motion_justification)
    shap_s2, shap_p2 = shapiro(robot_motion_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def robot_safe_cooperation_analysis(participants: list[dict]) -> dict[str, Any]:
    safe_cooperation_justification = []
    safe_cooperation_no_justification = []
    for participant in participants:
        just_safe = participant.get("justification_questionnaire_analysis", {}).get("safe_cooperation_trust")
        nojust_safe = participant.get("no_justification_questionnaire_analysis", {}).get("safe_cooperation_trust")

        if just_safe is not None and nojust_safe is not None:
            safe_cooperation_justification.append(just_safe)
            safe_cooperation_no_justification.append(nojust_safe)

    stat, p, z, r = wilcoxon_effect_size(safe_cooperation_justification, safe_cooperation_no_justification)
    paired_plot(safe_cooperation_justification, safe_cooperation_no_justification, f"Safe Cooperation Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(safe_cooperation_justification, safe_cooperation_no_justification, f"Mean Safe Cooperation Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(safe_cooperation_justification)
    shap_s2, shap_p2 = shapiro(safe_cooperation_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def robot_reliability_analysis(participants: list[dict]) -> dict[str, Any]:
    reliability_justification = []
    reliability_no_justification = []
    for participant in participants:
        just_reliability = participant.get("justification_questionnaire_analysis", {}).get("robot_reliability_trust")
        nojust_reliability = participant.get("no_justification_questionnaire_analysis", {}).get("robot_reliability_trust")

        if just_reliability is not None and nojust_reliability is not None:
            reliability_justification.append(just_reliability)
            reliability_no_justification.append(nojust_reliability)

    stat, p, z, r = wilcoxon_effect_size(reliability_justification, reliability_no_justification)
    paired_plot(reliability_justification, reliability_no_justification, f"Robot Reliability Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(reliability_justification, reliability_no_justification, f"Mean Robot Reliability Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(reliability_justification)
    shap_s2, shap_p2 = shapiro(reliability_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def total_trust_analysis(participants: list[dict]) -> dict[str, Any]:
    total_trust_justification = []
    total_trust_no_justification = []
    for participant in participants:
        just_total = participant.get("justification_questionnaire_analysis", {}).get("total_trust")
        nojust_total = participant.get("no_justification_questionnaire_analysis", {}).get("total_trust")

        if just_total is not None and nojust_total is not None:
            total_trust_justification.append(just_total)
            total_trust_no_justification.append(nojust_total)

    stat, p, z, r = wilcoxon_effect_size(total_trust_justification, total_trust_no_justification)
    paired_plot(total_trust_justification, total_trust_no_justification, f"Total Trust (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(total_trust_justification, total_trust_no_justification, f"Mean Total Trust (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(total_trust_justification)
    shap_s2, shap_p2 = shapiro(total_trust_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def rely_question_analysis(participants: list[dict]) -> dict[str, Any]:
    rely_justification = []
    rely_no_justification = []
    for participant in participants:
        just_rely = participant.get("justification_questionnaire", {}).get("i_felt_i_could_rely_on_the_robot_to_do_what_it_was_supposed_to_do")
        nojust_rely = participant.get("no_justification_questionnaire", {}).get("i_felt_i_could_rely_on_the_robot_to_do_what_it_was_supposed_to_do")

        if just_rely is not None and nojust_rely is not None:
            rely_justification.append(just_rely)
            rely_no_justification.append(nojust_rely)

    stat, p, z, r = wilcoxon_effect_size(rely_justification, rely_no_justification)
    paired_plot(rely_justification, rely_no_justification, f"Reliability Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(rely_justification, rely_no_justification, f"Mean Reliability Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(rely_justification)
    shap_s2, shap_p2 = shapiro(rely_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def gripper_drop_question_analysis(participants: list[dict]) -> dict[str, Any]:
    drop_justification = []
    drop_no_justification = []
    for participant in participants:
        just_drop = participant.get("justification_questionnaire", {}).get("i_knew_the_gripper_would_not_drop_the_components")
        nojust_drop = participant.get("no_justification_questionnaire", {}).get("i_knew_the_gripper_would_not_drop_the_components")

        if just_drop is not None and nojust_drop is not None:
            drop_justification.append(just_drop)
            drop_no_justification.append(nojust_drop)

    stat, p, z, r = wilcoxon_effect_size(drop_justification, drop_no_justification)
    paired_plot(drop_justification, drop_no_justification, f"Gripper Drop Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(drop_justification, drop_no_justification, f"Mean Gripper Drop Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(drop_justification)
    shap_s2, shap_p2 = shapiro(drop_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def gripper_trust_question_analysis(participants: list[dict]) -> dict[str, Any]:
    reliable_justification = []
    reliable_no_justification = []
    for participant in participants:
        just_reliable = participant.get("justification_questionnaire", {}).get("the_gripper_seemed_like_it_could_be_trusted")
        nojust_reliable = participant.get("no_justification_questionnaire", {}).get("the_gripper_seemed_like_it_could_be_trusted")

        if just_reliable is not None and nojust_reliable is not None:
            reliable_justification.append(just_reliable)
            reliable_no_justification.append(nojust_reliable)

    stat, p, z, r = wilcoxon_effect_size(reliable_justification, reliable_no_justification)
    paired_plot(reliable_justification, reliable_no_justification, f"Gripper Reliable Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(reliable_justification, reliable_no_justification, f"Mean Gripper Reliable Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(reliable_justification)
    shap_s2, shap_p2 = shapiro(reliable_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def gripper_not_rely_question_analysis(participants: list[dict]) -> dict[str, Any]:
    not_rely_justification = []
    not_rely_no_justification = []
    for participant in participants:
        just_not_rely = participant.get("justification_questionnaire", {}).get("the_robot_gripper_did_not_look_reliable")
        nojust_not_rely = participant.get("no_justification_questionnaire", {}).get("the_robot_gripper_did_not_look_reliable")

        if just_not_rely is not None and nojust_not_rely is not None:
            not_rely_justification.append(just_not_rely)
            not_rely_no_justification.append(nojust_not_rely)

    stat, p, z, r = wilcoxon_effect_size(not_rely_justification, not_rely_no_justification)
    paired_plot(not_rely_justification, not_rely_no_justification, f"Gripper Not Reliable Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    means, stds = mean_error_plot(not_rely_justification, not_rely_no_justification, f"Mean Gripper Not Reliable Question (p = {p:.3g}, r = {r:.3g})", "Rating")
    shap_s, shap_p = shapiro(not_rely_justification)
    shap_s2, shap_p2 = shapiro(not_rely_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def time_arm_zero_rate_analysis(participants: list[dict]) -> dict[str, Any]:
    time_arm_zero_justification = []
    time_arm_zero_no_justification = []
    for participant in participants:
        just_time_arm_zero = participant.get("justification", {}).get("time_arm_zeros_rate")
        nojust_time_arm_zero = participant.get("no_justification", {}).get("time_arm_zeros_rate")

        if just_time_arm_zero is not None and nojust_time_arm_zero is not None:
            time_arm_zero_justification.append(just_time_arm_zero)
            time_arm_zero_no_justification.append(nojust_time_arm_zero)

    stat, p, z, r = wilcoxon_effect_size(time_arm_zero_justification, time_arm_zero_no_justification)
    paired_plot(time_arm_zero_justification, time_arm_zero_no_justification, f"Time Arm Zero Rate (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    means, stds = mean_error_plot(time_arm_zero_justification, time_arm_zero_no_justification, f"Mean Time Arm Zero Rate (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    shap_s, shap_p = shapiro(time_arm_zero_justification)
    shap_s2, shap_p2 = shapiro(time_arm_zero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_time_arm_zero_rate_analysis(participants: list[dict]) -> dict[str, Any]:
    time_arm_zero_justification = []
    time_arm_zero_no_justification = []
    for participant in participants:
        just_time_arm_zero = participant.get("justification", {}).get("easy_time_arm_zeros_rate")
        nojust_time_arm_zero = participant.get("no_justification", {}).get("easy_time_arm_zeros_rate")

        if just_time_arm_zero is not None and nojust_time_arm_zero is not None:
            time_arm_zero_justification.append(just_time_arm_zero)
            time_arm_zero_no_justification.append(nojust_time_arm_zero)

    stat, p, z, r = wilcoxon_effect_size(time_arm_zero_justification, time_arm_zero_no_justification)
    paired_plot(time_arm_zero_justification, time_arm_zero_no_justification, f"Time Arm Zero Rate (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    means, stds = mean_error_plot(time_arm_zero_justification, time_arm_zero_no_justification, f"Mean Time Arm Zero Rate (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    shap_s, shap_p = shapiro(time_arm_zero_justification)
    shap_s2, shap_p2 = shapiro(time_arm_zero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_time_arm_zero_rate_analysis(participants: list[dict]) -> dict[str, Any]:
    time_arm_zero_justification = []
    time_arm_zero_no_justification = []
    for participant in participants:
        just_time_arm_zero = participant.get("justification", {}).get("hard_time_arm_zeros_rate")
        nojust_time_arm_zero = participant.get("no_justification", {}).get("hard_time_arm_zeros_rate")

        if just_time_arm_zero is not None and nojust_time_arm_zero is not None:
            time_arm_zero_justification.append(just_time_arm_zero)
            time_arm_zero_no_justification.append(nojust_time_arm_zero)

    stat, p, z, r = wilcoxon_effect_size(time_arm_zero_justification, time_arm_zero_no_justification)
    paired_plot(time_arm_zero_justification, time_arm_zero_no_justification, f"Time Arm Zero Rate (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    means, stds = mean_error_plot(time_arm_zero_justification, time_arm_zero_no_justification, f"Mean Time Arm Zero Rate (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    shap_s, shap_p = shapiro(time_arm_zero_justification)
    shap_s2, shap_p2 = shapiro(time_arm_zero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def time_speaker_zero_rate_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_justification = []
    time_speaker_zero_no_justification = []
    for participant in participants:
        just_time_speaker_zero = participant.get("justification", {}).get("time_speaker_zeros_rate")
        nojust_time_speaker_zero = participant.get("no_justification", {}).get("time_speaker_zeros_rate")

        if just_time_speaker_zero is not None and nojust_time_speaker_zero is not None:
            time_speaker_zero_justification.append(just_time_speaker_zero)
            time_speaker_zero_no_justification.append(nojust_time_speaker_zero)

    stat, p, z, r = wilcoxon_effect_size(time_speaker_zero_justification, time_speaker_zero_no_justification)
    paired_plot(time_speaker_zero_justification, time_speaker_zero_no_justification, f"Time Speaker Zero Rate (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    means, stds = mean_error_plot(time_speaker_zero_justification, time_speaker_zero_no_justification, f"Mean Time Speaker Zero Rate (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    shap_s, shap_p = shapiro(time_speaker_zero_justification)
    shap_s2, shap_p2 = shapiro(time_speaker_zero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_time_speaker_zero_rate_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_justification = []
    time_speaker_zero_no_justification = []
    for participant in participants:
        just_time_speaker_zero = participant.get("justification", {}).get("easy_time_speaker_zeros_rate")
        nojust_time_speaker_zero = participant.get("no_justification", {}).get("easy_time_speaker_zeros_rate")

        if just_time_speaker_zero is not None and nojust_time_speaker_zero is not None:
            time_speaker_zero_justification.append(just_time_speaker_zero)
            time_speaker_zero_no_justification.append(nojust_time_speaker_zero)

    stat, p, z, r = wilcoxon_effect_size(time_speaker_zero_justification, time_speaker_zero_no_justification)
    paired_plot(time_speaker_zero_justification, time_speaker_zero_no_justification, f"Time Speaker Zero Rate (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    means, stds = mean_error_plot(time_speaker_zero_justification, time_speaker_zero_no_justification, f"Mean Time Speaker Zero Rate (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    shap_s, shap_p = shapiro(time_speaker_zero_justification)
    shap_s2, shap_p2 = shapiro(time_speaker_zero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_time_speaker_zero_rate_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_justification = []
    time_speaker_zero_no_justification = []
    for participant in participants:
        just_time_speaker_zero = participant.get("justification", {}).get("hard_time_speaker_zeros_rate")
        nojust_time_speaker_zero = participant.get("no_justification", {}).get("hard_time_speaker_zeros_rate")

        if just_time_speaker_zero is not None and nojust_time_speaker_zero is not None:
            time_speaker_zero_justification.append(just_time_speaker_zero)
            time_speaker_zero_no_justification.append(nojust_time_speaker_zero)

    stat, p, z, r = wilcoxon_effect_size(time_speaker_zero_justification, time_speaker_zero_no_justification)
    paired_plot(time_speaker_zero_justification, time_speaker_zero_no_justification, f"Time Speaker Zero Rate (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    means, stds = mean_error_plot(time_speaker_zero_justification, time_speaker_zero_no_justification, f"Mean Time Speaker Zero Rate (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Zero Rate")
    shap_s, shap_p = shapiro(time_speaker_zero_justification)
    shap_s2, shap_p2 = shapiro(time_speaker_zero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def time_speaker_zero_arm_nonzero_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_arm_nonzero_justification = []
    time_speaker_zero_arm_nonzero_no_justification = []
    for participant in participants:
        just_time_speaker_zero_arm_nonzero = participant.get("justification", {}).get("time_speaker_zeros_arm_nonzeros_rate")
        nojust_time_speaker_zero_arm_nonzero = participant.get("no_justification", {}).get("time_speaker_zeros_arm_nonzeros_rate")

        if just_time_speaker_zero_arm_nonzero is not None and nojust_time_speaker_zero_arm_nonzero is not None:
            time_speaker_zero_arm_nonzero_justification.append(just_time_speaker_zero_arm_nonzero)
            time_speaker_zero_arm_nonzero_no_justification.append(nojust_time_speaker_zero_arm_nonzero)

    stat, p, z, r = wilcoxon_effect_size(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification)
    paired_plot(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification, f"Time Speaker Zero and Arm Non-Zero Rate (p = {p:.3g}, r = {r:.3g})", "Rate")
    means, stds = mean_error_plot(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification, f"Mean Time Speaker Zero and Arm Non-Zero Rate (p = {p:.3g}, r = {r:.3g})", "Rate")
    shap_s, shap_p = shapiro(time_speaker_zero_arm_nonzero_justification)
    shap_s2, shap_p2 = shapiro(time_speaker_zero_arm_nonzero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def easy_time_speaker_zero_arm_nonzero_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_arm_nonzero_justification = []
    time_speaker_zero_arm_nonzero_no_justification = []
    for participant in participants:
        just_time_speaker_zero_arm_nonzero = participant.get("justification", {}).get("easy_time_speaker_zeros_arm_nonzeros_rate")
        nojust_time_speaker_zero_arm_nonzero = participant.get("no_justification", {}).get("easy_time_speaker_zeros_arm_nonzeros_rate")

        if just_time_speaker_zero_arm_nonzero is not None and nojust_time_speaker_zero_arm_nonzero is not None:
            time_speaker_zero_arm_nonzero_justification.append(just_time_speaker_zero_arm_nonzero)
            time_speaker_zero_arm_nonzero_no_justification.append(nojust_time_speaker_zero_arm_nonzero)

    stat, p, z, r = wilcoxon_effect_size(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification)
    paired_plot(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification, f"Time Speaker Zero and Arm Non-Zero Rate (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Rate")
    means, stds = mean_error_plot(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification, f"Mean Time Speaker Zero and Arm Non-Zero Rate (Easy Trials) (p = {p:.3g}, r = {r:.3g})", "Rate")
    shap_s, shap_p = shapiro(time_speaker_zero_arm_nonzero_justification)
    shap_s2, shap_p2 = shapiro(time_speaker_zero_arm_nonzero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def hard_time_speaker_zero_arm_nonzero_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_arm_nonzero_justification = []
    time_speaker_zero_arm_nonzero_no_justification = []
    for participant in participants:
        just_time_speaker_zero_arm_nonzero = participant.get("justification", {}).get("hard_time_speaker_zeros_arm_nonzeros_rate")
        nojust_time_speaker_zero_arm_nonzero = participant.get("no_justification", {}).get("hard_time_speaker_zeros_arm_nonzeros_rate")

        if just_time_speaker_zero_arm_nonzero is not None and nojust_time_speaker_zero_arm_nonzero is not None:
            time_speaker_zero_arm_nonzero_justification.append(just_time_speaker_zero_arm_nonzero)
            time_speaker_zero_arm_nonzero_no_justification.append(nojust_time_speaker_zero_arm_nonzero)

    stat, p, z, r = wilcoxon_effect_size(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification)
    paired_plot(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification, f"Time Speaker Zero and Arm Non-Zero Rate (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Rate")
    means, stds = mean_error_plot(time_speaker_zero_arm_nonzero_justification, time_speaker_zero_arm_nonzero_no_justification, f"Mean Time Speaker Zero and Arm Non-Zero Rate (Hard Trials) (p = {p:.3g}, r = {r:.3g})", "Rate")
    shap_s, shap_p = shapiro(time_speaker_zero_arm_nonzero_justification)
    shap_s2, shap_p2 = shapiro(time_speaker_zero_arm_nonzero_no_justification)
    return {"means": means, "stds": stds, "p": p, "r": r, "shapiro_justification": (shap_s, shap_p), "shapiro_no_justification": (shap_s2, shap_p2)}

def time_speaker_zero_arm_nonzero_ratio_analysis(participants: list[dict]) -> dict[str, Any]:
    time_speaker_zero_arm_nonzero_ratio_justification = []
    time_speaker_zero_arm_nonzero_ratio_no_justification = []
    for participant in participants:
        just_time_speaker_zero_arm_nonzero_ratio = participant.get("justification", {}).get("time_speaker_zeros_arm_nonzeros_ratio")

        if just_time_speaker_zero_arm_nonzero_ratio is not None:
            time_speaker_zero_arm_nonzero_ratio_justification.append(just_time_speaker_zero_arm_nonzero_ratio)

    mean, std = np.mean(time_speaker_zero_arm_nonzero_ratio_justification), np.std(time_speaker_zero_arm_nonzero_ratio_justification)
    return time_speaker_zero_arm_nonzero_ratio_justification, mean, std
   

def age_and_gender_analysis(participants: list[dict]) -> dict[str, Any]:
    ages = []
    genders = []
    for participant in participants:
        age = participant.get("age")
        gender = participant.get("gender")
        if age is not None:
            ages.append(age)
        if gender is not None:
            genders.append(gender)

    age_mean = np.mean(ages) if ages else None
    age_std = np.std(ages) if ages else None
    gender_counts = [0, 0]
    for gender in genders:
        if gender == "Male":
            gender_counts[0] += 1
        elif gender == "Female":
            gender_counts[1] += 1
    return {"ages": {"mean": age_mean, "std": age_std}, "genders": {"counts": gender_counts}}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment metrics and an HTML report from participant JSON.")
    parser.add_argument("--input", required=True, help="Path to the participants JSON file")
    parser.add_argument("--output-html", default="experiment_report.html", help="Path to the output HTML report")
    parser.add_argument("--output-json", default="experiment_metrics.json", help="Path to the output metrics JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    participants = load_dataset(input_path)
    print(f"Loaded participants from {input_path}")

    wilcoxon_results = {}

    report = analyze_dataset(participants)
    report = compute_overall_metrics(report)
    report = compute_overall_questionnaire_metrics(participants, report)
    participants = analyze_quetionnaire_responses(participants)
    # plot_time_boxplots(report["overall_metrics"])
    # plot_decay_boxplots(report["overall_metrics"])
    # wilcoxon_results["compliance_when_correct"] = compliance_when_correct_analysis(report["participants"])
    # wilcoxon_results["easy_compliance_when_correct"] = easy_compliance_when_correct_analysis(report["participants"])
    # wilcoxon_results["hard_compliance_when_correct"] = hard_compliance_when_correct_analysis(report["participants"])
    # wilcoxon_results["override_when_incorrect"] = override_when_incorrect_analysis(report["participants"])
    # wilcoxon_results["easy_override_when_incorrect"] = easy_override_when_incorrect_analysis(report["participants"])
    # wilcoxon_results["hard_override_when_incorrect"] = hard_override_when_incorrect_analysis(report["participants"])
    wilcoxon_results["trust_decay"] = trust_decay_analysis(report["participants"])
    wilcoxon_results["easy_trust_decay"] = easy_trust_decay_analysis(report["participants"])
    wilcoxon_results["hard_trust_decay"] = hard_trust_decay_analysis(report["participants"])
    # wilcoxon_results["memory_test"] = memory_test_analysis(report["participants"])
    # wilcoxon_results["easy_memory_test"] = easy_memory_test_analysis(report["participants"])
    # wilcoxon_results["hard_memory_test"] = hard_memory_test_analysis(report["participants"])
    # wilcoxon_results["time_speaker"] = time_speaker_analysis(report["participants"])
    # wilcoxon_results["easy_time_speaker"] = easy_time_speaker_analysis(report["participants"])
    # wilcoxon_results["hard_time_speaker"] = hard_time_speaker_analysis(report["participants"])
    # wilcoxon_results["time_arm"] = time_arm_analysis(report["participants"])
    # wilcoxon_results["easy_time_arm"] = easy_time_arm_analysis(report["participants"])
    # wilcoxon_results["hard_time_arm"] = hard_time_arm_analysis(report["participants"])
    # wilcoxon_results["robot_motion"] = robot_motion_question_analysis(participants)
    # wilcoxon_results["robot_safe_cooperation"] = robot_safe_cooperation_analysis(participants)
    wilcoxon_results["robot_reliability"] = robot_reliability_analysis(participants)
    # wilcoxon_results["total_trust"] = total_trust_analysis(participants)
    # wilcoxon_results["rely"] = rely_question_analysis(participants)
    # wilcoxon_results["gripper_drop"] = gripper_drop_question_analysis(participants)
    # wilcoxon_results["gripper_trust"] = gripper_trust_question_analysis(participants)
    # wilcoxon_results["gripper_not_rely"] = gripper_not_rely_question_analysis(participants)
    # wilcoxon_results["time_arm_zero"] = time_arm_zero_rate_analysis(report["participants"])
    # wilcoxon_results["easy_time_arm_zero"] = easy_time_arm_zero_rate_analysis(report["participants"])
    # wilcoxon_results["hard_time_arm_zero"] = hard_time_arm_zero_rate_analysis(report["participants"])
    # wilcoxon_results["time_speaker_zero"] = time_speaker_zero_rate_analysis(report["participants"])
    # wilcoxon_results["easy_time_speaker_zero"] = easy_time_speaker_zero_rate_analysis(report["participants"])
    # wilcoxon_results["hard_time_speaker_zero"] = hard_time_speaker_zero_rate_analysis(report["participants"])
    # wilcoxon_results["time_speaker_zero_arm_nonzero"] = time_speaker_zero_arm_nonzero_analysis(report["participants"])
    # wilcoxon_results["easy_time_speaker_zero_arm_nonzero"] = easy_time_speaker_zero_arm_nonzero_analysis(report["participants"])
    # wilcoxon_results["hard_time_speaker_zero_arm_nonzero"] = hard_time_speaker_zero_arm_nonzero_analysis(report["participants"])
    # wilcoxon_results["time_speaker_zero_arm_nonzero_ratio"] = time_speaker_zero_arm_nonzero_ratio_analysis(report["participants"])
    # wilcoxon_results["age_and_gender"] = age_and_gender_analysis(participants)

    question_results = save_questionnaire_grouped_barplots(participants, report)

    for q, res in question_results.items():
        print(q, res)

    print(wilcoxon_results)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)

    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    #output_html.write_text(build_html(report), encoding="utf-8")

    #print(f"Loaded {report['n_participants']} participants from {input_path}")
    print(f"Metrics JSON written to {output_json}")
    #print(f"HTML report written to {output_html}")


if __name__ == "__main__":
    main()