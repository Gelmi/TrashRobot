
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


ITEM_COL_PATTERN = re.compile(r"^\s*\[Item\s+(\d+)\](?:\.1)?\s*$", re.IGNORECASE)


def snake_case(text: str) -> str:
    text = str(text).strip()
    text = text.replace("&", "and")
    text = re.sub(r"\.+1$", "", text)
    text = re.sub(r"[\[\]\(\)]", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        return v
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
    return value


def is_memory_item_column(col: str) -> bool:
    return bool(ITEM_COL_PATTERN.match(str(col)))


def split_questionnaire_columns(columns: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    first_block = []
    second_block = []
    first_memory = []
    second_memory = []

    for col in columns:
        if col in {"Carimbo de data/hora", "Participant ID", "Age", "First Scenario"}:
            continue

        is_second = str(col).endswith(".1")
        if is_memory_item_column(col):
            if is_second:
                second_memory.append(col)
            else:
                first_memory.append(col)
        else:
            if is_second:
                second_block.append(col)
            else:
                first_block.append(col)

    first_memory.sort(key=lambda c: int(ITEM_COL_PATTERN.match(str(c)).group(1)))
    second_memory.sort(key=lambda c: int(ITEM_COL_PATTERN.match(str(c)).group(1)))
    return first_block, second_block, first_memory, second_memory


def build_questionnaire_block(row: pd.Series, columns: List[str]) -> Dict[str, Any]:
    return {snake_case(col): clean_value(row[col]) for col in columns}


def build_memory_sequence(row: pd.Series, columns: List[str]) -> List[Any]:
    return [clean_value(row[col]) for col in columns]


def load_questionnaires(questionnaire_csv: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(questionnaire_csv)
    first_block_cols, second_block_cols, first_memory_cols, second_memory_cols = split_questionnaire_columns(list(df.columns))

    questionnaire_by_participant: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        participant_id = str(clean_value(row["Participant ID"]))
        first_scenario = clean_value(row.get("First Scenario"))

        block_a = build_questionnaire_block(row, first_block_cols)
        block_b = build_questionnaire_block(row, second_block_cols)
        memory_a = build_memory_sequence(row, first_memory_cols)
        memory_b = build_memory_sequence(row, second_memory_cols)

        if first_scenario == "justification":
            justification_q = block_a
            no_justification_q = block_b
            justification_memory = memory_a
            no_justification_memory = memory_b
        else:
            no_justification_q = block_a
            justification_q = block_b
            no_justification_memory = memory_a
            justification_memory = memory_b

        questionnaire_by_participant[participant_id] = {
            "participant_id": participant_id,
            "age": clean_value(row.get("Age")),
            "timestamp": clean_value(row.get("Carimbo de data/hora")),
            "first_scenario": first_scenario,
            "justification_questionnaire": justification_q,
            "no_justification_questionnaire": no_justification_q,
            "_memory_map": {
                "justification": justification_memory,
                "no_justification": no_justification_memory,
            },
        }

    return questionnaire_by_participant


def parse_metadata_line(first_line: str) -> Dict[str, Any]:
    first_line = first_line.strip().lstrip("#").strip()
    parts = [p.strip() for p in first_line.split(",") if p.strip()]
    metadata = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            metadata[snake_case(key)] = value.strip()
    return metadata


def load_participant_trials(participant_csv: Path) -> Dict[str, Any]:
    with participant_csv.open("r", encoding="utf-8") as f:
        first_line = f.readline()

    metadata = parse_metadata_line(first_line)

    df = pd.read_csv(participant_csv, skiprows=1)
    df.columns = [snake_case(c) for c in df.columns]

    trials = [{col: clean_value(row[col]) for col in df.columns} for _, row in df.iterrows()]
    justification_trials = [t for t in trials if t.get("classification") == "justification"]
    no_justification_trials = [t for t in trials if t.get("classification") == "no_justification"]

    return {
        "participant_id": str(metadata.get("id")),
        "name": metadata.get("name"),
        "gender": metadata.get("gender"),
        "justification": justification_trials,
        "no_justification": no_justification_trials,
    }


def apply_memory_tests(participant_data: Dict[str, Any], questionnaire_data: Dict[str, Any]) -> None:
    memory_map = questionnaire_data.get("_memory_map", {})

    for scenario in ("justification", "no_justification"):
        trials = participant_data.get(scenario, [])
        memory_values = memory_map.get(scenario, [])

        for idx, trial in enumerate(trials):
            trial["memory_test"] = memory_values[idx] if idx < len(memory_values) else None


def merge_data(questionnaires: Dict[str, Dict[str, Any]], participant_files: List[Path]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for participant_file in sorted(participant_files):
        participant_data = load_participant_trials(participant_file)
        participant_id = participant_data["participant_id"]
        questionnaire_data = questionnaires.get(participant_id, {})

        if questionnaire_data:
            apply_memory_tests(participant_data, questionnaire_data)

        merged = {**participant_data, **questionnaire_data}
        merged["participant_id"] = participant_id
        merged.pop("_memory_map", None)
        results.append(merged)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert experiment questionnaires + participant CSVs into JSON."
    )
    parser.add_argument("--questionnaire", required=True, help="Path to the master questionnaire CSV.")
    parser.add_argument("--participants-dir", required=True, help="Directory containing the participant CSV files.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--glob", default="*.csv", help="File pattern for participant CSVs inside --participants-dir.")

    args = parser.parse_args()

    questionnaire_csv = Path(args.questionnaire)
    participants_dir = Path(args.participants_dir)
    output_json = Path(args.output)

    participant_files = [p for p in participants_dir.glob(args.glob) if p.name != questionnaire_csv.name]

    questionnaires = load_questionnaires(questionnaire_csv)
    data = merge_data(questionnaires, participant_files)

    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(data)} participants to: {output_json}")


if __name__ == "__main__":
    main()
