"""
play_wav.py
-----------
Prompts for an item name and a condition, then plays the matching WAV file.

Folder structure (as produced by generate_tts.py):
    tts_output/
        no_justification/    ->  {item_name}.wav
        justification_true/  ->  {item_name}.wav
        justification_false/ ->  {item_name}.wav

Usage:
    python play_wav.py
    python play_wav.py --dir /path/to/tts_output

Requirements:
    ffmpeg must be installed: sudo apt install ffmpeg  /  brew install ffmpeg
"""

import argparse
import os
import subprocess
import sys

OUTPUT_DIR = "tts_output"
CONDITIONS = ["no_justification", "justification_true", "justification_false"]


def list_available(base_dir: str) -> None:
    """Print all WAV files grouped by condition folder."""
    print()
    for condition in CONDITIONS:
        folder = os.path.join(base_dir, condition)
        if not os.path.isdir(folder):
            print(f"  [{condition}]  (folder not found)")
            continue
        files = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".wav"))
        print(f"  [{condition}]")
        for name in files:
            print(f"    • {name}")
    print()


def find_wav(base_dir: str, item_name: str, condition: str) -> str | None:
    """Return the WAV path if it exists, else None."""
    path = os.path.join(base_dir, condition, f"{item_name}.wav")
    return path if os.path.isfile(path) else None


def play_wav(path: str) -> None:
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=OUTPUT_DIR,
                        help=f"Base tts_output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()
    base_dir = args.dir

    print(f"WAV Player — reads from '{base_dir}/'")
    print("Enter an item name to play. Type 'list' to browse, 'quit' to exit.\n")

    while True:
        try:
            item_name = input("▶  item name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

        if not item_name:
            continue
        if item_name.lower() in ("quit", "exit", "q"):
            print("Bye!")
            sys.exit(0)
        if item_name.lower() == "list":
            list_available(base_dir)
            continue

        # Case-insensitive name resolution against actual files on disk
        matched_name = item_name
        sample_folder = os.path.join(base_dir, CONDITIONS[0])
        if os.path.isdir(sample_folder):
            files = {f[:-4].lower(): f[:-4] for f in os.listdir(sample_folder) if f.endswith(".wav")}
            if item_name.lower() in files:
                matched_name = files[item_name.lower()]

        # Find which conditions have this item
        available = [c for c in CONDITIONS if find_wav(base_dir, matched_name, c)]

        if not available:
            print(f"  [!] '{matched_name}.wav' not found in any condition folder. Type 'list' to browse.\n")
            continue

        # If only one condition exists, play it directly
        if len(available) == 1:
            condition = available[0]
            path = find_wav(base_dir, matched_name, condition)
            print(f"  ▶ [{condition}] {matched_name}.wav")
            play_wav(path)

        else:
            print(f"  Found in {len(available)} conditions:")
            for i, c in enumerate(available, 1):
                print(f"    {i}. {c}")
            try:
                choice = input("  Pick (number, or Enter for all): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                sys.exit(0)

            if choice == "":
                for condition in available:
                    path = find_wav(base_dir, matched_name, condition)
                    print(f"  ▶ [{condition}] {matched_name}.wav")
                    play_wav(path)
            elif choice.isdigit() and 1 <= int(choice) <= len(available):
                condition = available[int(choice) - 1]
                path = find_wav(base_dir, matched_name, condition)
                print(f"  ▶ [{condition}] {matched_name}.wav")
                play_wav(path)
            else:
                print("  Invalid choice, skipping.")

        print()


if __name__ == "__main__":
    main()
