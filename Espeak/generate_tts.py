"""
generate_tts.py
---------------
Generates human-like TTS .wav files for each justification entry
using Microsoft Edge TTS (neural voices via edge-tts library).

Usage:
    python generate_tts.py

Output structure:
    tts_output/
        no_justification/    ->  <item_name>.wav   (x20)
        justification_true/  ->  <item_name>.wav   (x20)
        justification_false/ ->  <item_name>.wav   (x20)

Requirements:
    pip install edge-tts
    ffmpeg must be installed (sudo apt install ffmpeg / brew install ffmpeg)
"""

import asyncio
import os
import tempfile

import edge_tts

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

NO_JUSTIFICATION = [
    # Recycle - Easy
    {"item_name": "Plastic_Bottle",        "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable."},
    {"item_name": "Metal_Can",             "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable."},
    {"item_name": "Cardboard_Box",         "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable."},
    {"item_name": "Paper_Sheet",           "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable."},
    {"item_name": "Glass_Bottle",          "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable."},
    # Recycle - Hard
    {"item_name": "Tetra_Pak",             "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable."},
    {"item_name": "Plastic_Bag",           "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable."},
    {"item_name": "Blister_Pack",          "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable."},
    {"item_name": "Aluminium_Foil",        "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable."},
    {"item_name": "Bubble_Wrap",           "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable."},
    # Waste - Easy
    {"item_name": "Paper_Towel",           "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable."},
    {"item_name": "Used_Tissue",           "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable."},
    {"item_name": "Surgical_Mask",         "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable."},
    {"item_name": "Food_Waste",            "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable."},
    {"item_name": "Broken_Ceramic",        "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable."},
    # Waste - Hard
    {"item_name": "Black_Plastic",         "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable."},
    {"item_name": "Plasticized_Paper_Cup", "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable."},
    {"item_name": "Waxed_Cardboard",       "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable."},
    {"item_name": "Foam",                  "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable."},
    {"item_name": "Wooden_Packaging",      "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable."},
]

JUSTIFICATION_TRUE = [
    # Recycle - Easy
    {"item_name": "Plastic_Bottle",        "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable because it is made of rigid PET plastic."},
    {"item_name": "Metal_Can",             "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable because metal is fully recovered in sorting facilities."},
    {"item_name": "Cardboard_Box",         "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable because cardboard fibers are easily reprocessed."},
    {"item_name": "Paper_Sheet",           "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable because clean paper is one of the most recycled materials."},
    {"item_name": "Glass_Bottle",          "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is recyclable because glass can be melted and reformed indefinitely."},
    # Recycle - Hard
    {"item_name": "Tetra_Pak",             "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable because dedicated facilities can separate its layers."},
    {"item_name": "Plastic_Bag",           "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable because soft plastics are accepted in the yellow bin in Paris."},
    {"item_name": "Blister_Pack",          "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable because mixed plastic packaging is accepted in the sorting bin."},
    {"item_name": "Aluminium_Foil",        "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable because aluminium is fully recoverable if not heavily soiled."},
    {"item_name": "Bubble_Wrap",           "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is recyclable because plastic protective packaging is accepted in the yellow bin."},
    # Waste - Easy
    {"item_name": "Paper_Towel",           "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable because used paper towels are contaminated and too degraded."},
    {"item_name": "Used_Tissue",           "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable because hygiene waste cannot be processed in recycling streams."},
    {"item_name": "Surgical_Mask",         "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable because it combines multiple materials and poses hygiene risks."},
    {"item_name": "Food_Waste",            "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable because organic matter contaminates the recycling stream."},
    {"item_name": "Broken_Ceramic",        "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is not recyclable because ceramics cannot be processed with glass or plastics."},
    # Waste - Hard
    {"item_name": "Black_Plastic",         "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable because black pigment blocks optical sorting sensors."},
    {"item_name": "Plasticized_Paper_Cup", "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable because the plastic lining cannot be separated from the paper."},
    {"item_name": "Waxed_Cardboard",       "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable because the wax coating prevents fiber recovery."},
    {"item_name": "Foam",                  "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable because expanded foam is not accepted in standard sorting centers."},
    {"item_name": "Wooden_Packaging",      "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is not recyclable because wood is not part of household packaging recycling."},
]

JUSTIFICATION_FALSE = [
    # Recycle - Easy (false: claimed NOT recyclable)
    {"item_name": "Plastic_Bottle",        "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is not recyclable because plastic bottles are too thick to be processed."},
    {"item_name": "Metal_Can",             "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is not recyclable because residual food inside contaminates the batch."},
    {"item_name": "Cardboard_Box",         "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is not recyclable because cardboard with any printing cannot be reprocessed."},
    {"item_name": "Paper_Sheet",           "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is not recyclable because handled paper absorbs too many impurities."},
    {"item_name": "Glass_Bottle",          "ground_truth": "recycle", "difficulty": "easy", "justification": "This item is not recyclable because colored glass cannot be mixed in recycling bins."},
    # Recycle - Hard (false: claimed NOT recyclable)
    {"item_name": "Tetra_Pak",             "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is not recyclable because its layers of plastic, aluminum and cardboard cannot be separated."},
    {"item_name": "Plastic_Bag",           "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is not recyclable because thin plastic films jam sorting machines."},
    {"item_name": "Blister_Pack",          "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is not recyclable because the mix of plastic and foil makes it non-sortable."},
    {"item_name": "Aluminium_Foil",        "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is not recyclable because small foil pieces are too light to be captured by sorters."},
    {"item_name": "Bubble_Wrap",           "ground_truth": "recycle", "difficulty": "hard", "justification": "This item is not recyclable because soft plastics are rejected by standard recycling centers."},
    # Waste - Easy (false: claimed IS recyclable)
    {"item_name": "Paper_Towel",           "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is recyclable because it is made of paper, just like cardboard."},
    {"item_name": "Used_Tissue",           "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is recyclable because tissues are a paper product and belong in the paper bin."},
    {"item_name": "Surgical_Mask",         "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is recyclable because its main layer is a non-woven plastic fiber."},
    {"item_name": "Food_Waste",            "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is recyclable because organic matter can be composted alongside recyclables."},
    {"item_name": "Broken_Ceramic",        "ground_truth": "waste",   "difficulty": "easy", "justification": "This item is recyclable because ceramics are mineral-based like glass and can be melted down."},
    # Waste - Hard (false: claimed IS recyclable)
    {"item_name": "Black_Plastic",         "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is recyclable because all plastic packaging regardless of color goes in the yellow bin."},
    {"item_name": "Plasticized_Paper_Cup", "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is recyclable because it is mostly cardboard and can go with paper waste."},
    {"item_name": "Waxed_Cardboard",       "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is recyclable because cardboard is always accepted in the recycling bin."},
    {"item_name": "Foam",                  "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is recyclable because foam is a type of plastic and belongs in the plastic bin."},
    {"item_name": "Wooden_Packaging",      "ground_truth": "waste",   "difficulty": "hard", "justification": "This item is recyclable because wood can be processed like cardboard in recycling streams."},
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOICE = "en-US-JennyNeural"
RATE  = "-5%"
OUTPUT_DIR = "tts_output"

DATASETS = {
    "no_justification":    NO_JUSTIFICATION,
    "justification_true":  JUSTIFICATION_TRUE,
    "justification_false": JUSTIFICATION_FALSE,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def synthesize_to_mp3(text: str, mp3_path: str) -> None:
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(mp3_path)


def mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    ret = os.system(f'ffmpeg -y -loglevel error -i "{mp3_path}" "{wav_path}"')
    if ret != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed for {mp3_path}. "
            "Make sure ffmpeg is installed: sudo apt install ffmpeg"
        )


async def generate_wav(folder: str, item_name: str, text: str) -> None:
    wav_path = os.path.join(folder, f"{item_name}.wav")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name
    try:
        await synthesize_to_mp3(text, mp3_path)
        mp3_to_wav(mp3_path, wav_path)
        print(f"  v  {wav_path}")
    finally:
        os.unlink(mp3_path)


async def process_dataset(condition: str, dataset: list[dict]) -> None:
    folder = os.path.join(OUTPUT_DIR, condition)
    os.makedirs(folder, exist_ok=True)
    print(f"\n[{condition.upper()}] Generating {len(dataset)} files ...")
    tasks = [
        generate_wav(folder, entry["item_name"], entry["justification"])
        for entry in dataset
    ]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    for condition, dataset in DATASETS.items():
        await process_dataset(condition, dataset)
    print(f"\nDone! All 60 WAV files saved under ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
