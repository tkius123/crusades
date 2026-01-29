#!/usr/bin/env python3
"""Pre-download dataset for offline evaluation.

This script runs during Docker build to cache dataset samples.
The cached data is shuffled at runtime using validator-provided seed.
"""

import json
from pathlib import Path

from datasets import load_dataset

# Configuration
DATASET_NAME = "HuggingFaceFW/fineweb"
NUM_SAMPLES = 50000  # Cache 50k samples (more than needed for shuffling pool)
CACHE_PATH = Path("/home/appuser/.cache/templar")
OUTPUT_FILE = CACHE_PATH / "dataset.json"


def main():
    print(f"Downloading {NUM_SAMPLES} samples from {DATASET_NAME}...")

    # Create cache directory
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    # Load dataset with streaming
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Collect text samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        text = sample.get("text", sample.get("content", ""))
        if text and len(text) > 100:  # Skip very short samples
            samples.append(text)
        if (i + 1) % 10000 == 0:
            print(f"  Downloaded {i + 1} samples...")

    print(f"Collected {len(samples)} valid samples")

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(samples, f)

    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved to {OUTPUT_FILE} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
