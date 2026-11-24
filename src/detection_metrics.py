"""
Non-identifying face detection evaluation.
"""

import json
from pathlib import Path


def summarize_results(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print("Results file not found.")
        return

    print(f"Results file: {csv_path}")
    print("Evaluation summary will be generated from recorded observations.")


def main():
    results_file = "evaluation_data/evaluation_results.csv"
    summarize_results(results_file)


if __name__ == "__main__":
    main()