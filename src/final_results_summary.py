import pandas as pd
import os

FILES = {
    "Face Analysis": "results/tables/face_analysis_results.csv",
    "Performance": "results/tables/performance_results.csv",
    "Robustness": "results/tables/robustness_results.csv",
    "Detection Evaluation": "results/tables/face_detection_evaluation.csv"
}


def summarize():

    print("\nPROJECT EXPERIMENT SUMMARY")
    print("=" * 50)

    for name, path in FILES.items():

        print(f"\n{name}")

        if not os.path.exists(path):
            print("File not found:", path)
            continue

        df = pd.read_csv(path)

        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print(df.head().to_string(index=False))

    print("\nSummary inspection completed.")


if __name__ == "__main__":
    summarize()