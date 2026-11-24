import os
import pandas as pd

FILES = {
    "face_analysis": "results/tables/face_analysis_results.csv",
    "performance": "results/tables/performance_results.csv",
    "robustness": "results/tables/robustness_results.csv",
    "detection_evaluation": "results/tables/face_detection_evaluation.csv",
}

OUTPUT = "results/tables/project_results_index.csv"


def consolidate():

    os.makedirs("results/tables", exist_ok=True)

    records = []

    for name, path in FILES.items():

        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        df = pd.read_csv(path)

        records.append({
            "dataset": name,
            "file": path,
            "rows": len(df),
            "columns": len(df.columns),
            "status": "Available"
        })

    result = pd.DataFrame(records)

    result.to_csv(OUTPUT, index=False)

    print(result.to_string(index=False))
    print("\nConsolidation completed.")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    consolidate()