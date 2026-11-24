import os

FILES = [
    "src/face_detection.py",
    "src/preprocessing.py",
    "src/analysis_pipeline.py",
    "src/batch_image_quality.py",
    "src/batch_face_analysis.py",
    "src/evaluate_face_detection.py",
    "src/live_face_detection.py",
    "src/performance_evaluation.py",
    "src/robustness_evaluation.py",
    "src/plot_image_quality.py",
    "src/plot_detection_evaluation.py",
    "src/plot_robustness_results.py",
    "src/combined_results_plot.py",
    "src/run_project.py",
    "src/final_results_summary.py",
    "models/face_detection_yunet.onnx",
]

OUTPUTS = [
    "results/tables/face_analysis_results.csv",
    "results/tables/performance_results.csv",
    "results/tables/robustness_results.csv",
    "results/tables/face_detection_evaluation.csv",
    "results/graphs/face_detection_output_rate.png",
    "results/graphs/robustness_comparison.png",
    "results/graphs/combined_comparison.png",
]


def audit():

    print("\nPROJECT FILE AUDIT")
    print("=" * 45)

    missing = []

    for path in FILES + OUTPUTS:

        status = os.path.exists(path)

        print(
            f"[{'OK' if status else 'MISSING'}] {path}"
        )

        if not status:
            missing.append(path)

    print("\n" + "=" * 45)

    if missing:
        print(f"{len(missing)} item(s) missing.")
    else:
        print("All listed items are present.")

    print("Audit completed.")


if __name__ == "__main__":
    audit()