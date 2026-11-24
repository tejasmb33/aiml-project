import subprocess
import sys

scripts = [
    "src/batch_image_quality.py",
    "src/batch_face_analysis.py",
    "src/evaluate_face_detection.py",
    "src/performance_evaluation.py",
    "src/robustness_evaluation.py",
    "src/plot_image_quality.py",
    "src/plot_detection_evaluation.py",
    "src/plot_robustness_results.py",
    "src/combined_results_plot.py"
]

for script in scripts:

    print("\n" + "=" * 50)
    print(f"Running: {script}")
    print("=" * 50)

    result = subprocess.run(
        [sys.executable, script],
        check=False
    )

    if result.returncode != 0:
        print(f"Failed: {script}")
        sys.exit(result.returncode)

print("\nPROJECT PIPELINE COMPLETED.")