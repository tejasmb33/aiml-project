
import csv
import os
from collections import defaultdict

INPUT_FILE = "results/tables/face_analysis_results.csv"
OUTPUT_FILE = "results/tables/face_detection_evaluation.csv"


def evaluate():

    if not os.path.exists(INPUT_FILE):
        print("Face analysis CSV not found.")
        return

    groups = defaultdict(list)

    with open(INPUT_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            condition = row["condition"]
            face_count = int(row["face_count"])

            groups[condition].append(face_count)

    summary = []

    for condition, counts in groups.items():

        total_images = len(counts)
        images_with_faces = sum(1 for count in counts if count > 0)
        total_faces = sum(counts)

        detection_rate = (
            images_with_faces / total_images * 100
            if total_images else 0
        )

        average_faces = (
            total_faces / total_images
            if total_images else 0
        )

        summary.append({
            "condition": condition,
            "total_images": total_images,
            "images_with_faces": images_with_faces,
            "images_without_faces": total_images - images_with_faces,
            "detection_rate_percent": round(detection_rate, 2),
            "average_faces_detected": round(average_faces, 2)
        })

    os.makedirs("results/tables", exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nFace Detection Evaluation")
    print("-" * 60)

    for row in summary:
        print(f"\nCondition: {row['condition']}")
        print(f"Images: {row['total_images']}")
        print(f"Images with detected faces: {row['images_with_faces']}")
        print(f"Images without detected faces: {row['images_without_faces']}")
        print(f"Detection rate: {row['detection_rate_percent']}%")
        print(f"Average faces detected: {row['average_faces_detected']}")

    print(f"\nEvaluation saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()
