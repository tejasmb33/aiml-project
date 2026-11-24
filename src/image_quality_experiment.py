import cv2
import os
import csv

def analyze_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "width": image.shape[1],
        "height": image.shape[0]
    }


def analyze_folder(folder_path, output_csv):
    results = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        if not os.path.isfile(image_path):
            continue

        result = analyze_image(image_path)

        if result is not None:
            result["filename"] = filename
            results.append(result)

    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "brightness", "contrast", "width", "height"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAnalyzed {len(results)} images.")
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    folder_path = input("Enter image folder path: ")
    output_csv = "image_quality_results.csv"

    analyze_folder(folder_path, output_csv)