import cv2
import numpy as np


def analyze_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
    }


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    result = analyze_image(image_path)

    if result:
        print("\nImage Quality Measurements")
        print("-------------------------")
        print(f"Brightness : {result['brightness']}")
        print(f"Contrast   : {result['contrast']}")
        print(f"Sharpness  : {result['sharpness']}")
    else:
        print("Could not load image.")