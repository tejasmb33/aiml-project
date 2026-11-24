
import cv2


def preprocess_image(image):
    """
    Prepare an image for computer vision analysis.
    Returns resized image, grayscale image and normalized image.
    """

    if image is None:
        raise ValueError("Invalid image.")

    height, width = image.shape[:2]

    # Resize while preserving aspect ratio
    max_dimension = 800

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(image, (new_width, new_height))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize contrast
    normalized = cv2.equalizeHist(gray)

    return image, gray, normalized
