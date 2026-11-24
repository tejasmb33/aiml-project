
import cv2


def detect_faces(image):
    """
    Detect faces in an image using OpenCV Haar Cascade.
    Returns the annotated image and detected face locations.
    """

    if image is None:
        raise ValueError("Invalid image. Unable to load image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Unable to load face detection model.")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    output = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return output, faces


def analyze_image(image_path):
    """
    Load an image, detect faces and display detection information.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output, faces = detect_faces(image)

    print(f"Faces detected: {len(faces)}")

    for i, (x, y, w, h) in enumerate(faces, start=1):
        print(f"Face {i}: x={x}, y={y}, width={w}, height={h}")

    cv2.imshow("OpenCV Face Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    analyze_image(image_path)
