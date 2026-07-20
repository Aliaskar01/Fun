from __future__ import annotations
from pathlib import Path
import cv2
import mediapipe as mp

ASSET_DIR = Path(__file__).with_name("assets")

DEFAULT_IMAGES = {
    "shock": ASSET_DIR / "C:\\Users\\alias\\Downloads\\cat-shock.jpeg",
    "tongue": ASSET_DIR / "C:\\Users\\alias\\Downloads\\cat-tongue.jpeg",
    "glare": ASSET_DIR / "C:\\Users\\alias\\Downloads\\cat-glare.jpeg",
    "neutral": ASSET_DIR / "C:\\Users\\alias\\Downloads\\larry.jpeg",
}

EYE_OPENING_THRESHOLD = 0.025
MOUTH_OPEN_THRESHOLD = 0.03
SQUINTING_THRESHOLD = 0.018


def eye_opening(face_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> float:
    left_top = face_landmarks.landmark[159]
    left_bottom = face_landmarks.landmark[145]
    right_top = face_landmarks.landmark[386]
    right_bottom = face_landmarks.landmark[374]
    return (abs(left_top.y - left_bottom.y) + abs(right_top.y - right_bottom.y)) / 2.0


def mouth_opening(face_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> float:
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    return abs(top_lip.y - bottom_lip.y)


def detect_emotion(face_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> str:
    if mouth_opening(face_landmarks) > MOUTH_OPEN_THRESHOLD:
        return "tongue"
    if eye_opening(face_landmarks) > EYE_OPENING_THRESHOLD:
        return "shock"
    if eye_opening(face_landmarks) < SQUINTING_THRESHOLD:
        return "glare"
    return "neutral"


def resolve_image(label: str) -> Path:
    return DEFAULT_IMAGES.get(label, DEFAULT_IMAGES["neutral"])


def draw_landmarks(image, face_landmarks) -> None:
    height, width = image.shape[:2]
    for lm in face_landmarks.landmark:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(image, (x, y), 1, (0, 100, 0), -1)


def show_cat_image(window_name: str, image_path: Path, fallback_frame) -> None:
    cat = cv2.imread(str(image_path))
    if cat is not None:
        cat = cv2.resize(cat, (640, 480))
        cv2.imshow(window_name, cat)
        return

    blank = fallback_frame * 0
    cv2.putText(
        blank,
        f"Missing: {image_path}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.imshow(window_name, blank)


def main() -> None:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Camera not available. Check camera index or permissions.")
 
    with mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1,
    ) as face_mesh:
        while True:
            ret, image = cam.read()
            if not ret:
                break

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = face_mesh.process(rgb_image)
            face_landmark_points = processed_image.multi_face_landmarks

            label = "neutral"
            if face_landmark_points:
                face_landmarks = face_landmark_points[0]
                label = detect_emotion(face_landmarks)
                draw_landmarks(image, face_landmarks)

            cv2.imshow("Face Detection", image)
            show_cat_image("Cat Image", resolve_image(label), image)

            key = cv2.waitKey(1)
            if key == 27:
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()