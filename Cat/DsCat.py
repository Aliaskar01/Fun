from __future__ import annotations
import cv2
import mediapipe as mp
import pyvirtualcam
from pathlib import Path

DEFAULT_IMAGES = {
    "shock": Path(r"C:\Users\alias\Downloads\cat-shock.jpeg"),
    "tongue": Path(r"C:\Users\alias\Downloads\cat-tongue.jpeg"),
    "glare": Path(r"C:\Users\alias\Downloads\cat-glare.jpeg"),
    "neutral": Path(r"C:\Users\alias\Downloads\larry.jpeg"),
}

EYE_OPENING_THRESHOLD = 0.025
MOUTH_OPEN_THRESHOLD = 0.03
SQUINTING_THRESHOLD = 0.018


def eye_opening(face_landmarks):
    left_top = face_landmarks.landmark[159]
    left_bottom = face_landmarks.landmark[145]
    right_top = face_landmarks.landmark[386]
    right_bottom = face_landmarks.landmark[374]
    return (abs(left_top.y - left_bottom.y) +
            abs(right_top.y - right_bottom.y)) / 2.0


def mouth_opening(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    return abs(top_lip.y - bottom_lip.y)


def detect_emotion(face_landmarks):
    if mouth_opening(face_landmarks) > MOUTH_OPEN_THRESHOLD:
        return "tongue"
    if eye_opening(face_landmarks) > EYE_OPENING_THRESHOLD:
        return "shock"
    if eye_opening(face_landmarks) < SQUINTING_THRESHOLD:
        return "glare"
    return "neutral"


def load_image(label, width, height):
    path = DEFAULT_IMAGES.get(label, DEFAULT_IMAGES["neutral"])
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.resize(img, (width, height))


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Camera not available.")

    width = 640
    height = 480
    fps = 30

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    with mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1,
    ) as face_mesh:

        with pyvirtualcam.Camera(width=width, height=height, fps=fps) as vcam:
            print(f"Virtual camera started: {vcam.device}")

            while True:
                ret, frame = cam.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                label = "neutral"

                if result.multi_face_landmarks:
                    face_landmarks = result.multi_face_landmarks[0]
                    label = detect_emotion(face_landmarks)

                cat_frame = load_image(label, width, height)

                if cat_frame is None:
                    output = frame
                else:
                    output = cat_frame

                # pyvirtualcam ожидает RGB
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                vcam.send(output_rgb)
                vcam.sleep_until_next_frame()


if __name__ == "__main__":
    main()
