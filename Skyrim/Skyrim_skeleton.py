import cv2
import shutil
import subprocess
import time
from pathlib import Path

try:
    import mediapipe as mp
except ImportError:
    mp = None


VIDEO_PATH = Path(r"C:\Users\alias\Downloads\skyrim-skeleton.mp4")

AWAY_SECONDS = 0.8
REARM_SECONDS = 0.25
CALIBRATION_SECONDS = 1.5
CALIBRATION_MIN_SAMPLES = 10

EYE_HORIZONTAL_START_DELTA = 0.08
EYE_VERTICAL_START_DELTA = 0.09
HEAD_START_DELTA = 0.07

EYE_HORIZONTAL_STOP_DELTA = 0.05
EYE_VERTICAL_STOP_DELTA = 0.06
HEAD_STOP_DELTA = 0.045

SMOOTHING = 0.35

FACE_MISSING_IS_AWAY = True
VLC_PATHS = (
    Path(r"C:\Program Files\VideoLAN\VLC\vlc.exe"),
    Path(r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"),
)

LEFT_EYE = {
    "upper": 159,
    "lower": 145,
    "left": 33,
    "right": 133,
    "iris": 468,
}

RIGHT_EYE = {
    "upper": 386,
    "lower": 374,
    "left": 362,
    "right": 263,
    "iris": 473,
}


def make_calibration():
    return {
        "started_at": time.time(),
        "ready": False,
        "vertical_samples": [],
        "horizontal_samples": [],
        "head_samples": [],
        "vertical": 0.5,
        "horizontal": 0.5,
        "head": 0.5,
    }


def find_player():
    vlc = shutil.which("vlc")
    if vlc:
        return "vlc", Path(vlc)

    for vlc_path in VLC_PATHS:
        if vlc_path.exists():
            return "vlc", vlc_path

    ffplay = shutil.which("ffplay")
    if ffplay:
        return "ffplay", Path(ffplay)

    return None, None


def open_video(video_path: Path):
    player_name, player_path = find_player()
    if player_path is None:
        print("No player with audio found. Install VLC or ffmpeg/ffplay.")
        return None

    if player_name == "vlc":
        command = [
            str(player_path),
            "--play-and-exit",
            "--no-video-title-show",
            "--no-qt-privacy-ask",
            str(video_path),
        ]
    else:
        command = [
            str(player_path),
            "-autoexit",
            "-loglevel",
            "quiet",
            str(video_path),
        ]

    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        print(f"Could not start video player: {error}")
        return None


def close_video(process):
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()


def update_video(process):
    if process is not None and process.poll() is not None:
        return None

    return process


def draw_warning(frame, text="look at screen"):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.1
    thickness = 2
    text_size, _ = cv2.getTextSize(text.upper(), font, font_scale, thickness)
    box_w = min(max(text_size[0] + 48, 360), w - 20)
    box_h = 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 0, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160), 3)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 24, y1 + 48),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def axis_ratio(value, edge_a, edge_b):
    start = min(edge_a, edge_b)
    end = max(edge_a, edge_b)
    return (value - start) / ((end - start) + 1e-6)


def smooth(previous, current):
    if previous is None:
        return current

    return previous * (1 - SMOOTHING) + current * SMOOTHING


def eye_ratios(landmarks, eye):
    upper = landmarks[eye["upper"]]
    lower = landmarks[eye["lower"]]
    left = landmarks[eye["left"]]
    right = landmarks[eye["right"]]
    iris = landmarks[eye["iris"]]

    vertical = axis_ratio(iris.y, upper.y, lower.y)
    horizontal = axis_ratio(iris.x, left.x, right.x)
    return vertical, horizontal


def head_ratio(landmarks):
    nose = landmarks[1]
    left_face = landmarks[234]
    right_face = landmarks[454]
    return axis_ratio(nose.x, left_face.x, right_face.x)


def gaze_metrics(landmarks):
    left_vertical, left_horizontal = eye_ratios(landmarks, LEFT_EYE)
    right_vertical, right_horizontal = eye_ratios(landmarks, RIGHT_EYE)

    vertical = (left_vertical + right_vertical) / 2
    horizontal = (left_horizontal + right_horizontal) / 2
    head = head_ratio(landmarks)
    return vertical, horizontal, head


def update_calibration(calibration, vertical, horizontal, head, now):
    if calibration["ready"]:
        return

    calibration["vertical_samples"].append(vertical)
    calibration["horizontal_samples"].append(horizontal)
    calibration["head_samples"].append(head)

    enough_time = now - calibration["started_at"] >= CALIBRATION_SECONDS
    enough_samples = len(calibration["vertical_samples"]) >= CALIBRATION_MIN_SAMPLES
    if not enough_time or not enough_samples:
        return

    calibration["vertical"] = mean(calibration["vertical_samples"])
    calibration["horizontal"] = mean(calibration["horizontal_samples"])
    calibration["head"] = mean(calibration["head_samples"])
    calibration["ready"] = True


def mean(values):
    return sum(values) / len(values)


def gaze_deltas(vertical, horizontal, head, calibration):
    vertical_delta = abs(vertical - calibration["vertical"])
    horizontal_delta = abs(horizontal - calibration["horizontal"])
    head_delta = abs(head - calibration["head"])
    return vertical_delta, horizontal_delta, head_delta


def is_looking_away(vertical, horizontal, head, calibration, already_away):
    if not calibration["ready"]:
        return False

    vertical_delta, horizontal_delta, head_delta = gaze_deltas(
        vertical,
        horizontal,
        head,
        calibration,
    )

    if already_away:
        return (
            vertical_delta > EYE_VERTICAL_STOP_DELTA
            or horizontal_delta > EYE_HORIZONTAL_STOP_DELTA
            or head_delta > HEAD_STOP_DELTA
        )

    return (
        vertical_delta > EYE_VERTICAL_START_DELTA
        or horizontal_delta > EYE_HORIZONTAL_START_DELTA
        or head_delta > HEAD_START_DELTA
    )


def main():
    if mp is None:
        print("MediaPipe is not installed. Install it with: pip install mediapipe")
        return

    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera")
        face_mesh.close()
        return

    away_start = None
    screen_start = None
    alarm_ready = True
    video_process = None
    calibration = make_calibration()
    smoothed_vertical = None
    smoothed_horizontal = None
    smoothed_head = None

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, _ = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            now = time.time()
            looking_away = False
            vertical = None
            horizontal = None
            head = None
            vertical_delta = None
            horizontal_delta = None
            head_delta = None

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                vertical, horizontal, head = gaze_metrics(landmarks)
                smoothed_vertical = smooth(smoothed_vertical, vertical)
                smoothed_horizontal = smooth(smoothed_horizontal, horizontal)
                smoothed_head = smooth(smoothed_head, head)
                vertical = smoothed_vertical
                horizontal = smoothed_horizontal
                head = smoothed_head

                update_calibration(calibration, vertical, horizontal, head, now)
                looking_away = is_looking_away(
                    vertical,
                    horizontal,
                    head,
                    calibration,
                    away_start is not None,
                )
                if calibration["ready"]:
                    vertical_delta, horizontal_delta, head_delta = gaze_deltas(
                        vertical,
                        horizontal,
                        head,
                        calibration,
                    )
            elif FACE_MISSING_IS_AWAY:
                looking_away = calibration["ready"]

            if video_process is not None and not looking_away:
                close_video(video_process)
                video_process = None

            if looking_away:
                screen_start = None

                if away_start is None:
                    away_start = now

                away_time = now - away_start
                if alarm_ready and video_process is None and away_time >= AWAY_SECONDS:
                    started_process = open_video(VIDEO_PATH)
                    if started_process is not None:
                        video_process = started_process
                        alarm_ready = False
            else:
                away_start = None

                if screen_start is None:
                    screen_start = now

                if now - screen_start >= REARM_SECONDS:
                    alarm_ready = True

            previous_video_process = video_process
            video_process = update_video(video_process)
            if previous_video_process is not None and video_process is None and looking_away:
                alarm_ready = True

            if vertical is not None and horizontal is not None and head is not None:
                cv2.putText(
                    frame,
                    f"v:{vertical:.3f} h:{horizontal:.3f} head:{head:.3f}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if calibration["ready"] and vertical_delta is not None:
                cv2.putText(
                    frame,
                    f"dv:{vertical_delta:.3f} dh:{horizontal_delta:.3f} head:{head_delta:.3f}",
                    (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            elif not calibration["ready"]:
                elapsed = now - calibration["started_at"]
                remaining = max(0.0, CALIBRATION_SECONDS - elapsed)
                draw_warning(frame, f"look at screen {remaining:.1f}s")

            if away_start and alarm_ready:
                remaining = max(0.0, AWAY_SECONDS - (now - away_start))
                draw_warning(frame, f"look at screen {remaining:.1f}s")

            cv2.imshow("lock in", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("c"):
                calibration = make_calibration()
                smoothed_vertical = None
                smoothed_horizontal = None
                smoothed_head = None
                away_start = None
                screen_start = None
                alarm_ready = True

    finally:
        close_video(video_process)
        cam.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
