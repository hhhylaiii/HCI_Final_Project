import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def draw_pose_landmarks(image, pose_landmarks):
    """
    Draw pose landmarks (skeleton) on the image.
    """
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )


def draw_posture_ui(image, result, fps=None):
    """
    Draw score, status, detailed metrics (debug), and FPS on the image.

    result: dict returned by PostureScore.compute(...) or None.
    fps: current frames-per-second value (float) or None.
    """
    h, w = image.shape[:2]

    if result is not None:
        current_score = result["score"]
        current_status = result["status"]
        smoothed_feats = result["features"]
        penalties = result["penalties"]

        # Color by score
        if current_score >= 80:
            color = (0, 255, 0)  # Green (80-100)
        elif current_score >= 50:
            color = (0, 255, 255)  # Yellow (50-79)
        else:
            color = (0, 0, 255)  # Red (<50)

        # Score and status
        cv2.putText(
            image, f"Score: {current_score}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            image, f"Status: {current_status}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

        # Detailed metrics (debug values in black)
        y_start = 110
        gap = 22

        def draw_metric(img, name, value, penalty, y):
            text = f"{name}: {value:.2f} (Pen: {int(penalty)})"
            cv2.putText(
                img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        draw_metric(
            image, "Tilt (deg)",
            smoothed_feats["shoulder_tilt_deg"],
            penalties["shoulder_tilt"],
            y_start
        )
        draw_metric(
            image, "Roll (deg)",
            smoothed_feats["head_roll_deg"],
            penalties["head_roll"],
            y_start + gap
        )
        draw_metric(
            image, "Dist (ind)",
            smoothed_feats["distance_indicator"],
            penalties["head_distance"],
            y_start + gap * 2
        )

    # FPS display (always draw if provided)
    if fps is not None:
        cv2.putText(
            image, f"FPS: {int(fps)}",
            (w - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (100, 100, 100),
            1
        )

