import cv2
import time
import mediapipe as mp
from posture_score import extract_face_shoulder_features, PostureScore
from ui_painter import draw_pose_landmarks, draw_posture_ui


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    scorer = PostureScore()
    mp_pose = mp.solutions.pose

    prev_time = 0.0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Pose estimation
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            frame.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            result_dict = None

            if results.pose_landmarks:
                # Draw skeleton
                draw_pose_landmarks(frame_bgr, results.pose_landmarks)

                # Feature extraction + scoring
                landmarks = results.pose_landmarks.landmark
                features = extract_face_shoulder_features(landmarks)
                result_dict = scorer.compute(features)

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Draw UI (score / status / metrics / FPS)
            draw_posture_ui(frame_bgr, result_dict, fps=fps)

            cv2.imshow("Posture Assistant (Frontal)", frame_bgr)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

