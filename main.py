import cv2
import time
import mediapipe as mp
import numpy as np
import os
from posture_score import extract_face_shoulder_features, PostureScore
from ui_painter import draw_pose_landmarks, draw_posture_ui


# for test
def main():
    cap = cv2.VideoCapture(0)
    # 解析度設定需與標定時一致
    W, H = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- [張氏標定參數載入] ---
    mapx, mapy = None, None
    if os.path.exists("camera_params.npz"):
        try:
            with np.load("camera_params.npz") as data:
                mtx = data['mtx']
                dist = data['dist']
            
            print("已載入相機校正參數，啟用畸變修正。")
            
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 0, (W, H))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (W, H), 5)
            
        except Exception as e:
            print(f"載入參數失敗: {e}")
    else:
        print("警告：找不到 'camera_params.npz'。將使用原始畫面 (無畸變修正)。")
        print("請先執行 calibration.py 進行標定。")


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

            # --- [畸變修正] ---
            if mapx is not None and mapy is not None:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # Pose estimation
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            frame.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            result_dict = None

            if results.pose_landmarks:
                h, w = frame.shape[:2]

                # Draw skeleton
                draw_pose_landmarks(frame_bgr, results.pose_landmarks)

                # Feature extraction + scoring
                # 傳入 2D landmarks 與畫面寬高 (Pixel Calculation)
                features = extract_face_shoulder_features(
                    results.pose_landmarks.landmark, 
                    w, 
                    h
                )
                result_dict = scorer.compute(features)

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Draw UI
            draw_posture_ui(frame_bgr, result_dict, fps=fps)

            cv2.imshow("Posture Assistant (Calibrated)", frame_bgr)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()