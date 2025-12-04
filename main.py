import cv2
import time
import mediapipe as mp
import numpy as np
import os
from posture_score import extract_face_shoulder_features, PostureScore
from posture_history import PostureHistory
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
    history = PostureHistory()
    mp_pose = mp.solutions.pose
    prev_time = 0.0

    # Baseline Calibration Variables
    is_calibrated = False
    calibration_data = []
    CALIBRATION_FRAMES = 90  # Approx 3 seconds

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

                # Feature extraction
                features = extract_face_shoulder_features(
                    results.pose_landmarks.landmark, 
                    w, 
                    h
                )

                if not is_calibrated:
                    # --- Calibration Phase ---
                    calibration_data.append(features)
                    
                    # Draw Calibration UI
                    progress = len(calibration_data) / CALIBRATION_FRAMES
                    bar_w = 400
                    bar_h = 30
                    x_start = (w - bar_w) // 2
                    y_start = h - 100
                    
                    # Draw text
                    cv2.putText(frame_bgr, "Calibrating... Please sit upright and look at screen", 
                                (x_start - 50, y_start - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Draw progress bar
                    cv2.rectangle(frame_bgr, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (100, 100, 100), -1)
                    cv2.rectangle(frame_bgr, (x_start, y_start), (x_start + int(bar_w * progress), y_start + bar_h), (0, 255, 0), -1)

                    if len(calibration_data) >= CALIBRATION_FRAMES:
                        # Compute average features
                        avg_features = {}
                        keys = features.keys()
                        for k in keys:
                            values = [d[k] for d in calibration_data]
                            avg_features[k] = sum(values) / len(values)
                        
                        scorer.set_baseline(avg_features)
                        is_calibrated = True
                        print("Baseline calibration complete.")
                
                else:
                    # --- Normal Tracking Phase ---
                    result_dict = scorer.compute(features)
                    
                    # Update history
                    history.update(time.time(), result_dict)

            else:
                if not is_calibrated:
                     cv2.putText(frame_bgr, "Please sit in front of camera to calibrate", 
                                (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Draw UI (only if calibrated or if we want to show FPS during calibration)
            # We pass result_dict which is None during calibration, so draw_posture_ui handles it gracefully (shows nothing or just FPS)
            history_summary = history.snapshot()
            draw_posture_ui(frame_bgr, result_dict, fps=fps, history_summary=history_summary)

            cv2.imshow("Posture Assistant (Calibrated)", frame_bgr)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()