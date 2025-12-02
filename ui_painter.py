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
    """
    h, w = image.shape[:2]

    if result is not None:
        current_score = result["score"]
        current_status = result["status"]
        smoothed_feats = result["features"]
        penalties = result["penalties"]

        # 1. 設定顏色：分數高綠色，中等黃色，低分紅色
        if current_score >= 80:
            color = (0, 255, 0)  # Green
        elif current_score >= 50:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # 2. 繪製總分與狀態
        cv2.putText(
            image, f"Score: {current_score}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            image, f"Status: {current_status}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

        # 3. 詳細數值顯示 (除錯用，黑色字體)
        y_start = 110
        gap = 25  # 稍微加大行距比較好讀

        def draw_metric(img, name, value, penalty, y):
            # 為了避免 value 是 None 的情況，加個簡單防呆
            val_display = value if value is not None else 0.0
            pen_display = int(penalty) if penalty is not None else 0
            
            text = f"{name}: {val_display:.1f} (Pen: {pen_display})"
            
            # 根據是否有被扣分來改變文字顏色 (有扣分變紅，沒扣分黑色)
            text_color = (0, 0, 255) if pen_display > 0 else (0, 0, 0)
            
            cv2.putText(
                img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                text_color,
                1,
                cv2.LINE_AA
            )

        # --- A. 肩膀傾斜 ---
        draw_metric(
            image, "Tilt (deg)",
            smoothed_feats.get("shoulder_tilt_deg", 0),
            penalties.get("shoulder_tilt", 0),
            y_start
        )
        
        # --- B. 頭部歪斜 ---
        draw_metric(
            image, "Roll (deg)",
            smoothed_feats.get("head_roll_deg", 0),
            penalties.get("head_roll", 0),
            y_start + gap
        )
        
        # --- C. 距離檢測 ---
        draw_metric(
            image, "Dist (px)",
            smoothed_feats.get("eye_dist_px", 0), # UI顯示建議用 eye_dist_px 比較直觀
            penalties.get("head_distance", 0),
            y_start + gap * 2
        )
        
        # --- D. [新增] 駝背檢測 (Hunchback) ---
        draw_metric(
            image, "Hunch (deg)", 
            smoothed_feats.get("nose_shoulder_angle", 0), # 取得計算出的角度
            penalties.get("hunchback", 0),                # 取得駝背扣分
            y_start + gap * 3
        )

    # 4. FPS 顯示
    if fps is not None:
        cv2.putText(
            image, f"FPS: {int(fps)}",
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (50, 50, 50),
            2
        )