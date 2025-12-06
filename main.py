import cv2
import time
import mediapipe as mp
import numpy as np
import os
import threading
import pyttsx3
import matplotlib.pyplot as plt
from datetime import datetime

# 引用您原本的模組 (請確保這些檔案在同一目錄下)
from posture_score import extract_face_shoulder_features, PostureScore
from posture_history import PostureHistory
from ui_painter import draw_pose_landmarks, draw_posture_ui

# --- 設定全域常數 ---
W, H = 1280, 720
POMODORO_LIMIT_SECONDS = 30 * 60  # 久坐提醒時間 (30分鐘)
LOW_SCORE_THRESHOLD = 70          # 低於幾分開始警告
WARNING_COOLDOWN = 5.0            # 語音警告冷卻時間 (秒)

class VoiceAssistant:
    """處理語音輸出的類別，使用執行緒避免卡住畫面"""
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.is_speaking = False
        except:
            print("語音模組初始化失敗，將以純文字顯示。")
            self.engine = None

    def _speak_thread(self, text):
        if self.engine:
            self.is_speaking = True
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass
            self.is_speaking = False

    def say(self, text):
        if self.engine and not self.is_speaking:
            t = threading.Thread(target=self._speak_thread, args=(text,))
            t.start()

def generate_report(score_log, duration_minutes):
    """程式結束時生成圖表"""
    if not score_log:
        print("沒有足夠數據生成報告。")
        return

    times, scores = zip(*score_log)
    
    # 計算統計數據
    avg_score = sum(scores) / len(scores)
    good_time = sum(1 for s in scores if s > 80)
    good_ratio = (good_time / len(scores)) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(times, scores, label='Posture Score', color='blue', linewidth=1.5)
    plt.axhline(y=80, color='g', linestyle='--', label='Excellent (80)')
    plt.axhline(y=60, color='r', linestyle='--', label='Poor (60)')
    
    plt.title(f"Posture Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 加入文字統計
    info_text = (f"Duration: {duration_minutes:.1f} mins\n"
                 f"Avg Score: {avg_score:.1f}\n"
                 f"Good Posture: {good_ratio:.1f}%")
    plt.gcf().text(0.15, 0.15, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))

    filename = f"posture_report_{int(time.time())}.png"
    plt.savefig(filename)
    print(f"健康報告已儲存為: {filename}")
    # 如果想直接顯示，把下面這行取消註解
    # plt.show()

def main():
    cap = cv2.VideoCapture(0)
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
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 0, (W, H))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (W, H), 5)
            print("已載入相機校正參數。")
        except Exception as e:
            print(f"載入參數失敗: {e}")

    # --- 初始化模組 ---
    scorer = PostureScore()
    history = PostureHistory()
    voice = VoiceAssistant()
    
    # 啟用 segmentation_mask 用於背景模糊
    mp_pose = mp.solutions.pose
    
    # --- 狀態變數 ---
    is_calibrated = False
    calibration_data = []
    CALIBRATION_FRAMES = 90
    
    # 新功能變數
    enable_blur = False          # 背景模糊開關
    start_time = time.time()     # 程式開始時間
    pomodoro_start = time.time() # 番茄鐘開始時間
    last_warning_time = 0        # 上次語音警告時間
    long_term_history = []       # 用於最後畫圖的數據 (時間, 分數)
    
    print("--- 操作說明 ---")
    print("按 'b': 切換背景模糊 (隱私模式)")
    print("按 'r': 重置久坐計時器")
    print("按 'q': 結束程式並生成報告")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True  # 關鍵：開啟分割功能以支援模糊
    ) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # 1. 畸變修正
            if mapx is not None and mapy is not None:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 2. MediaPipe 處理
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame.flags.writeable = True
            
            # 3. 背景模糊處理 (如果開啟)
            if enable_blur and results.segmentation_mask is not None:
                # 建立遮罩: 1 是人, 0 是背景
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = cv2.GaussianBlur(frame, (55, 55), 0) # 對原圖做模糊
                # 混合：如果是人就用原圖，不然用模糊圖
                frame = np.where(condition, frame, bg_image)

            frame_bgr = frame # 此時已經可能是模糊過的背景

            result_dict = None
            is_user_present = False

            # 4. 姿勢判斷核心邏輯
            if results.pose_landmarks:
                is_user_present = True
                h, w = frame_bgr.shape[:2]

                # 畫骨架
                draw_pose_landmarks(frame_bgr, results.pose_landmarks)
                
                features = extract_face_shoulder_features(results.pose_landmarks.landmark, w, h)

                if not is_calibrated:
                    # --- 校正階段 ---
                    calibration_data.append(features)
                    progress = len(calibration_data) / CALIBRATION_FRAMES
                    
                    # 畫進度條
                    bar_w, bar_h = 400, 30
                    x_start, y_start = (w - bar_w) // 2, h - 100
                    cv2.rectangle(frame_bgr, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (100, 100, 100), -1)
                    cv2.rectangle(frame_bgr, (x_start, y_start), (x_start + int(bar_w * progress), y_start + bar_h), (0, 255, 0), -1)
                    cv2.putText(frame_bgr, "Calibrating... Sit Upright", (x_start, y_start - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if len(calibration_data) >= CALIBRATION_FRAMES:
                        avg_features = {k: sum(d[k] for d in calibration_data) / len(calibration_data) for k in features}
                        scorer.set_baseline(avg_features)
                        is_calibrated = True
                        voice.say("校正完成，開始監控")
                        print("Calibration complete.")
                
                else:
                    # --- 監控階段 ---
                    result_dict = scorer.compute(features)
                    history.update(current_time, result_dict)
                    
                    score = result_dict.get("score", 100)
                    
                    # 記錄數據給最後的圖表
                    long_term_history.append((elapsed_time, score))

                    # [功能] 語音警告 (低分且冷卻時間已過)
                    if score < LOW_SCORE_THRESHOLD:
                        if (current_time - last_warning_time) > WARNING_COOLDOWN:
                            voice.say("請坐好，注意姿勢")
                            last_warning_time = current_time

            else:
                # --- [功能] 離席偵測 ---
                # 如果沒抓到人，就不扣分，顯示 "User Away"
                cv2.putText(frame_bgr, "User Away - Paused", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
                
                # 選項：離席時是否要重置久坐計時器？ 這裡假設離席不算休息，只有按 'r' 才算
                # 這裡可以根據需求修改邏輯

            # 5. [功能] 久坐提醒 (番茄鐘)
            pomodoro_elapsed = current_time - pomodoro_start
            time_left = POMODORO_LIMIT_SECONDS - pomodoro_elapsed
            
            # 顯示倒數計時
            timer_color = (0, 255, 0)
            if time_left < 60: timer_color = (0, 0, 255) # 最後一分鐘變紅
            
            minutes = int(time_left // 60)
            seconds = int(time_left % 60)
            timer_text = f"Break in: {minutes:02d}:{seconds:02d}"
            
            if time_left <= 0:
                cv2.putText(frame_bgr, "TIME TO STAND UP!", (W//2 - 200, H//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if int(current_time) % 5 == 0: # 每5秒叫一次
                    voice.say("時間到了，請起來活動一下")
            else:
                cv2.putText(frame_bgr, timer_text, (W - 250, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)

            # 6. 繪製標準 UI
            # 傳遞 fps 資訊
            fps = 1.0 / (time.time() - (current_time - 0.01)) # 簡單估算
            draw_posture_ui(frame_bgr, result_dict, fps=fps, history_summary=history.snapshot())

            cv2.imshow("Smart Posture Assistant", frame_bgr)
            
            # 7. 鍵盤控制
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                enable_blur = not enable_blur
                status = "ON" if enable_blur else "OFF"
                print(f"背景模糊: {status}")
            elif key == ord("r"):
                pomodoro_start = time.time()
                voice.say("計時器已重置")
                print("Timer Reset")

    cap.release()
    cv2.destroyAllWindows()
    
    # 8. [功能] 程式結束，生成報告
    print("正在生成健康報告...")
    generate_report(long_term_history, (time.time() - start_time)/60)

if __name__ == "__main__":
    main()