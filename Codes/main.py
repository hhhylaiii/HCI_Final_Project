import cv2
import time
import mediapipe as mp
import numpy as np
import os

from posture_score import extract_face_shoulder_features, PostureScore
from posture_history import PostureHistory
from ui_painter import draw_pose_landmarks, draw_posture_ui
from voice_assistant import VoiceAssistant
from report_generator import generate_report

# --- 設定全域常數 ---
W, H = 1280, 720
POMODORO_LIMIT_SECONDS =  60  # 久坐提醒時間 (30分鐘)
LOW_SCORE_THRESHOLD = 70      # 低於幾分開始警告
WARNING_COOLDOWN = 5.0        # 語音警告冷卻時間 (秒)



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
    last_pomodoro_warning_time = 0 # 上次番茄鐘語音警告時間
    long_term_history = []       # 用於最後畫圖的數據 (時間, 分數)
    
    # 用戶離席相關變數
    was_user_present = False     # 上一幀用戶是否在場
    paused_duration = 0          # 累計暫停時間 (用於計算有效的久坐時間)
    pause_start_time = None      # 開始暫停的時間點
    away_periods = []            # 離席時段列表 [(start_elapsed, end_elapsed), ...]
    
    # 防抖動相關變數 (Debounce)
    LOW_SCORE_DEBOUNCE = 1.5     # 低分警告延遲時間 (秒)
    RETURN_DEBOUNCE = 3.0        # 用戶回來確認時間 (秒)
    low_score_start_time = None  # 低分開始時間 (用於延遲警告)
    user_return_start_time = None # 用戶回來開始時間 (用於確認真正坐下)
    user_confirmed_back = True   # 用戶是否已確認回來
    
    warning_text = ""
    warning_display_start = 0
    WARNING_DURATION = 1.0
    
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
                
                # 用戶回來時的處理 (加入防抖動)
                # 條件: 用戶剛被偵測到 (之前不在場)，已校正，尚未確認回來，且尚未開始計時
                if not was_user_present and is_calibrated and not user_confirmed_back and user_return_start_time is None:
                    # 用戶剛被偵測到且之前已離開，開始計時確認
                    user_return_start_time = current_time
                    print("偵測到用戶，等待確認...")
                
                # 檢查是否處於確認回來的狀態
                is_confirming_return = (user_return_start_time is not None and not user_confirmed_back and is_calibrated)
                
                if is_confirming_return:
                    # --- 確認回來中 (3秒確認期) ---
                    confirm_elapsed = current_time - user_return_start_time
                    confirm_progress = min(confirm_elapsed / RETURN_DEBOUNCE, 1.0)
                    
                    if confirm_elapsed >= RETURN_DEBOUNCE:
                        # 用戶已穩定坐下 3 秒，確認回來
                        pomodoro_start = time.time()
                        paused_duration = 0
                        # 記錄離席時段結束
                        if pause_start_time is not None:
                            away_end = elapsed_time
                            away_start = away_end - (current_time - pause_start_time)
                            away_periods.append((away_start, away_end))
                        pause_start_time = None
                        user_confirmed_back = True
                        user_return_start_time = None
                        print("用戶回來 - 計時器重置")
                        voice.say("歡迎回來，計時器已重置")
                    else:
                        # 顯示確認中的 UI (不進行姿勢評分)
                        cv2.putText(frame_bgr, "Confirming return...", (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
                        
                        # 畫確認進度條
                        bar_w, bar_h = 300, 20
                        x_start, y_start = 50, 130
                        cv2.rectangle(frame_bgr, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (100, 100, 100), -1)
                        cv2.rectangle(frame_bgr, (x_start, y_start), (x_start + int(bar_w * confirm_progress), y_start + bar_h), (0, 200, 255), -1)
                        cv2.putText(frame_bgr, f"{confirm_elapsed:.1f}s / {RETURN_DEBOUNCE:.0f}s", 
                                    (x_start + bar_w + 10, y_start + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        
                        # 畫骨架 (但不評分)
                        draw_pose_landmarks(frame_bgr, results.pose_landmarks)
                
                elif not is_calibrated:
                    # --- 校正階段 ---
                    # 畫骨架
                    draw_pose_landmarks(frame_bgr, results.pose_landmarks)
                    
                    features = extract_face_shoulder_features(results.pose_landmarks.landmark, w, h)
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
                        voice.say("")
                        print("Calibration complete.")
                
                else:
                    # --- 正常監控階段 ---
                    # 畫骨架
                    draw_pose_landmarks(frame_bgr, results.pose_landmarks)
                    
                    features = extract_face_shoulder_features(results.pose_landmarks.landmark, w, h)
                    result_dict = scorer.compute(features)
                    history.update(current_time, result_dict)
                    
                    score = result_dict.get("score", 100)
                    
                    # 記錄數據給最後的圖表
                    long_term_history.append((elapsed_time, score))

                    # [功能] 語音警告 (低分且持續 1.5 秒)
                    if score < LOW_SCORE_THRESHOLD:
                        # 開始計時低分持續時間
                        if low_score_start_time is None:
                            low_score_start_time = current_time
                        elif (current_time - low_score_start_time) >= LOW_SCORE_DEBOUNCE:
                            # 低分已持續足夠時間，且冷却時間已過
                            if (current_time - last_warning_time) > WARNING_COOLDOWN:
                                print("低分持續，發出警告")
                                voice.say("請坐好，注意姿勢")
                                last_warning_time = current_time
                    else:
                        # 分數恢復，重置低分計時器
                        low_score_start_time = None

            else:
                # --- [功能] 離席偵測 ---
                # 如果沒抓到人，就不扣分，顯示 "User Away"
                cv2.putText(frame_bgr, "User Away - Paused", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
                
                # 用戶離開時開始計算暫停時間
                if was_user_present:
                    # 用戶剛剛離開，開始暫停計時
                    pause_start_time = current_time
                    user_confirmed_back = False  # 重置確認狀態
                    print("用戶離席 - 計時器暫停")
                
                # 用戶離開時重置回來計時器
                user_return_start_time = None
                
                # 用戶離開時重置低分計時器 (防止誤報)
                low_score_start_time = None

            # 5. [功能] 久坐提醒 (番茄鐘) - 只在用戶確認在場時運行
            if is_user_present and user_confirmed_back:
                pomodoro_elapsed = current_time - pomodoro_start - paused_duration
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
                    if (current_time - last_pomodoro_warning_time) > 5.0: # 每5秒叫一次
                        voice.say("時間到了，請起來活動一下")
                        last_pomodoro_warning_time = current_time
                else:
                    cv2.putText(frame_bgr, timer_text, (W - 350, 32), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)
            else:
                # 用戶不在場或確認中，顯示暫停狀態
                cv2.putText(frame_bgr, "Timer: PAUSED", (W - 350, 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            tips_color = (0, 0, 0) 
            cv2.putText(frame_bgr, "'b': Blur background", (10, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tips_color, 2)
            cv2.putText(frame_bgr, "'r': Reset Timer", (10, H - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tips_color, 2)
            cv2.putText(frame_bgr, "'q': Quit & Generate Report", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tips_color, 2)

            if warning_text and (current_time - warning_display_start) < WARNING_DURATION:
                cv2.putText(frame_bgr, warning_text, (W//2 - 250, H - 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            elif (current_time - warning_display_start) >= WARNING_DURATION:
                warning_text = "" 

            # 6. 繪製標準 UI (只在用戶確認在場時顯示姿勢評分)
            # 傳遞 fps 資訊
            fps = 1.0 / (time.time() - (current_time - 0.01)) # 簡單估算
            if is_user_present and user_confirmed_back:
                draw_posture_ui(frame_bgr, result_dict, fps=fps, history_summary=history.snapshot())
            else:
                # 用戶不在場或確認中只顯示 FPS
                draw_posture_ui(frame_bgr, None, fps=fps, history_summary=None)
            
            # 更新用戶在場狀態
            was_user_present = is_user_present

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
                # 計算當前的剩餘時間 (如果用戶在場)
                if is_user_present:
                    current_elapsed = current_time - pomodoro_start - paused_duration
                    current_time_left = POMODORO_LIMIT_SECONDS - current_elapsed
                else:
                    current_time_left = 1  # 用戶不在場時不允許重置
                
                if current_time_left <= 0:
                    pomodoro_start = time.time()
                    paused_duration = 0  # 重置暫停時間
                    voice.stop() # 清除之前的語音排程
                    voice.say("計時器已重置")
                    print("Timer Reset")
                else:
                    warning_text = "Cannot reset: Timer is running!"
                    warning_display_start = time.time()
                    print("Cannot reset: Timer is still running.")

    cap.release()
    cv2.destroyAllWindows()
    
    # 8. [功能] 程式結束，生成報告
    # 如果用戶在離席狀態下結束程式，記錄最後一個離席時段
    if pause_start_time is not None:
        final_elapsed = time.time() - start_time
        away_start = final_elapsed - (time.time() - pause_start_time)
        away_periods.append((away_start, final_elapsed))
    
    print("正在生成健康報告...")
    generate_report(long_term_history, (time.time() - start_time)/60, away_periods)

if __name__ == "__main__":
    main()