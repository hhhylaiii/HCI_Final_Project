import cv2
import numpy as np
import os

def run_calibration():
    # --- 設定參數 ---
    # 棋盤格的內角點數量 (例如 9x6 的格子，內角點是 8x5)
    # 請根據您手上的棋盤格修改這裡！
    CHECKERBOARD = (9, 6) 
    SQUARE_SIZE = 25 # 每一格的實際邊長 (單位 mm，僅影響位移 tvecs，不影響畸變修正)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 儲存世界座標點 (3D) 和 影像座標點 (2D)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 定義世界座標系中的棋盤格點 (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    cap = cv2.VideoCapture(0)
    # 解析度需與 main.py 一致
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("--- 張氏相機標定工具 ---")
    print(f"請拿著 {CHECKERBOARD} 的棋盤格在鏡頭前移動")
    print("按 's' 拍攝一張樣本 (建議至少 10 張)")
    print("按 'q' 結束拍攝並開始計算")

    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # 尋找棋盤格角點
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_corners:
            # 畫出來給你看
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)

        cv2.putText(display_frame, f"Images Captured: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Calibration', display_frame)
        key = cv2.waitKey(1)

        if key == ord('s') and ret_corners:
            objpoints.append(objp)
            
            # 優化角點位置
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            count += 1
            print(f"已拍攝第 {count} 張樣本")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < 5:
        print("樣本太少，無法計算 (至少需要 5 張)")
        return

    print("正在計算相機參數... (這可能需要一點時間)")
    
    # 核心算法：張氏標定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print(f"標定誤差 (Reprojection Error): {ret}")
    print("內參矩陣 (Matrix):\n", mtx)
    print("畸變係數 (Distortion):\n", dist)

    # 存檔
    np.savez("camera_params.npz", mtx=mtx, dist=dist)
    print("參數已儲存至 'camera_params.npz'")

if __name__ == '__main__':
    run_calibration()