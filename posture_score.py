import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_face_shoulder_features(landmarks, img_w, img_h):
    """
    改用像素座標 (Pixel Coordinates) 進行計算。
    需傳入圖片的寬 (img_w) 與 高 (img_h)。
    (已移除膝蓋判斷邏輯)
    """
    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_EYE = mp_pose.PoseLandmark.LEFT_EYE.value
    R_EYE = mp_pose.PoseLandmark.RIGHT_EYE.value
    
    # --- 1. 轉換為像素座標 [x, y] ---
    l_shoulder = [landmarks[L_SH].x * img_w, landmarks[L_SH].y * img_h]
    r_shoulder = [landmarks[R_SH].x * img_w, landmarks[R_SH].y * img_h]
    
    l_eye = [landmarks[L_EYE].x * img_w, landmarks[L_EYE].y * img_h]
    r_eye = [landmarks[R_EYE].x * img_w, landmarks[R_EYE].y * img_h]
    
    # --- 2. 肩膀傾斜 (Shoulder Tilt) ---
    sh_dy = l_shoulder[1] - r_shoulder[1]
    sh_dx = l_shoulder[0] - r_shoulder[0]
    shoulder_tilt_deg = abs(math.degrees(math.atan2(sh_dy, abs(sh_dx))))

    # --- 3. 頭部歪斜 (Head Roll) ---
    eye_dy = l_eye[1] - r_eye[1]
    eye_dx = l_eye[0] - r_eye[0]
    
    head_roll_raw = math.degrees(math.atan2(eye_dy, eye_dx))
    head_roll_deg = abs(head_roll_raw)

    return {
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "head_roll_deg": head_roll_deg,
        "head_roll_raw": head_roll_raw, 
        "distance_indicator": 0, 
    }

class PostureScore:
    def __init__(self):
        self.ALPHA = 0.4
        self.history = None
        
        # 1. 角度類 (Degrees)
        self.SHOULDER_WARN = 5.0
        self.SHOULDER_BAD = 10.0
        self.HEAD_ROLL_WARN = 10.0
        self.HEAD_ROLL_BAD = 20.0

        # 2. 距離類 (Pixels) - 暫不使用
        self.DIST_WARN = 9999
        self.DIST_BAD = 9999

    def smooth_features(self, new_features):
        if self.history is None:
            self.history = new_features
            return new_features
        smoothed = {}
        for k, v in new_features.items():
            prev = self.history.get(k, v)
            smoothed[k] = self.ALPHA * v + (1.0 - self.ALPHA) * prev
        self.history = smoothed
        return smoothed

    def _penalty_from_tilt(self, deg):
        if deg < self.SHOULDER_WARN: return 0
        elif deg < self.SHOULDER_BAD: return 10
        else: return 20

    def _penalty_from_roll(self, deg):
        if deg < self.HEAD_ROLL_WARN: return 0
        elif deg < self.HEAD_ROLL_BAD: return 10
        else: return 20
        
    def compute(self, features):
        f = self.smooth_features(features)
        penalties = {}
        
        penalties["shoulder_tilt"] = self._penalty_from_tilt(f["shoulder_tilt_deg"])
        penalties["head_roll"] = self._penalty_from_roll(f["head_roll_deg"])
        penalties["head_distance"] = 0 
        
        # [已移除] penalties["leg_cross"]

        total_penalty = sum(penalties.values())
        score = max(0, 100 - total_penalty)
        
        active_labels = [name for name, p in penalties.items() if p > 0]
        status = "Good" if not active_labels else ", ".join(active_labels)

        return {
            "score": int(score),
            "status": status,
            "penalties": penalties,
            "features": f,
        }