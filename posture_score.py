import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_face_shoulder_features(landmarks, img_w, img_h):
    """
    更新：加入鼻子與肩膀的幾何計算來判斷駝背。
    """
    # 取得關鍵點索引
    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_EYE = mp_pose.PoseLandmark.LEFT_EYE.value
    R_EYE = mp_pose.PoseLandmark.RIGHT_EYE.value
    NOSE = mp_pose.PoseLandmark.NOSE.value  # 新增鼻子
    
    # --- 1. 轉換為像素座標 [x, y] ---
    l_shoulder = [landmarks[L_SH].x * img_w, landmarks[L_SH].y * img_h]
    r_shoulder = [landmarks[R_SH].x * img_w, landmarks[R_SH].y * img_h]
    
    l_eye = [landmarks[L_EYE].x * img_w, landmarks[L_EYE].y * img_h]
    r_eye = [landmarks[R_EYE].x * img_w, landmarks[R_EYE].y * img_h]
    
    nose = [landmarks[NOSE].x * img_w, landmarks[NOSE].y * img_h]
    
    # --- 2. 肩膀傾斜 (Shoulder Tilt) ---
    sh_dy = l_shoulder[1] - r_shoulder[1]
    sh_dx = l_shoulder[0] - r_shoulder[0]
    shoulder_tilt_deg = abs(math.degrees(math.atan2(sh_dy, abs(sh_dx))))

    # --- 3. 頭部歪斜 (Head Roll) ---
    eye_dy = l_eye[1] - r_eye[1]
    eye_dx = l_eye[0] - r_eye[0]
    head_roll_raw = math.degrees(math.atan2(eye_dy, eye_dx))
    head_roll_deg = abs(head_roll_raw)

    # --- 4. 眼睛離螢幕太近 (Distance Check) ---
    eye_dist_px = math.hypot(eye_dx, eye_dy)

    # --- 5. [新增] 駝背檢測 (Hunchback / Slouch Check) ---
    # 計算邏輯：計算 "鼻子到左肩" 與 "鼻子到右肩" 兩條向量的夾角
    # 向量 A: Nose -> L_Shoulder
    vec_n_l = [l_shoulder[0] - nose[0], l_shoulder[1] - nose[1]]
    # 向量 B: Nose -> R_Shoulder
    vec_n_r = [r_shoulder[0] - nose[0], r_shoulder[1] - nose[1]]
    
    # 計算向量長度
    len_n_l = math.hypot(vec_n_l[0], vec_n_l[1])
    len_n_r = math.hypot(vec_n_r[0], vec_n_r[1])
    
    # 使用餘弦定理或點積公式計算夾角: A . B = |A||B|cos(theta)
    # cos(theta) = (A . B) / (|A| * |B|)
    dot_product = vec_n_l[0] * vec_n_r[0] + vec_n_l[1] * vec_n_r[1]
    
    # 防呆：避免分母為 0
    if len_n_l * len_n_r == 0:
        nose_shoulder_angle = 0
    else:
        cos_theta = dot_product / (len_n_l * len_n_r)
        # 數值限制在 -1 到 1 之間，避免浮點數誤差導致 crash
        cos_theta = max(-1.0, min(1.0, cos_theta))
        nose_shoulder_angle = math.degrees(math.acos(cos_theta))

    return {
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "head_roll_deg": head_roll_deg,
        "head_roll_raw": head_roll_raw, 
        "eye_dist_px": eye_dist_px,
        "distance_indicator": eye_dist_px,
        "nose_shoulder_angle": nose_shoulder_angle # 新增回傳值
    }

class PostureScore:
    def __init__(self):
        self.ALPHA = 0.4
        self.history = None
        self.baseline = None
        
        # 1. 角度類 (Degrees) - 左右傾斜
        self.SHOULDER_WARN = 5.0
        self.SHOULDER_BAD = 10.0
        self.HEAD_ROLL_WARN = 10.0
        self.HEAD_ROLL_BAD = 20.0

        # 2. [新增] 駝背類、低頭 (Degrees) - 鼻肩夾角
        # 正常坐姿時，這個角度通常較小 (例如 85度，視攝像頭距離而定)
        # 當人駝背、頭往前伸時，鼻子靠近肩膀連線，角度會變大
        self.HUNCH_WARN = 85.0 
        self.HUNCH_BAD = 100.0

        # 3. 距離類 (Pixels)
        self.DIST_WARN = 90.0 
        self.DIST_BAD = 110.0

    def set_baseline(self, features):
        """Set the baseline features for relative scoring."""
        self.baseline = features

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
        if self.baseline:
            # Relative deviation
            diff = abs(deg - self.baseline["shoulder_tilt_deg"])
            if diff < 5.0: return 0
            elif diff < 10.0: return 10 
            else: return 20
        else:
            if deg < self.SHOULDER_WARN: return 0
            elif deg < self.SHOULDER_BAD: return 10
            else: return 20

    def _penalty_from_roll(self, deg):
        if self.baseline:
            # Relative deviation
            diff = abs(deg - self.baseline["head_roll_deg"])
            if diff < 10.0: return 0
            elif diff < 15.0: return 10
            else: return 20
        else:
            if deg < self.HEAD_ROLL_WARN: return 0
            elif deg < self.HEAD_ROLL_BAD: return 10
            else: return 20
    
    def _penalty_from_distance(self, dist_px):
        if self.baseline:
            # Relative deviation: positive means closer than baseline
            diff = dist_px - self.baseline["eye_dist_px"]
            if diff < 10.0: return 0
            elif diff < 30.0 : return 10
            else: return 20
        else:
            if dist_px < self.DIST_WARN: return 0
            elif dist_px < self.DIST_BAD: return 10
            else: return 20

    def _penalty_from_hunch(self, angle):
        if self.baseline:
            # Relative deviation: positive means angle increased (more hunch)
            diff = angle - self.baseline["nose_shoulder_angle"]
            if diff < 10.0: return 0
            elif diff < 25.0: return 15
            else: return 30
        else:
            if angle < self.HUNCH_WARN: return 0
            elif angle < self.HUNCH_BAD: return 15
            else: return 30  # 駝背通常是比較嚴重的姿勢問題，扣分重一點

    def compute(self, features):
        f = self.smooth_features(features)
        penalties = {}
        
        penalties["shoulder_tilt"] = self._penalty_from_tilt(f["shoulder_tilt_deg"])
        penalties["head_roll"] = self._penalty_from_roll(f["head_roll_deg"])
        penalties["head_distance"] = self._penalty_from_distance(f["eye_dist_px"])
        
        # 新增駝背計算
        penalties["hunchback"] = self._penalty_from_hunch(f["nose_shoulder_angle"])

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