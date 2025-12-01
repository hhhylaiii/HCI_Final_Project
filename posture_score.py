import math
import mediapipe as mp

mp_pose = mp.solutions.pose


def _line_tilt_deg(p1, p2):
    """
    Compute tilt of a line segment in degrees relative to horizontal.
    0 deg = perfectly horizontal, 90 deg = vertical.
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = math.degrees(math.atan2(dy, dx))  # [-180, 180]
    angle_abs = abs(angle)
    if angle_abs > 90.0:
        angle_abs = 180.0 - angle_abs
    return angle_abs  # [0, 90]


def extract_face_shoulder_features(landmarks):
    """
    Extract features based on shoulders and face region (frontal view).
    landmarks: list of MediaPipe Pose landmarks (normalized x, y).
    """
    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_EYE = mp_pose.PoseLandmark.LEFT_EYE.value
    R_EYE = mp_pose.PoseLandmark.RIGHT_EYE.value
    M_L = mp_pose.PoseLandmark.MOUTH_LEFT.value
    M_R = mp_pose.PoseLandmark.MOUTH_RIGHT.value

    left_shoulder = landmarks[L_SH]
    right_shoulder = landmarks[R_SH]
    left_eye = landmarks[L_EYE]
    right_eye = landmarks[R_EYE]
    mouth_left = landmarks[M_L]
    mouth_right = landmarks[M_R]

    # 1) Shoulder tilt in degrees (0 = horizontal)
    shoulder_tilt_deg = _line_tilt_deg(left_shoulder, right_shoulder)

    # 2) Head roll (eye line tilt) in degrees (0 = horizontal)
    head_roll_deg = _line_tilt_deg(left_eye, right_eye)

    # 3) Distance indicator: shoulder width in normalized coordinates
    #    Larger value means the user is closer to the camera/screen.
    shoulder_width = math.dist(
        (left_shoulder.x, left_shoulder.y),
        (right_shoulder.x, right_shoulder.y)
    )

    # Optional: face width (not used for scoring but kept for debugging if needed)
    face_width = math.dist(
        (mouth_left.x, mouth_left.y),
        (mouth_right.x, mouth_right.y)
    )

    return {
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "head_roll_deg": head_roll_deg,
        "distance_indicator": shoulder_width,
        "face_width": face_width,
    }


class PostureScore:
    def __init__(self):
        # Exponential moving average smoothing
        self.ALPHA = 0.4
        self.history = None

        # Shoulder tilt thresholds (deg)
        # Tuned based on your experimental results.
        self.SHOULDER_WARN = 5.0
        self.SHOULDER_BAD = 8.0

        # Head roll (eye line tilt) thresholds (deg)
        self.HEAD_ROLL_WARN = 10.0
        self.HEAD_ROLL_BAD = 20.0

        # Distance indicator (shoulder width) thresholds
        # Tuned based on your experimental results.
        self.DIST_WARN = 0.55
        self.DIST_BAD = 0.6

    def smooth_features(self, new_features):
        """Apply EMA smoothing to reduce frame-to-frame noise."""
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
        if deg < self.SHOULDER_WARN:
            return 0
        elif deg < self.SHOULDER_BAD:
            return 10
        else:
            return 20

    def _penalty_from_roll(self, deg):
        if deg < self.HEAD_ROLL_WARN:
            return 0
        elif deg < self.HEAD_ROLL_BAD:
            return 10
        else:
            return 20

    def _penalty_from_distance(self, dist):
        if dist < self.DIST_WARN:
            return 0
        elif dist < self.DIST_BAD:
            return 10
        else:
            return 20

    def compute(self, features):
        """
        Compute posture score and per-dimension penalties.
        """
        f = self.smooth_features(features)

        penalties = {}
        penalties["shoulder_tilt"] = self._penalty_from_tilt(f["shoulder_tilt_deg"])
        penalties["head_roll"] = self._penalty_from_roll(f["head_roll_deg"])
        penalties["head_distance"] = self._penalty_from_distance(f["distance_indicator"])

        total_penalty = sum(penalties.values())  # 0 ~ 60
        score = max(0, 100 - total_penalty)

        active_labels = [name for name, p in penalties.items() if p > 0]
        status = "Good" if not active_labels else ", ".join(active_labels)

        return {
            "score": int(score),
            "status": status,
            "penalties": penalties,
            "features": f,
        }
