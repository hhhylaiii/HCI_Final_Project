import time

class PostureHistory:
    """
    Tracks posture over time, calculating statistics and risk metrics.
    Scheme 2 implementation.
    """
    def __init__(self):
        # Time tracking
        self.total_time = 0.0
        self.good_time = 0.0
        self.warning_time = 0.0
        self.bad_time = 0.0

        # State tracking
        self.current_state = "good"  # "good", "warning", "bad"
        self.current_streak_time = 0.0
        self.max_bad_streak = 0.0
        self.bad_episodes_count = 0

        self.last_timestamp = None

    def _classify_state(self, penalties):
        """
        Classify state based on penalties from Scheme 1.
        - Good: all penalties are 0.
        - Bad: at least one penalty >= 20.
        - Warning: any other case.
        """
        if not penalties:
            return "good"

        # Check for bad state (any penalty >= 20)
        # Note: Hunchback penalty can be 30, so we use >= 20
        for p in penalties.values():
            if p >= 20:
                return "bad"

        # Check for good state (all penalties == 0)
        if all(p == 0 for p in penalties.values()):
            return "good"

        # Otherwise it's warning
        return "warning"

    def update(self, timestamp, result_dict):
        """
        Update history with the latest frame results.
        
        Args:
            timestamp (float): Current time in seconds.
            result_dict (dict): Result from PostureScore.compute().
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            # Initialize state based on first frame
            if result_dict and "penalties" in result_dict:
                self.current_state = self._classify_state(result_dict["penalties"])
            return

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        # Avoid large jumps if processing lags (e.g., > 1 sec), cap dt?
        # For now, we trust the timestamp, but safeguard against negative dt
        if dt < 0:
            dt = 0

        # 1. Determine new state
        new_state = "good"
        if result_dict and "penalties" in result_dict:
            new_state = self._classify_state(result_dict["penalties"])

        # 2. Update total times
        self.total_time += dt
        if new_state == "good":
            self.good_time += dt
        elif new_state == "warning":
            self.warning_time += dt
        elif new_state == "bad":
            self.bad_time += dt

        # 3. Update streaks
        if new_state == self.current_state:
            self.current_streak_time += dt
            # If we are currently in bad state, update max_bad_streak immediately 
            # so it reflects current ongoing streak
            if new_state == "bad":
                self.max_bad_streak = max(self.max_bad_streak, self.current_streak_time)
        else:
            # State changed
            # If leaving bad state, we might want to finalize max_bad_streak logic, 
            # but we already update it continuously above.
            
            # If entering bad state, increment episode count
            if new_state == "bad":
                self.bad_episodes_count += 1
            
            self.current_state = new_state
            self.current_streak_time = dt
            
            # Initialize max_bad_streak for this new bad episode if applicable
            if new_state == "bad":
                self.max_bad_streak = max(self.max_bad_streak, self.current_streak_time)

    def snapshot(self):
        """
        Return a summary dictionary of the current history statistics.
        """
        good_ratio = 0.0
        bad_ratio = 0.0
        if self.total_time > 0:
            good_ratio = self.good_time / self.total_time
            bad_ratio = self.bad_time / self.total_time

        return {
            "total_time": self.total_time,
            "good_time": self.good_time,
            "warning_time": self.warning_time,
            "bad_time": self.bad_time,
            "good_ratio": good_ratio,
            "bad_ratio": bad_ratio,
            "current_state": self.current_state,
            "current_streak_time": self.current_streak_time,
            "max_bad_streak": self.max_bad_streak,
            "bad_episodes_count": self.bad_episodes_count
        }
