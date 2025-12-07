import threading
import pyttsx3
import time

class VoiceAssistant:
    """非阻塞版，每次播放新 engine，播放完後釋放記憶體"""
    def __init__(self):
        self.lock = threading.Lock()  # 防止多執行緒同時播放時出錯
        self.last_reset_time = 0      # 記錄最後一次重置的時間

    def say(self, text):
        if not text:
            return
        # 用背景執行緒播放，傳入當前的生成時間
        current_req_time = time.time()
        threading.Thread(target=self._play, args=(text, current_req_time), daemon=True).start()

    def stop(self):
        """取消所有尚未播放的語音指令"""
        self.last_reset_time = time.time()

    def _play(self, text, req_time):
        # 如果這個請求是在最後一次重置之前發出的，就忽略它
        if req_time < self.last_reset_time:
            return

        with self.lock:
            # 再次檢查，因為排隊時可能又發生了重置
            if req_time < self.last_reset_time:
                return
                
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print("語音播放失敗:", e)
