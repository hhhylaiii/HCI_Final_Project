import threading
import pyttsx3

class VoiceAssistant:
    """非阻塞版，每次播放新 engine，播放完後釋放記憶體"""
    def __init__(self):
        self.lock = threading.Lock()  # 防止多執行緒同時播放時出錯

    def say(self, text):
        if not text:
            return
        # 用背景執行緒播放
        threading.Thread(target=self._play, args=(text,), daemon=True).start()

    def _play(self, text):
        with self.lock:
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print("語音播放失敗:", e)
