# 🧘 Smart Posture Assistant

一個基於 MediaPipe 的智慧坐姿監測系統，透過電腦視覺即時偵測您的坐姿並提供語音提醒，幫助您養成良好的工作習慣。

## ✨ 功能特色

- **📐 即時姿勢評分** - 透過 MediaPipe Pose 分析您的坐姿，計算姿勢分數
- **🔊 語音提醒** - 當姿勢不佳時，系統會語音提醒您調整坐姿
- **⏰ 番茄鐘計時器** - 內建久坐提醒功能，提醒您適時起身活動
- **🎨 背景模糊 (隱私模式)** - 支援背景模糊，保護您的隱私
- **👤 離席偵測** - 自動偵測用戶離開，暫停計時器
- **📊 健康報告** - 程式結束時自動生成姿勢分析圖表報告
- **📷 相機校正** - 支援張氏標定法校正相機畸變

## 📋 系統需求

- Python 3.8+
- 網路攝影機 (Webcam)
- Windows / macOS / Linux

## 🚀 快速開始

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 執行程式

**Windows:**
```bash
run.bat
```

**macOS / Linux:**
```bash
chmod +x run.sh
./run.sh
```

或手動執行：
```bash
# Step 1: 相機校正 (可選，用於修正鏡頭畸變)
python Codes/calibration.py

# Step 2: 啟動主程式
python Codes/main.py
```

## 🎮 操作說明

### 相機校正 (`calibration.py`)

1. 準備一個 **9x6 棋盤格** (可從網路下載列印)
2. 按 `s` 拍攝校正樣本 (建議至少 10 張)
3. 按 `q` 結束拍攝並計算校正參數
4. 校正完成後會產生 `camera_params.npz` 檔案

> **注意**：相機校正為可選步驟，若無棋盤格可跳過

### 主程式 (`main.py`)

啟動後會自動進行姿勢校正，請 **保持正確坐姿** 約 3 秒鐘等待校正完成。

| 按鍵 | 功能 |
|------|------|
| `b` | 切換背景模糊 (隱私模式) |
| `r` | 重置久坐計時器 (限時間到時) |
| `q` | 結束程式並生成報告 |

## 📁 專案結構

```
HCI_Final_Project/
├── Codes/
│   ├── main.py              # 主程式入口
│   ├── calibration.py       # 相機校正工具
│   ├── posture_score.py     # 姿勢評分模組
│   ├── posture_history.py   # 姿勢歷史記錄
│   ├── ui_painter.py        # UI 繪製模組
│   ├── voice_assistant.py   # 語音助手模組
│   └── report_generator.py  # 報告產生器
├── camera_params.npz        # 相機校正參數 (選用)
├── requirements.txt         # 依賴套件
├── run.bat                  # Windows 執行腳本
├── run.sh                   # Unix 執行腳本
└── README.md
```

## ⚙️ 可自訂參數

在 `Codes/main.py` 中可調整以下參數：

```python
W, H = 1280, 720              # 畫面解析度
POMODORO_LIMIT_SECONDS = 60   # 久坐提醒時間 (秒)
LOW_SCORE_THRESHOLD = 70      # 低於幾分開始警告
WARNING_COOLDOWN = 5.0        # 語音警告冷卻時間 (秒)
```

## 📊 輸出報告

程式結束時會自動生成：
- **姿勢分數折線圖** - 顯示整個使用期間的姿勢變化
- **離席時段標記** - 標記您離開座位的時間區間

## 🛠️ 依賴套件

| 套件 | 用途 |
|------|------|
| opencv-python | 影像處理與攝影機控制 |
| mediapipe | 人體姿勢偵測 |
| numpy | 數值運算 |
| pyttsx3 | 本地語音合成 |
| matplotlib | 報告圖表生成 |

## 📝 License

MIT License