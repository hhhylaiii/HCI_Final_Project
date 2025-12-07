import time
import matplotlib.pyplot as plt
from datetime import datetime

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
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.2))
    plt.grid(True, alpha=0.3)

    # 加入文字統計
    info_text = (f"Duration: {duration_minutes:.1f} mins\n"
                 f"Avg Score: {avg_score:.1f}\n"
                 f"Good Posture: {good_ratio:.1f}%")
    plt.gcf().text(0.3, 0.15, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))

    filename = f"posture_report_{int(time.time())}.png"
    plt.savefig(filename)
    print(f"健康報告已儲存為: {filename}")
    # 如果想直接顯示，把下面這行取消註解
    plt.show()
