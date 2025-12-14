import time
import matplotlib.pyplot as plt
from datetime import datetime

def generate_report(score_log, duration_minutes, away_periods=None):
    """程式結束時生成圖表
    
    Args:
        score_log: 分數記錄列表 [(time, score), ...]
        duration_minutes: 總時長(分鐘)
        away_periods: 離席時段列表 [(start_elapsed, end_elapsed), ...]
    """
    if not score_log:
        print("沒有足夠數據生成報告。")
        return

    times, scores = zip(*score_log)
    
    # 計算統計數據 (只計算實際在場時間的分數)
    avg_score = sum(scores) / len(scores)
    good_time = sum(1 for s in scores if s > 80)
    good_ratio = (good_time / len(scores)) * 100
    
    # 計算離席總時間
    total_away_time = 0
    if away_periods:
        total_away_time = sum(end - start for start, end in away_periods)

    plt.figure(figsize=(10, 6))
    
    # 繪製離席時段 (橙色半透明區域)
    if away_periods:
        for start, end in away_periods:
            plt.axvspan(start, end, alpha=0.3, color='orange', label='_nolegend_')
        # 添加一個用於圖例的代表性區域
        plt.axvspan(0, 0, alpha=0.3, color='orange', label='User Away')
    
    plt.plot(times, scores, label='Posture Score', color='blue', linewidth=1.5)
    plt.axhline(y=80, color='g', linestyle='--', label='Excellent (80)')
    plt.axhline(y=60, color='r', linestyle='--', label='Poor (60)')
    
    plt.title(f"Posture Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score")
    plt.ylim(0, 105)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.25))
    plt.grid(True, alpha=0.3)

    # 加入文字統計
    away_time_str = f"\nAway Time: {total_away_time:.1f}s" if total_away_time > 0 else ""
    info_text = (f"Duration: {duration_minutes:.1f} mins\n"
                 f"Avg Score: {avg_score:.1f}\n"
                 f"Good Posture: {good_ratio:.1f}%{away_time_str}")
    plt.gcf().text(0.3, 0.12, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))

    filename = f"posture_report_{int(time.time())}.png"
    plt.savefig(filename)
    print(f"健康報告已儲存為: {filename}")
    # 如果想直接顯示，把下面這行取消註解
    plt.show()
