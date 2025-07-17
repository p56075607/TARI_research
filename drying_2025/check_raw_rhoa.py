# check_raw_rhoa.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 設定中文字體
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 讀取時間窗口資料
df_window = pd.read_csv('E1_window.csv')
df_window['x'] = pd.to_datetime(df_window['x'])

# 將時間窗口往前往後各延長24小時
# 偶數索引（最濕潤時段）往前延長24小時，奇數索引（最乾燥時段）往後延長24小時
for i in range(len(df_window)):
    if i % 2 == 0:  # 偶數索引，最濕潤時段往前延長24小時
        df_window.loc[i, 'x'] = df_window.loc[i, 'x'] - timedelta(hours=24)
    else:  # 奇數索引，最乾燥時段往後延長24小時
        df_window.loc[i, 'x'] = df_window.loc[i, 'x'] + timedelta(hours=24)

# 顯示延長後的時間窗口
print("=== 時間窗口延長結果 ===")
print("原始時間窗口往前往後各延長24小時")
for i in range(len(df_window)):
    window_type = "最濕潤時段" if i % 2 == 0 else "最乾燥時段"
    extension = "往前延長24小時" if i % 2 == 0 else "往後延長24小時"
    print(f"{window_type} {i//2 + 1}: {df_window.loc[i, 'x'].strftime('%Y/%m/%d %H:%M')} ({extension})")
print()

# 讀取 alpha_one_rhoa 資料
df_alpha_one = pd.read_csv('alpha_one_rhoa_data.csv', index_col=0)
df_alpha_one.index = pd.to_datetime(df_alpha_one.index)

# 選定要分析的欄位
selected_columns = ['alpha_one_RHOA_1', 'alpha_one_RHOA_2', 'alpha_one_RHOA_3', 'alpha_one_RHOA_4', 
                   'alpha_one_RHOA_7', 'alpha_one_RHOA_10', 'alpha_one_RHOA_11', 'alpha_one_RHOA_12', 
                   'alpha_one_RHOA_13', 'alpha_one_RHOA_14']

# 定義顏色和標記
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

print("=== 乾燥期間分析 ===")
print(f"總共有 {len(df_window)} 個時間點，{len(df_window)//2} 個乾燥期間")
print(f"選定的欄位: {selected_columns}")
print()

# 分析每個乾燥期間
for i in range(0, len(df_window)-1, 2):
    period_num = i//2 + 1
    start_time = df_window.loc[i, 'x']
    end_time = df_window.loc[i+1, 'x']
    
    print(f"=== 第 {period_num} 個乾燥期間 (延長時間窗口) ===")
    print(f"延長後開始時間: {start_time.strftime('%Y/%m/%d %H:%M')}")
    print(f"延長後結束時間: {end_time.strftime('%Y/%m/%d %H:%M')}")
    print(f"延長後期間長度: {(end_time - start_time).days} 天 {(end_time - start_time).seconds // 3600} 小時")
    
    # 篩選該時間段的資料
    mask = (df_alpha_one.index >= start_time) & (df_alpha_one.index <= end_time)
    period_data = df_alpha_one.loc[mask, selected_columns].copy()
    
    if period_data.empty:
        print("該時間段沒有資料，跳過...")
        continue
    
    print(f"該時間段資料點數: {len(period_data)}")
    
    # 建立圖表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 繪製每個選定的欄位
    for j, col in enumerate(selected_columns):
        if col in period_data.columns:
            valid_data = period_data[col].dropna()
            
            if len(valid_data) > 0:
                ax.plot(valid_data.index, valid_data.values,
                       marker=markers[j], 
                       color=colors[j],
                       linestyle=line_styles[j],
                       linewidth=2,
                       markersize=5,
                       alpha=0.8,
                       label=f'{col} (n={len(valid_data)})')
                
                print(f"  {col}: {len(valid_data)} 個資料點, 範圍: {valid_data.min():.1f} - {valid_data.max():.1f} Ω·m")
    
    # 設定圖表樣式
    ax.set_xlabel('時間', fontsize=12, fontweight='bold')
    ax.set_ylabel('視電阻率 (Ω·m)', fontsize=12, fontweight='bold')
    ax.set_title(f'第 {period_num} 個乾燥期間 (延長時間窗口: {start_time.strftime("%Y/%m/%d %H:%M")} - {end_time.strftime("%Y/%m/%d %H:%M")})',
                fontsize=14, fontweight='bold')
    
    # 設定 y 軸為對數尺度
    ax.set_yscale('log')
    ax.set_ylim([10, 500])
    
    # 設定刻度
    ax.tick_params(axis='both', which='major', length=5, width=1.5, direction='in')
    
    # 設定網格
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.grid(True, which='major', linestyle='-', linewidth=1, alpha=0.5)
    
    # 設定 x 軸時間格式
    if (end_time - start_time).days > 30:
        # 超過30天的期間，使用週為單位
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    else:
        # 30天以內的期間，使用天為單位
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    plt.xticks(rotation=45)
    
    # 添加圖例
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    for text in legend.get_texts():
        text.set_weight('bold')
    
    # 標記最濕潤和最乾燥時段
    ax.axvline(x=start_time, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='最濕潤時段')
    ax.axvline(x=end_time, color='red', linestyle='--', alpha=0.7, linewidth=2, label='最乾燥時段')
    
    # 添加時間段資訊文字框
    info_text = f'延長時間窗口: {(end_time - start_time).days} 天\n實際延長了48小時\n資料點數: {len(period_data)}\n有效欄位: {sum(1 for col in selected_columns if not period_data[col].dropna().empty)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 調整佈局
    plt.tight_layout()
    plt.show()
    
    # 計算該時間段的統計資訊
    print("\n統計資訊:")
    for col in selected_columns:
        if col in period_data.columns:
            valid_data = period_data[col].dropna()
            if len(valid_data) > 0:
                # 計算最濕潤和最乾燥時的數值
                start_value = period_data[col].iloc[0] if not pd.isna(period_data[col].iloc[0]) else "N/A"
                end_value = period_data[col].iloc[-1] if not pd.isna(period_data[col].iloc[-1]) else "N/A"
                
                if start_value != "N/A" and end_value != "N/A":
                    change_percent = ((end_value - start_value) / start_value) * 100
                    print(f"  {col}: 起始值 {start_value:.1f} → 結束值 {end_value:.1f} Ω·m (變化: {change_percent:+.1f}%)")
                else:
                    print(f"  {col}: 起始值 {start_value} → 結束值 {end_value} Ω·m")
    
    print("\n" + "="*50 + "\n")

# 整體統計分析
print("=== 整體統計分析 ===")
all_periods_stats = []

for i in range(0, len(df_window)-1, 2):
    period_num = i//2 + 1
    start_time = df_window.loc[i, 'x']
    end_time = df_window.loc[i+1, 'x']
    
    mask = (df_alpha_one.index >= start_time) & (df_alpha_one.index <= end_time)
    period_data = df_alpha_one.loc[mask, selected_columns].copy()
    
    if not period_data.empty:
        period_stats = {
            'period': period_num,
            'start_time': start_time,
            'end_time': end_time,
            'duration_days': (end_time - start_time).days,
            'data_points': len(period_data)
        }
        
        # 計算每個欄位的變化
        for col in selected_columns:
            valid_data = period_data[col].dropna()
            if len(valid_data) >= 2:
                start_val = valid_data.iloc[0]
                end_val = valid_data.iloc[-1]
                change_percent = ((end_val - start_val) / start_val) * 100
                period_stats[f'{col}_change_percent'] = change_percent
            else:
                period_stats[f'{col}_change_percent'] = np.nan
        
        all_periods_stats.append(period_stats)

# 將統計結果轉換為 DataFrame 並儲存
df_stats = pd.DataFrame(all_periods_stats)
df_stats.to_csv('drying_periods_analysis.csv', index=False)

print(f"分析完成！共處理了 {len(all_periods_stats)} 個乾燥期間")
print("詳細統計資料已儲存至 drying_periods_analysis.csv")
