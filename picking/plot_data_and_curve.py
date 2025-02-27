# %%
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import numpy as np

# 讀取資料
# df_window = pd.read_csv(join('RHOA_data', '竹塘水田', 'E3_window.csv'))
# df_RHOA = pd.read_csv(join('RHOA_data', '竹塘水田', 'E3_rhoa.csv'))
# df_window = pd.read_csv(join('RHOA_data', '農試所水田', 'E1_window.csv'))
# df_RHOA = pd.read_csv(join('RHOA_data', '農試所水田', 'E1_rhoa.csv'))
df_window = pd.read_csv(join('RHOA_data', '農試所旱田', 'E2_window.csv'))
df_RHOA = pd.read_csv(join('RHOA_data', '農試所旱田', 'E2_rhoa.csv'))
df_RHOA = df_RHOA[df_RHOA['median_RHOA']>0]
# 將時間欄位轉換為 datetime 格式
df_window['x'] = pd.to_datetime(df_window['x'])
df_RHOA['date'] = pd.to_datetime(df_RHOA['date'])

# 初始化結果列表
results = []

# 遍歷 df_window 的索引，步長為2
for i in range(0, len(df_window) - 1, 2):
    start_time = df_window.loc[i, 'x']
    end_time = df_window.loc[i + 1, 'x']
    
    # 篩選 df_RHOA 中在當前時間窗口內的資料
    mask = (df_RHOA['date'] >= start_time) & (df_RHOA['date'] <= end_time)
    df_filtered = df_RHOA.loc[mask].copy()
    
    # 計算延時（以小時為單位）
    df_filtered['delay_hours'] = (df_filtered['date'] - start_time).dt.total_seconds() / 3600
    
    # 添加起始電阻率值
    start_rhoa = df_filtered['median_RHOA'].iloc[0]
    df_filtered['start_rhoa'] = start_rhoa
    
    # 添加窗口編號
    df_filtered['window_id'] = i // 2  # 每個窗口的唯一編號
    
    # 添加當前窗口的結果到列表
    results.append(df_filtered[['window_id', 'delay_hours', 'median_RHOA','Q1_RHOA', 'Q3_RHOA', 'start_rhoa']])

# 將所有結果合併為一個 DataFrame
df_all = pd.concat(results, ignore_index=True)
# 按起始電阻率值對窗口進行排序
window_order = df_all.groupby('window_id')['start_rhoa'].first().sort_values().index

cumulative_data = pd.DataFrame()

time_offset = 0



for id, window_id in enumerate(window_order):
    window_data = df_all[df_all['window_id'] == window_id]
    start_rhoa = window_data['start_rhoa'].iloc[0]
    
    if not cumulative_data.empty:
        # 在累積資料中找到與當前窗口起始電阻率最接近的值
        closest_idx = (cumulative_data['median_RHOA'] - start_rhoa).abs().idxmin()
        time_offset = cumulative_data.loc[closest_idx, 'delay_hours']
        # 調整時間軸
        adjusted_time = window_data['delay_hours'] + time_offset
    else:
        adjusted_time = window_data['delay_hours']
    
    new_window_data = window_data.copy()
    new_window_data['delay_hours'] = adjusted_time
    
    # 更新累積資料
    cumulative_data = pd.concat([cumulative_data, new_window_data], ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 6))
# plot error bar between QCV
QCV = (cumulative_data['Q3_RHOA']- cumulative_data['Q1_RHOA'])/cumulative_data['median_RHOA']
ax.errorbar(cumulative_data['delay_hours'], cumulative_data['median_RHOA'], 
            yerr=QCV,
                  fmt='none', ecolor='b',alpha=1, capsize=3, label='QCV')
ax.plot(cumulative_data['delay_hours'], cumulative_data['median_RHOA'], 'ko', markersize=3, label='觀測資料')
t = np.linspace(0, max(cumulative_data['delay_hours']), 100)
# theo_curve = (0.0003859846972252487 * np.exp(-0.0025465262282503166 * t) - 0.0039892514401080985) / -0.0025465262282503166
# theo_curve = ((0.0005008284757849254) * np.exp(-0.0021842301349774863 * t) - 0.004869288745739898) / -0.0021842301349774863
theo_curve = ((0.0002808284757849254) * np.exp(-0.0007037736082029306 * t) - 0.0018939996426083727) / -0.0007037736082029306
ax.plot(t, 10**theo_curve, 'r-', label='理論曲線')
ax.set_xlabel('Drying Time (hours)')
ax.set_ylabel('Apparent Resistivity (ohm-m)')
ax.legend()
ax.grid(True)
