# %%
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.dates as mdates

def load_data(conf_file, file_key):
    """
    Load window and RHOA data files based on the specified file key.

    Parameters:
    conf_file (str): The path to the YAML configuration file.
    file_key (str): The file key, e.g., 'E1', 'E2', 'E3'.

    Returns:
    tuple: Two DataFrames for window and RHOA data.
    """
    with open(conf_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    if file_key in config:
        window_path = config[file_key]['window']
        rhoa_path = config[file_key]['rhoa']
        
        # Check if the files exist
        if not os.path.exists(window_path):
            raise FileNotFoundError(f"檔案不存在: {window_path}")
        if not os.path.exists(rhoa_path):
            raise FileNotFoundError(f"檔案不存在: {rhoa_path}")
        
        df_window = pd.read_csv(window_path)
        df_rhoa = pd.read_csv(rhoa_path)
        return df_window, df_rhoa
    else:
        raise ValueError(f"未找到鍵值 '{file_key}' 對應的檔案配置。")

def func(x, a, b, c):
    # Exponential decay function: a * exp(-b * x) + c
    return a * np.exp(-b * x) + c

# 定義要處理的資料集鍵值
file_keys = ['E1', 'E2', 'E3']
dataset_results = {}  # 用來存放各資料集處理後的結果與擬合參數

for key in file_keys:
    try:
        df_window, df_RHOA = load_data('Refit_config.yaml', key)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        continue

    # 過濾 median_RHOA > 0
    df_RHOA = df_RHOA[df_RHOA['median_RHOA'] > 0]
    if key == 'E2':
        # filter data after 2024/12/1
        df_RHOA = df_RHOA[df_RHOA['date'] < '2024/11/28 00:00:00']
    
    # 轉換時間格式
    df_window['x'] = pd.to_datetime(df_window['x'])
    df_RHOA['date'] = pd.to_datetime(df_RHOA['date'])
    
    results = []
    # 每兩列定義一個時間窗口，遍歷 df_window
    for i in range(0, len(df_window) - 1, 2):
        start_time = df_window.loc[i, 'x']
        end_time = df_window.loc[i + 1, 'x']
        
        # 篩選出在此時間窗口內的資料
        mask = (df_RHOA['date'] >= start_time) & (df_RHOA['date'] <= end_time)
        df_filtered = df_RHOA.loc[mask].copy()
        if df_filtered.empty:
            continue
        
        # 計算延時（以小時為單位）
        df_filtered['delay_hours'] = (df_filtered['date'] - start_time).dt.total_seconds() / 3600
        
        # 添加起始電阻率值與窗口編號
        df_filtered['start_rhoa'] = df_filtered['median_RHOA'].iloc[0]
        df_filtered['window_id'] = i // 2
        
        results.append(df_filtered[['window_id', 'delay_hours', 'median_RHOA', 'Q1_RHOA', 'Q3_RHOA', 'start_rhoa']])
    
    if len(results) == 0:
        continue
    
    # 合併各窗口的資料
    df_all = pd.concat(results, ignore_index=True)
    # 根據窗口起始電阻率排序
    window_order = df_all.groupby('window_id')['start_rhoa'].first().sort_values().index
    
    cumulative_data = pd.DataFrame()
    # 調整每個窗口的延時，使得時間軸連續
    for window_id in window_order:
        window_data = df_all[df_all['window_id'] == window_id]
        start_rhoa = window_data['start_rhoa'].iloc[0]
        if not cumulative_data.empty:
            # 找到累積資料中與當前窗口起始電阻率最接近的值
            closest_idx = (cumulative_data['median_RHOA'] - start_rhoa).abs().idxmin()
            time_offset = cumulative_data.loc[closest_idx, 'delay_hours']
            adjusted_time = window_data['delay_hours'] + time_offset
        else:
            adjusted_time = window_data['delay_hours']
        new_window_data = window_data.copy()
        new_window_data['delay_hours'] = adjusted_time
        cumulative_data = pd.concat([cumulative_data, new_window_data], ignore_index=True)
    
    # 計算對數後的電阻率
    cumulative_data['median_log_RHOA'] = np.log10(cumulative_data['median_RHOA'])
    
    # 單獨對每個資料集做曲線擬合
    x_data = cumulative_data['delay_hours']
    y_data = cumulative_data['median_log_RHOA']
    try:
        popt, pcov = curve_fit(func, x_data, y_data)
    except RuntimeError as e:
        print(f"曲線擬合失敗於 {key}: {e}")
        popt = [np.nan, np.nan, np.nan]
    
    dataset_results[key] = {'data': cumulative_data, 'fit_params': popt}
    if key == 'E1':
        # delete dataset_results[key][data][median_log_RHOA] > 2.15
        dataset_results[key]['data'] = dataset_results[key]['data'][dataset_results[key]['data']['median_log_RHOA'] < 2.15]
    

# 繪圖：每個資料集的觀測資料與其擬合線都會以不同 marker 與顏色呈現
fig, ax = plt.subplots(figsize=(8, 6))
markers = {'E1': 'o', 'E2': 'o', 'E3': 'o'}
colors = {'E1': 'k', 'E2': 'green', 'E3': 'b'}
line_styles = {'E1': '-', 'E2': '--', 'E3': ':'}
for key in dataset_results:
    data = dataset_results[key]['data']
    popt = dataset_results[key]['fit_params']
    
    # 繪製觀測資料散點圖
    ax.plot(data['delay_hours'], data['median_log_RHOA'], markers[key], markersize=3, color=colors[key], label=f'{key} 觀測資料')
    
    # 計算 R^2 擬合度
    R_squared = 1 - (np.sum((data['median_log_RHOA'] - func(data['delay_hours'], *popt))**2) / np.sum((data['median_log_RHOA'] - np.mean(data['median_log_RHOA']))**2))

    # 計算擬合曲線並繪製
    t = np.linspace(0, max(data['delay_hours']), 100)
    ax.plot(t, func(t, *popt), '-', color='r',linewidth=2.5, linestyle=line_styles[key],
            label=f'y = {popt[0]:.2f} exp(-{popt[1]:.4f}x) + {popt[2]:.2f}\n$R^2$ = {R_squared:.2f}')
    
    


ax.set_xlabel('Drying Time (hours)')
ax.set_ylabel('Log10(Aparrent Resistivity)')
# ax.set_title('E1、E2、E3 各自的擬合結果')
ax.legend()
ax.grid(True)
plt.show()

# %%
# 計算並繪製各資料集的微分 (Derivative)
fig, ax = plt.subplots(figsize=(8, 6))
markers = {'E1': 'o', 'E2': 's', 'E3': '^'}  # Marker for each dataset
colors = {'E1': 'k', 'E2': 'green', 'E3': 'b'}

for key in ['E1', 'E2', 'E3']:
    # Retrieve processed data and fit parameters for the current dataset
    data = dataset_results[key]['data']       # Processed data DataFrame
    popt = dataset_results[key]['fit_params']   # Fit parameters from curve_fit
    
    # Generate time vector (t) based on the delay_hours range of the current dataset
    t = np.linspace(0, data['delay_hours'].max(), 100)
    
    # Compute the fitted curve values
    fitted_vals = func(t, *popt)
    
    # Calculate derivative using np.diff (Note: diff output length is one less than t)
    dt = np.diff(t)             # Time differences
    dfitted = np.diff(fitted_vals)  # Difference of fitted values
    slope = dfitted / dt        # Numerical derivative
    
    # Plot derivative:
    # x-axis: fitted function values (excluding the last point to match slope length)
    # y-axis: calculated slope
    ax.plot(fitted_vals[:-1], slope, '-' + markers[key], markersize=3, 
            color=colors[key], label=f'{key} 微分')
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax.set_xlabel('Log Apparent Resistivity log(ohm-m)')
ax.set_ylabel('Slope of the drying curve')
ax.legend()
ax.grid(True)
plt.show()



































# ---------------------------------------------------------------------------------------------------------------

# %%

df_window, df_RHOA = load_data('Refit_config.yaml', 'E2')
df_RHOA = df_RHOA[df_RHOA['median_RHOA'] > 10**0.5]
df_RHOA['date'] = pd.to_datetime(df_RHOA['date'])
# filter some dates
# df_RHOA = df_RHOA[df_RHOA['date'] > '2024/03/01 00:00:00']
# df_RHOA = df_RHOA[(df_RHOA['date'] < '2024/07/05 16:00') | (df_RHOA['date'] > '2024/07/08 11:00')]
# df_RHOA = df_RHOA[df_RHOA['date'] < '2024/11/21 16:00:00']

df_RHOA = df_RHOA[df_RHOA['date'] < '2024/11/28 00:00:00']

# df_RHOA = df_RHOA[df_RHOA['date'] > '2024/03/10 00:00:00']

plt.rcParams['font.family'] = 'Microsoft YaHei'
# plot the picking slope histogram
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df_RHOA['date'], np.log10(df_RHOA['median_RHOA']), 'ro',markersize=3 ,zorder=2)
fontsize = 20
ax.set_ylabel('視電阻率'+'\n'+r'$log(\rho_a)$', fontsize=fontsize+5,fontweight='bold')
fz_minor = 25
plt.yticks(fontsize=fz_minor,fontweight='bold')
plt.xticks(fontsize=fz_minor,rotation=45, ha='right', rotation_mode='anchor',fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax.yaxis.get_offset_text().set_fontsize(fz_minor)


ax.grid(True, which='major', linestyle='--', linewidth=0.5)


ax.grid(True, which='major', linestyle='--', linewidth=0.5)
# set xy ticks label fontsize 
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
ax.set_xlabel('Time (2024/mm)', fontsize=fz_minor, fontweight='bold')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
ax.grid(True, which='major', linestyle='-', linewidth=1)
plt.yticks(fontsize=fz_minor,fontweight='bold')