# %% ----------------------------------------------------------------------------------------------------
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import os
from os.path import join
import pygimli as pg
from pygimli.physics import ert  # the module
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %%
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
        window_path = r'C:\Users\Git\TARI_research\drying_2025\E1_window.csv'#config[file_key]['window']
        rhoa_path = r'C:\Users\Git\TARI_research\drying_2025\alpha_one_rhoa_data.csv'#config[file_key]['rhoa']
        
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

def func(x, a, b):
    # Linear function: a + b * x
    return a + b * x

# 定義要處理的資料集鍵值
file_keys = ['E1']
dataset_results = {}  # 用來存放各資料集處理後的結果與擬合參數

# 定義 14 個 alpha_one_RHOA 欄位
alpha_columns = [f'alpha_one_RHOA_{i+1}' for i in range(14)]

for key in file_keys:
    try:
        df_window, df_RHOA = load_data(r'C:\Users\Git\TARI_research\picking\Refit_config.yaml', key)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        continue

    # 轉換時間格式
    df_RHOA['datetime'] = pd.to_datetime(df_RHOA['datetime'])
    df_window['x'] = pd.to_datetime(df_window['x'])
    
    # 儲存所有 alpha_one_RHOA 欄位的結果
    all_alpha_results = {}
    
    # 對每個 alpha_one_RHOA 欄位分別處理
    for alpha_col in alpha_columns:
        print(f"處理 {alpha_col}...")
        
        # 過濾出有效資料（非 NaN 且 > 0）
        df_RHOA_filtered = df_RHOA.dropna(subset=[alpha_col])
        df_RHOA_filtered = df_RHOA_filtered[df_RHOA_filtered[alpha_col] > 0]
        
        if df_RHOA_filtered.empty:
            print(f"{alpha_col} 沒有有效資料，跳過...")
            continue
            
        # 重新命名欄位以便後續處理
        df_RHOA_temp = df_RHOA_filtered.copy()
        df_RHOA_temp['date'] = df_RHOA_temp['datetime']
        df_RHOA_temp['median_RHOA'] = df_RHOA_temp[alpha_col]

        if key == 'E2':
            # filter data after 2024/12/1
            df_RHOA_temp = df_RHOA_temp[df_RHOA_temp['date'] < '2024/11/28 00:00:00']
        
        if key == 'E1': 
            #24111114_m_E1
            df_RHOA_temp = df_RHOA_temp[df_RHOA_temp['date'] <= '2024/11/14 00:00:00']
            
            # 然後再刪除 2024/7/5 整天的資料
            df_RHOA_temp = df_RHOA_temp[~((df_RHOA_temp['date'] >= pd.Timestamp('2024-07-05')) & 
                                (df_RHOA_temp['date'] < pd.Timestamp('2024-07-06')))]
        
        results = []
        # 每兩列定義一個時間窗口，遍歷 df_window
        for i in range(0, len(df_window) - 1, 2):
            start_time = df_window.loc[i, 'x']
            end_time = df_window.loc[i + 1, 'x']
            
            # 篩選出在此時間窗口內的資料
            mask = (df_RHOA_temp['date'] >= start_time) & (df_RHOA_temp['date'] <= end_time)
            df_filtered = df_RHOA_temp.loc[mask].copy()
            if df_filtered.empty:
                continue
            
            # 計算延時（以小時為單位）
            df_filtered['delay_hours'] = (df_filtered['date'] - start_time).dt.total_seconds() / 3600
            
            # 添加起始電阻率值與窗口編號
            df_filtered['start_rhoa'] = df_filtered['median_RHOA'].iloc[0]
            df_filtered['window_id'] = i // 2
            
            results.append(df_filtered[['window_id', 'delay_hours', 'median_RHOA', 
                                        'start_rhoa','date']])
        
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
            print(f"曲線擬合失敗於 {alpha_col}: {e}")
            popt = [np.nan, np.nan]
        
        if key == 'E1':
            # delete dataset_results[key][data][median_log_RHOA] > 2.15
            cumulative_data = cumulative_data[cumulative_data['median_log_RHOA'] < 2.15]
            # cumulative_data = cumulative_data[cumulative_data['median_log_RHOA'] > 1 ]
            
        # 儲存結果
        all_alpha_results[alpha_col] = {
            'data': cumulative_data[['delay_hours', 'median_RHOA']].copy(),
            'fit_params': popt
        }
        
        print(f"{alpha_col} 處理完成，資料點數: {len(cumulative_data)}")
# %%
# 合併所有結果成一個 DataFrame

# === 改成「每個 alpha_one_RHOA 各自輸出獨立的 DataFrame」 ===
if not all_alpha_results:
    print("沒有有效的 alpha_one_RHOA 資料可處理")
else:
    output_dir = "drying_2025/alpha_one_by_column"  # 輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    for alpha_col, result in all_alpha_results.items():
        df_out = result['data'].copy().reset_index(drop=True)
        out_path = os.path.join(output_dir, f"dryingtime_{alpha_col}_E1.csv")
        df_out.to_csv(out_path, index=False)
        print(f"{alpha_col} 已輸出 {len(df_out)} 筆資料 ➜ {out_path}")
