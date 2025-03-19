# %%
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
file_keys = ['E1']#, 'E2', 'E3']
dataset_results = {}  # 用來存放各資料集處理後的結果與擬合參數

for key in file_keys:
    try:
        df_window, df_RHOA = load_data('Refit_config.yaml', key)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        continue

    # 過濾 median_RHOA > 0
    df_RHOA = df_RHOA[df_RHOA['median_RHOA'] > 0]
     # 先轉換時間格式
    df_RHOA['date'] = pd.to_datetime(df_RHOA['date'])

    if key == 'E2':
        # filter data after 2024/12/1
        df_RHOA = df_RHOA[df_RHOA['date'] < '2024/11/28 00:00:00']
    
    if key == 'E1': 
        #24111114_m_E1
        df_RHOA = df_RHOA[df_RHOA['date'] <= '2024/11/14 00:00:00']

       

        # 然後再刪除 2024/7/5 整天的資料
        df_RHOA = df_RHOA[~((df_RHOA['date'] >= pd.Timestamp('2024-07-05')) & 
                            (df_RHOA['date'] < pd.Timestamp('2024-07-06')))]
    
    
    # 轉換時間格式
    df_window['x'] = pd.to_datetime(df_window['x'])
            
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
        
        results.append(df_filtered[['window_id', 'delay_hours', 'median_RHOA', 'Q1_RHOA', 'Q3_RHOA', 'start_rhoa','date']])
    
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
fig, ax = plt.subplots(figsize=(15, 3))
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
# sort data by delay_hours
data = dataset_results['E1']['data']
popt = dataset_results['E1']['fit_params']
data = data.sort_values(by='delay_hours')
t = np.linspace(0, max(data['delay_hours']), len(np.unique(data['delay_hours'])))
theoretical_data = func(t, *popt)

# 為每個唯一的delay_hours選擇最佳的median_RHOA值
delay_hours_unique = np.unique(data['delay_hours'])
selected_data = []

for dh in delay_hours_unique:
    # 篩選出當前delay_hours的所有數據
    subset = data[data['delay_hours'] == dh]
    
    # 如果只有一個值，直接選擇它
    if len(subset) == 1:
        selected_data.append(subset.iloc[0])
    else:
        # 找出理論值在當前delay_hours的索引位置
        t_idx = np.abs(t - dh).argmin()
        theory_value = theoretical_data[t_idx]
        
        # 計算每個實際值與理論值的差異
        subset['diff_from_theory'] = np.abs(subset['median_log_RHOA'] - theory_value)
        
        # 選擇最接近理論值的那一行
        best_row = subset.loc[subset['diff_from_theory'].idxmin()]
        selected_data.append(best_row)

# 將選出的數據轉換為DataFrame
filtered_data = pd.DataFrame(selected_data)

# plot filtered_data only plot even delay_hours
# filtered_data = filtered_data[filtered_data['delay_hours'] % 2 == 1]
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(filtered_data['delay_hours'], filtered_data['median_log_RHOA'], 'ko',markersize=3 ,zorder=2)
ax.plot(t, theoretical_data, '-', color='r',linewidth=2.5, linestyle=line_styles[key],
            label=f'y = {popt[0]:.2f} exp(-{popt[1]:.4f}x) + {popt[2]:.2f}\n$R^2$ = {R_squared:.2f}')
ax.legend()
ax.grid(True)
plt.show()
# %%
# 依序讀取 filtered_data 中的日期並組成 all_mgrs

# 設定輸出路徑
output_path = r'D:\R2MSDATA\TARI_E1_test\output_second_inversion'  # 請替換為您的實際輸出路徑
linename = 'E1'  # 使用的線名
all_mgrs = []
date_folder_names = []

def load_inversion_results(save_ph):
    output_ph = join(save_ph, 'ERTManager')
    para_domain = pg.load(join(output_ph, 'resistivity-pd.bms'))
    # 讀取數據文件
    data_path = join(output_ph, 'data.dat')
    data = ert.load(data_path)
    investg_depth = (max(pg.x(data)) - min(pg.x(data))) * 0.2
    # 讀取模型
    model = pg.load(join(output_ph, 'resistivity.vector'))
    coverage = pg.load(join(output_ph, 'resistivity-cov.vector'))

    # 讀取反演信息
    inv_info_path = join(output_ph, 'inv_info.txt')
    Line = []
    with open(inv_info_path, 'r') as read_obj:
        for i, line in enumerate(read_obj):
            Line.append(line.rstrip('\n'))

    final_result = Line[Line.index('## Final result ##') + 1:Line.index('## Inversion parameters ##')]
    rrms = float(final_result[0].split(':')[1])
    chi2 = float(final_result[1].split(':')[1])
    inversion_para = Line[Line.index('## Inversion parameters ##') + 1:Line.index('## Iteration ##')]
    lam = int(inversion_para[0].split(':')[1])
    iteration = Line[Line.index('## Iteration ##') + 2:]
    rrmsHistory = np.zeros(len(iteration))
    chi2History = np.zeros(len(iteration))
    for i in range(len(iteration)):
        rrmsHistory[i] = float(iteration[i].split(',')[1])
        chi2History[i] = float(iteration[i].split(',')[2])
    
    # 返回所需的結果
    return {'Name_date':date_str,
        'para_domain': para_domain,
        'data': data,
        'model': model,
        'coverage': coverage,
        'rrms': rrms,
        'chi2': chi2,
        'lam': lam,
        'rrmsHistory': rrmsHistory,
        'chi2History': chi2History
    }

# 依序處理 filtered_data 中的每一行
for i, row in filtered_data.iterrows():
    date_str = row['date'].strftime("%Y/%m/%d %H:00")
    rhoa_value = row['median_RHOA']
    
    print(f'處理數據點: x = {date_str}, y = {rhoa_value:.2f}')
    
    # 將日期格式化為文件夾名稱格式
    date_folder_name = date_str.replace('/', '').replace(':', '').replace(' ', '')[2:-2] + '_m_' + linename
    date_folder_names.append(date_folder_name)
    
    # 構建保存路徑
    save_ph = join(output_path, date_folder_name)
    
    # 檢查路徑是否存在
    if os.path.exists(save_ph):
        try:
            # 加載反演結果
            mgr = load_inversion_results(save_ph)
            all_mgrs.append(mgr)
            print(f"成功加載 {date_folder_name} 的反演結果")
        except Exception as e:
            print(f"加載 {date_folder_name} 時出錯: {str(e)}")
    else:
        print(f"警告: 路徑不存在 {save_ph}")

# %%
# 設定繪圖參數
left = min(pg.x(all_mgrs[0]['para_domain']))
right = max(pg.x(all_mgrs[0]['para_domain']))
depth = (right - left) * 0.2  # 調查深度通常為測線長度的20%

# 繪製電阻率差異對比圖
def plot_difference_contour(mgr1, mgr2, date_str1, date_str2, **kw_diff):
    model1 = mgr1['model']
    model2 = mgr2['model']
    mesh_x = np.linspace(left, right, 250)
    mesh_y = np.linspace(-depth, 0, 150)
    X, Y = np.meshgrid(mesh_x, mesh_y)
    grid = pg.createGrid(x=mesh_x, y=mesh_y)
    
    # 計算電阻率差異百分比
    one_line_diff = (np.log10(model2) - np.log10(model1)) / np.log10(model1) * 100
    diff_grid = np.reshape(pg.interpolate(mgr1['para_domain'], one_line_diff, grid.positions()), (len(mesh_y), len(mesh_x)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, diff_grid, cmap=kw_diff['cMap'], levels=20,
                vmin=kw_diff['cMin'], vmax=kw_diff['cMax'], antialiased=True)
    
    ax.set_aspect('equal')
    ax.set_xlim(left, right)
    ax.set_ylim(-depth, 0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.set_xlabel(kw_diff['xlabel'])
    ax.set_ylabel(kw_diff['ylabel'])
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.5)
    
    # 添加三角形遮罩
    triangle_left = np.array([[left, -depth], [depth, -depth], [left, 0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right, 0], [right, -depth]])
    ax.add_patch(plt.Polygon(triangle_left, color='white'))
    ax.add_patch(plt.Polygon(triangle_right, color='white'))
    
    # 添加顏色條
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="4.5%", pad=.1)
    m = plt.cm.ScalarMappable(cmap=kw_diff['cMap'])
    m.set_array(diff_grid)
    m.set_clim(kw_diff['cMin'], kw_diff['cMax'])
    cb = plt.colorbar(m, boundaries=np.linspace(kw_diff['cMin'], kw_diff['cMax'], 64), cax=cbaxes)
    cb.ax.set_yticks(np.linspace(kw_diff['cMin'], kw_diff['cMax'], 5))
    cb.ax.set_yticklabels(['{:.2f}'.format(x) for x in cb.ax.get_yticks()])
    cb.ax.set_ylabel(kw_diff['label'])
    
    # 添加標題
    ax.set_title(f'電阻率差異: {date_str2} vs {date_str1}', fontsize=14)
    
    return X,Y, diff_grid, fig

# 創建保存差異對比圖的資料夾
difference_output_path = 'difference_contour'
os.makedirs(difference_output_path, exist_ok=True)

# 設定繪圖參數
colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 1-(-3/-20), 1]  
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))     
kw = dict(cMin=-20, cMax=0,logScale=False,
            label='Relative resistivity \ndifference (%)',
            xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)  

# 以第一個時間點為參考，繪製後續時間點的差異對比圖
if len(all_mgrs) > 1:
    reference_mgr = all_mgrs[-1]
    reference_date = reference_mgr['Name_date']
    
    print(f"參考時間點: {reference_date}")
    every_diff_grid = []
    for i in range(0, len(all_mgrs)-1):
        current_mgr = all_mgrs[i]
        current_date = current_mgr['Name_date']
        
        print(f"繪製差異對比圖: {current_date} vs {reference_date}, median_RHOA = {filtered_data.iloc[i]['median_RHOA']:.2f}")
        
        try:
            X,Y, diff_grid, fig = plot_difference_contour(reference_mgr, current_mgr, 
                                         reference_date, current_date, **kw)
            
            # 保存圖片
            filename = f"{date_folder_names[i]}_vs_{date_folder_names[0]}_contour.png"
            fig.savefig(join(difference_output_path, filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
            every_diff_grid.append(diff_grid)

            print(f"已保存差異對比圖: {filename}")
        except Exception as e:
            print(f"繪製差異對比圖時出錯: {str(e)}")
else:
    print("all_mgrs 中的數據不足，無法繪製差異對比圖")

# %%
# 計算每個時間點的平均電阻率差異,沿著X軸方向
if len(every_diff_grid) > 0:
    # 準備存儲每個時間點沿Y軸的平均差異
    mean_along_x = np.zeros((len(Y[:,0]), len(every_diff_grid)))
    
    # 對每個時間點的差異網格計算沿X軸的平均值
    for i, diff_grid in enumerate(every_diff_grid):
        # 沿X軸方向計算平均值（axis=1是沿著X軸方向）
        mean_values = np.nanmean(diff_grid, axis=1)
        mean_along_x[:, i] = mean_values
    
    # 取得Y軸的深度值（深度為負值）
    depths = Y[:, 0]    
else:
    print("沒有可用的差異網格數據進行分析")
# %%
# Plot intensity map
# 設定繪圖參數
colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 1-(-1.5/-10), 1]  
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))     
kw = dict(cMin=-10, cMax=0,logScale=False,
            label='Relative resistivity \ndifference (%)',
            xlabel='Drying Time (hours)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)  

Tmesh,Ymesh = np.meshgrid(filtered_data['delay_hours'][:-1],Y[:,0])
fig, ax = plt.subplots(figsize=(8, 4))
cf = ax.contourf(Tmesh, Ymesh, mean_along_x, cmap=kw['cMap'], levels=25,
            vmin=kw['cMin'], vmax=kw['cMax'], antialiased=True, zorder=1)
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
title_font = 15
ax.set_xlabel(kw['xlabel'],fontweight='bold',fontsize=title_font)
ax.set_ylabel(kw['ylabel'],fontweight='bold',fontsize=title_font)
ax.set_ylim(-5, 0)
ax.grid('both',linestyle='--', linewidth=0.5,zorder=10)
plt.draw()

# 添加顏色條
cb = fig.colorbar(cf, pad=0.12)
cb.ax.set_ylim(kw['cMin'],kw['cMax'])
cb.ax.set_yticks(np.linspace(kw['cMin'], kw['cMax'], 11))
cb.ax.set_yticklabels(['{:.1f}'.format(x) for x in cb.ax.get_yticks()])
cb.ax.set_ylabel(kw['label'],fontweight='bold',fontsize=title_font)

ax2 = ax.twinx()
ax2.plot(filtered_data['delay_hours'][:-1], filtered_data['median_RHOA'][:-1], 'ko',markersize=3 ,zorder=2)
ax2.set_ylabel('Median Apparent Resistivity',fontweight='bold',fontsize=title_font)
ax2.plot(t,10**func(t,*popt),'-',color='r',linewidth=1.5, linestyle=line_styles[key],
            label=f'y = {popt[0]:.2f} exp(-{popt[1]:.4f}x) + {popt[2]:.2f}\n$R^2$ = {R_squared:.2f}')
ax2.set_yticks(np.linspace(ax2.get_ylim()[0],ax2.get_ylim()[1],6))
ax2.set_yticklabels(['{:.0f}'.format(x) for x in ax2.get_yticks()])
fig.savefig(join('intensity_map.png'), dpi=300, bbox_inches='tight')

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