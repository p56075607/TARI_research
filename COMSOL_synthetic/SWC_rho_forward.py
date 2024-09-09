# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import pygimli.meshtools as mt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'#'Microsoft Sans Serif'
import matplotlib
# matplotlib.use('TkAgg')  # 設置後端為 TkAgg
import pickle
import matplotlib.dates as mdates
from datetime import datetime
from os import listdir
from os.path import isdir, join
import mplcursors
from matplotlib.widgets import Cursor
import tkinter as tk
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %%
data_path = r'C:\Users\Git\TARI_research\COMSOL_synthetic\inverison_data.ohm'
data = ert.load(data_path)
left = min(pg.x(data))
right = max(pg.x(data))
length = right - left
depth = length/4
plc = mt.createParaMeshPLC(data,quality=34,paraDX=1/3,paraMaxCellSize=0.5
                            ,paraDepth=10)
mesh = mt.createMesh(plc)
# print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarker() == 2) if x]))
# mid,conf = ert.visualization.midconfERT(data)
# %%
def filter_data(data):
    ABMN = np.array((data['a'],data['b'],data['m'],data['n']))
    # 初始化結果陣列
    result = []

    # 迭代每一列
    for i in range(ABMN.shape[1]):
        A, B, M, N = ABMN[:, i]

        # 檢查AB是否包在MN外面
        if min(A, B) < min(M, N) and max(A, B) > max(M, N):
            # 檢查四個電極是否皆相鄰
            if max(A, B, M, N) - min(A, B, M, N) <= 10:
                result.append(i)

    boolean_array = np.ones(ABMN.shape[1], dtype=bool)
    boolean_array[result] = False

    data_filtered = data.copy()
    data_filtered.remove(boolean_array)

    return data_filtered

data_filtered = filter_data(data)
# %%
# 讀取CSV檔案
pkl_ph = r'C:\Users\Git\TARI_research\COMSOL_synthetic'
data_SWC = pd.read_csv(join(pkl_ph,'2m','WaterContent_Ks_E-5.csv'), skiprows=8)
data_SWC.columns = [col.strip('% ') for col in data_SWC.columns]

# 提取時間點列表
time_points = data_SWC.columns[2:]

# %%
def read_and_process_csv(file_path,data):
    pg.boxprint('Read and process CSV file: ' + file_path)
    data_SWC = pd.read_csv(file_path, skiprows=8, encoding='latin1')
    data_SWC.columns = [col.strip('% ') for col in data_SWC.columns]

    # 顯示前幾行資料，確認正確讀取
    print(data_SWC.head())

    # 提取時間點列表
    time_points = data_SWC.columns[2:]

    # 初始化一個字典來存儲不同時間點對應的數據
    time_data = {time: data_SWC[['X', 'Y', time]].rename(columns={time: 'theta'}) for time in time_points}


    median_rhoa = []
    for time, df in time_data.items():
        print(f"Data for {time}:")
        # print(df.head())
        grid_SWC = griddata((df[['X', 'Y']].to_numpy()), time_data[time]['theta'].to_numpy(), 
                            (np.array(mesh.cellCenters())[:, :2]), method='linear', fill_value=np.nan)

        fill_value = np.nanmedian(grid_SWC)
        grid_SWC = np.nan_to_num(grid_SWC, nan=fill_value)

        resistivity = np.zeros(mesh.cellCount())
        # check each point
        for i, point in enumerate(np.array(mesh.cellCenters())[:,:-1]):

            n = 1.83
            cFluid = 5/(0.57*106)
            resistivity[i] = -3.3*np.log(grid_SWC[i]*100)+27.456
            # resistivity[i] = -25*np.log(grid_SWC[i]*100)+120
        # ax,cb = pg.show(mesh, resistivity, cmap=pg.cmap('jet'), label='Resistivity (Ohm m)', showMesh=False)
        # ax.set_xlim(left, right)
        # ax.set_ylim(-10, 0)

        def show_simulation(data):
            pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
            pg.info('Simulated data', data)
            pg.info('The data contains:', data.dataMap().keys())
            pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
            pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)
        kw_noise = dict(noiseLevel=0.01, noiseAbs=0.001, seed=1337)
        data_sim = ert.simulate(mesh = mesh, res=resistivity,
                            scheme=data,
                            **kw_noise)
        show_simulation(data_sim)
        print(np.median(data_sim['rhoa']))
        median_rhoa.append(np.median(data_sim['rhoa']))

    return time_points, median_rhoa
output_path = r'D:\R2MSDATA\TARI_E3_test\output_second_timelapse_inveriosn'
process_data = True
if process_data:
    case_name = ['4m','05m','6m','8m']
    for i in range(len(case_name)):
        time_points, median_rhoa_6 = read_and_process_csv(file_path = join(pkl_ph,case_name[i],'WaterContent_Ks_E-6.csv'),data = data_filtered)
        time_points, median_rhoa_5 = read_and_process_csv(file_path = join(pkl_ph,case_name[i],'WaterContent_Ks_E-5.csv'),data = data_filtered)
        time_points, median_rhoa_7 = read_and_process_csv(file_path = join(pkl_ph,case_name[i],'WaterContent_Ks_E-7.csv'),data = data_filtered)

        with open(join(pkl_ph,case_name[i],'median_rhoa_6.pkl'), 'wb') as f:
            pickle.dump(median_rhoa_6, f)

        with open(join(pkl_ph,case_name[i],'median_rhoa_5.pkl'), 'wb') as f:
            pickle.dump(median_rhoa_5, f)

        with open(join(pkl_ph,case_name[i],'median_rhoa_7.pkl'), 'wb') as f:
            pickle.dump(median_rhoa_7, f)

    
    # output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]

    # median_rhoa_E3 = []
    # DATA = []
    # for i,output_folder_name in enumerate(output_folders):
    #     print(output_folder_name)
    #     data_path = join(output_path,output_folder_name,'ERTManager','inverison_data.ohm')
    #     data = ert.load(data_path)
    #     data_filtered = filter_data(data)
    #     DATA.append(data_filtered)
    #     median_rhoa_E3.append(np.median(data_filtered['rhoa']))

    # with open('median_rhoa_E3.pkl', 'wb') as f:
    #     pickle.dump(median_rhoa_E3, f)
# %%
def check_files_in_directory(directory_path):
    # 存儲解析出來的日期
    dates = []

    # 遍歷資料夾中的所有檔案
    for filename in sorted(listdir(directory_path)):
        # 檢查檔案名稱是否符合特定格式
        if filename.endswith('_m_E3'):
            date_str = filename[:8]  # 提取日期部分
            try:
                # 轉換日期格式從 'YYMMDDHH' 到 datetime 對象
                date = datetime.strptime(date_str, '%y%m%d%H')
                dates.append(date)
            except ValueError:
                # 如果日期格式不正確，忽略此檔案
                continue

    return dates

dates_E3 = check_files_in_directory(output_path)
# # read field data from pickle file

with open(join(pkl_ph,'median_RHOA_E3.pkl'), 'rb') as f:
    median_rhoa_E3 = pickle.load(f)

case_name = ['2m','4m','05m','6m','8m']
median_rhoa_dict = {}
for i in range(len(case_name)):
    with open(join(pkl_ph, case_name[i], 'median_rhoa_6.pkl'), 'rb') as f:
        median_rhoa_dict[case_name[i]+'_6'] = pickle.load(f)

    with open(join(pkl_ph, case_name[i], 'median_rhoa_5.pkl'), 'rb') as f:
        median_rhoa_dict[case_name[i]+'_5'] = pickle.load(f)

    with open(join(pkl_ph, case_name[i], 'median_rhoa_7.pkl'), 'rb') as f:
        median_rhoa_dict[case_name[i]+'_7'] = pickle.load(f)

print(median_rhoa_dict.keys())
# %%
time_values = [int(time.strip('t=h')) for time in time_points]
plt.close('all')
fig,ax = plt.subplots(figsize=(15, 6))

def plot_real_rhoa(begin_index,end_index):
    time_segment = dates_E3[begin_index:end_index]
    time_diffs_in_hours = [(time - time_segment[0]).total_seconds() / 3600 for time in time_segment]
    rhoa_diffs = [(rhoa - median_rhoa_E3[begin_index:end_index][0]) for rhoa in median_rhoa_E3[begin_index:end_index]]
    obs_gray = ax.plot(time_diffs_in_hours, rhoa_diffs, marker='.', linestyle='-',label='observed data: '+str(time_segment[0].date())+'~'+str(time_segment[-1].date()))
    return time_diffs_in_hours, rhoa_diffs
# # begin_index = dates_E3.index(datetime(2024, 3, 18, 13, 0))
# # end_index = dates_E3.index(datetime(2024, 3, 20, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 3, 21, 13, 0))
# # end_index = dates_E3.index(datetime(2024, 3, 22, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# outlier
# begin_index = dates_E3.index(datetime(2024, 3, 26, 11, 0))
# end_index = dates_E3.index(datetime(2024, 3, 27, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)
DIFF_time = []#np.zeros((len(np.linspace(0, 98, 50)), 17))
DIFF_rhoa = []#np.zeros((len(np.linspace(0, 98, 50)), 17))
Begin_End_IDX = []
begin_index = dates_E3.index(datetime(2024, 3, 27, 13, 0))
end_index = dates_E3.index(datetime(2024, 3, 28, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))

begin_index = dates_E3.index(datetime(2024, 3, 28, 13, 0))
end_index = dates_E3.index(datetime(2024, 3, 29, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))

begin_index = dates_E3.index(datetime(2024, 3, 29, 13, 0))
end_index = dates_E3.index(datetime(2024, 3, 30, 1, 0))
Begin_End_IDX.append((begin_index,end_index+1))

begin_index = dates_E3.index(datetime(2024, 3, 30, 15, 0))
end_index = dates_E3.index(datetime(2024, 4, 1, 1, 0))
Begin_End_IDX.append((begin_index,end_index+1))

begin_index = dates_E3.index(datetime(2024, 4, 1, 11, 0))
end_index = dates_E3.index(datetime(2024, 4, 4, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))

begin_index = dates_E3.index(datetime(2024, 4, 15, 13, 0))
end_index = dates_E3.index(datetime(2024, 4, 17, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 4, 18, 13, 0))
end_index = dates_E3.index(datetime(2024, 4, 22, 11, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 7, 15, 0))
end_index = dates_E3.index(datetime(2024, 5, 10, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 10, 13, 0))
end_index = dates_E3.index(datetime(2024, 5, 13, 11, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 13, 23, 0))
end_index = dates_E3.index(datetime(2024, 5, 16, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 16, 19, 0))
end_index = dates_E3.index(datetime(2024, 5, 19, 13, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 19, 15, 0))
end_index = dates_E3.index(datetime(2024, 5, 21, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 21, 21, 0))
end_index = dates_E3.index(datetime(2024, 5, 24, 11, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 24, 15, 0))
end_index = dates_E3.index(datetime(2024, 5, 26, 17, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 26, 19, 0))
end_index = dates_E3.index(datetime(2024, 5, 28, 3, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 5, 29, 11, 0))
end_index = dates_E3.index(datetime(2024, 6, 1, 11, 0))
Begin_End_IDX.append((begin_index,end_index+1))
begin_index = dates_E3.index(datetime(2024, 6, 2, 19, 0))
end_index = dates_E3.index(datetime(2024, 6, 6, 21, 0))
Begin_End_IDX.append((begin_index,end_index+1))

DIFF_interpolated_rhoa = []
for i in range(int(len(Begin_End_IDX))):
    time_diffs_in_hours, rhoa_diffs = plot_real_rhoa(Begin_End_IDX[i][0],Begin_End_IDX[i][1])
    DIFF_time.append(time_diffs_in_hours)
    DIFF_rhoa.append(rhoa_diffs)
    # Interpolation to time_values 
    interpolated_rhoa_diffs = np.interp(np.linspace(0,98,50), time_diffs_in_hours, rhoa_diffs)
    DIFF_interpolated_rhoa.append(interpolated_rhoa_diffs)

# outlier
# begin_index = dates_E3.index(datetime(2024, 4, 8, 11, 0))
# end_index = dates_E3.index(datetime(2024, 4, 11, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# outlier
# begin_index = dates_E3.index(datetime(2024, 4, 17, 11, 0))
# end_index = dates_E3.index(datetime(2024, 4, 18, 1, 0))
# plot_real_rhoa(begin_index,end_index+1)
mean_target = []
all_time = np.linspace(0, 98, 50)
for j in range(len(all_time)):
    target = []
    for i in range(len(Begin_End_IDX)):
        # If all_time[j] in DIFF_time[i] is not the last time point of DIFF_time[i]
        if all_time[j] <= DIFF_time[i][-1]:
            # Find the index of all_time[j] in DIFF_time[i]
    #         idx = DIFF_time[i].index(all_time[j])
            target.append(DIFF_interpolated_rhoa[i][j])
    # print(target)
    mean_target.append(np.mean(target))

# ax.plot(time_values, [(rhoa - median_rhoa_5[0]) for rhoa in median_rhoa_5],color='b', marker='o', linestyle='-', label='Comsol modeling result: Ks=1e-5 (m/s)')
# ax.plot(time_values, [(rhoa - median_rhoa_6[0]) for rhoa in median_rhoa_6],color='r', marker='o', linestyle='-', label='Comsol modeling result: Ks=1e-6 (m/s)')
# ax.plot(time_values, [(rhoa - median_rhoa_7[0]) for rhoa in median_rhoa_7],color='k', marker='o', linestyle='-', label='Comsol modeling result: Ks=1e-7 (m/s)')
# case_name = ['2m','4m','6m','8m']
# for i in range(len(case_name)):
#     ax.plot(time_values, [(rhoa - median_rhoa_dict[case_name[i]+'_5'][0]) for rhoa in median_rhoa_dict[case_name[i]+'_5']], marker='x', linestyle='-', label='Comsol result, Wet Depth= '+case_name[i]+', Ks=1e-5 (m/s)')
#     ax.plot(time_values, [(rhoa - median_rhoa_dict[case_name[i]+'_6'][0]) for rhoa in median_rhoa_dict[case_name[i]+'_6']], marker='o', linestyle='-', label='Comsol result, Wet Depth= '+case_name[i]+', Ks=1e-6 (m/s)')
#     ax.plot(time_values, [(rhoa - median_rhoa_dict[case_name[i]+'_7'][0]) for rhoa in median_rhoa_dict[case_name[i]+'_7']], marker='+', linestyle='-', label='Comsol result, Wet Depth= '+case_name[i]+', Ks=1e-7 (m/s)')


ax.plot(all_time, mean_target, color='g', marker='v', linestyle='-',linewidth=3,markersize=10, label='Mean of observed data')

ax.set_xlabel('Time after irrigation (hours)',fontsize=16)
ax.set_ylabel(r'Increase of Median Apparent Resistivity $+\rho_a$ ($\Omega \cdot m$)',fontsize=16)
ax.set_xticks( np.linspace(0, 96, 17) ) # 0,6,12...96
ax.set_xticklabels(np.arange(0, 97, 6), fontsize=14)
ax.set_yticklabels([str(round(tick, 1)) for tick in ax.get_yticks()], fontsize=14)
ax.set_title('Time Series of Median Apparent Resistivity Increase')
ax.grid(True)
# xtick rotation
plt.xticks(rotation=45)
# Put a legend to the right of outside the current axis
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)
# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], color='gray', linestyle='-', marker='o', label='Observation data', alpha=0.5),
#                    Line2D([0], [0], color='green', marker='v', label='Mean of observed data')]
# for i in range(len(case_name)):
#     legend_elements.append(Line2D([0], [0], marker='o', label='Comsol modeling result: '+case_name[i]+' Ks=1e-5 (m/s)'))
#     legend_elements.append(Line2D([0], [0], marker='o', label='Comsol modeling result: '+case_name[i]+' Ks=1e-6 (m/s)'))
#     legend_elements.append(Line2D([0], [0], marker='o', label='Comsol modeling result: '+case_name[i]+' Ks=1e-7 (m/s)'))

# plt.legend(handles=legend_elements,loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)
ax.set_xlim(-6,99)
plt.tight_layout(rect=[0, 0, 1, 1])  # 調整圖形布局
fig.savefig(join(pkl_ph,'delta_rhoa_hydro_estimate.png'), dpi=300, bbox_inches='tight')

plt.show()  # 顯示圖形

# %%

# %%

# # output to csv file
# df = pd.DataFrame({'dates': dates_E3, 'median_rhoa_E3': median_rhoa_E3})
# df.to_csv('median_rhoa_E3.csv', index=False)