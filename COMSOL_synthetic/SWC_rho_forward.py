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
matplotlib.use('TkAgg')  # 設置後端為 TkAgg
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
def read_and_process_csv(file_path,data):
    pg.boxprint('Read and process CSV file: ' + file_path)
    data_SWC = pd.read_csv(file_path, comment='%')

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
output_path = r'D:\R2MSDATA\TARI_E3_test\output_318_606_rep2'
process_data = False
if process_data:
    time_points, median_rhoa_6 = read_and_process_csv(file_path = 'WaterContent_Ks_E-6.csv',data = data_filtered)
    time_points, median_rhoa_5 = read_and_process_csv(file_path = 'WaterContent_Ks_E-5.csv',data = data_filtered)
    time_points, median_rhoa_7 = read_and_process_csv(file_path = 'WaterContent_Ks_E-7.csv',data = data_filtered)

    with open('median_rhoa_6.pkl', 'wb') as f:
        pickle.dump(median_rhoa_6, f)

    with open('median_rhoa_5.pkl', 'wb') as f:
        pickle.dump(median_rhoa_5, f)

    with open('median_rhoa_7.pkl', 'wb') as f:
        pickle.dump(median_rhoa_7, f)

    
    output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]

    median_rhoa_E3 = []
    DATA = []
    for i,output_folder_name in enumerate(output_folders):
        print(output_folder_name)
        data_path = join(output_path,output_folder_name,'ERTManager','inverison_data.ohm')
        data = ert.load(data_path)
        data_filtered = filter_data(data)
        DATA.append(data_filtered)
        median_rhoa_E3.append(np.median(data_filtered['rhoa']))

    with open('median_rhoa_E3.pkl', 'wb') as f:
        pickle.dump(median_rhoa_E3, f)
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
with open(r'C:\Users\Git\TARI_research\COMSOL_synthetic\median_RHOA_E3.pkl', 'rb') as f:
    median_rhoa_E3 = pickle.load(f)

# with open('dates_E3.pkl', 'rb') as f:
#     dates_E3 = pickle.load(f)
# %%




# %%
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(dates_E3, median_rhoa_E3, 'ro',markersize=3 )
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
# ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# Rotate dates for better readability
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()  # Adjust layout to make room for the rotated date labels

begin_index = dates_E3.index(datetime(2024, 3, 18, 1, 0))
end_index = dates_E3.index(datetime(2024, 6, 6, 21, 0))
# for i in range(0, len(dates)):
#     ax.text(dates[i], median_RHOA[i], f'{rrms[i]:.1f}', fontsize=7.5)
ax.set_xlim(dates_E3[begin_index], dates_E3[end_index])
# ax.set_ylim([28,32])
ax.grid(True)

# 使用 matplotlib.widgets.Cursor 來顯示游標
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

selected_points = []  # 用於存儲選取的座標
all_mgrs = []
date_folder_names = []
fig2 = None  # 用於存儲第二張圖的引用

def onclick(event):
    if event.inaxes == ax:  # 確保點擊發生在 ax 上
        x_click = event.xdata
        y_click = event.ydata
        # 設置一個合理的距離閾值
        threshold = 0.1
        for x, y in zip(mdates.date2num(dates_E3), median_rhoa_E3):
            if abs(x - x_click) < threshold and abs(y - y_click) < threshold:
                x_value = mdates.num2date(x).strftime("%Y-%m-%d %H:00")
                selected_points.append((x_value, y))
                print(f'Selected point: x = {x_value}, y = {y:.2f}')
                # date: 2024-03-18 13:00:00 to '24031801_m_E3'
                date_folder_names.append(selected_points[-1][0].replace('-','').replace(':','').replace(' ','') [2:-2]+'_m_E3')
                save_ph = join(output_path,date_folder_names[-1])
                all_mgrs.append(load_inversion_results(save_ph))
                
                kw = dict(label='Resistivity $\Omega m$',
                        logScale=True,cMap='jet',cMin=10,cMax=316,
                        xlabel="x (m)", ylabel="z (m)",
                        orientation = 'vertical')
                # draw_second_fig(all_mgrs, date_folder_name, **kw)    
                break

        if len(selected_points)%2 == 0:  
            if selected_points[-1][1] - selected_points[-2][1] > 0:
                cmap = +1
            else:
                cmap = -1
            
            draw_second_fig(all_mgrs, date_folder_names,cmap = cmap)

def load_inversion_results(save_ph):
    output_ph = join(save_ph,'ERTManager')
    para_domain = pg.load(join(output_ph,'resistivity-pd.bms'))
    # mesh_fw = pg.load(join(output_ph,'resistivity-mesh.bms'))
    # Load data file
    data_path = join(output_ph,'data.dat')
    data = ert.load(data_path)
    investg_depth = (max(pg.x(data))-min(pg.x(data)))*0.25
    # Load model response
    # resp_path = join(output_ph,'model_response.txt')
    # response = np.loadtxt(resp_path)
    model = pg.load(join(output_ph,'resistivity.vector'))
    coverage = pg.load(join(output_ph,'resistivity-cov.vector'))

    inv_info_path = join(output_ph,'inv_info.txt')
    Line = []
    section_idx = 0
    with open(inv_info_path, 'r') as read_obj:
        for i,line in enumerate(read_obj):
                Line.append(line.rstrip('\n'))

    final_result = Line[Line.index('## Final result ##')+1:Line.index('## Inversion parameters ##')]
    rrms = float(final_result[0].split(':')[1])
    chi2 = float(final_result[1].split(':')[1])
    inversion_para = Line[Line.index('## Inversion parameters ##')+1:Line.index('## Iteration ##')]
    lam = int(inversion_para[0].split(':')[1])
    iteration = Line[Line.index('## Iteration ##')+2:]
    rrmsHistory = np.zeros(len(iteration))
    chi2History = np.zeros(len(iteration))
    for i in range(len(iteration)):
        rrmsHistory[i] = float(iteration[i].split(',')[1])
        chi2History[i] = float(iteration[i].split(',')[2])

    mgr_dict = {'paraDomain': para_domain, 
                # 'mesh_fw': mesh_fw, 
                'data': data, 
                # 'response': response, 
                'model': model, 'coverage': coverage, 
                'investg_depth': investg_depth, 
                'rrms': rrms, 'chi2': chi2, 'lam': lam,
                'rrmsHistory': rrmsHistory, 'chi2History': chi2History}

    return mgr_dict


def plot_inverted_profile(mgr, urf_file_name, **kw):
    data = mgr['data']
    lam = mgr['lam']
    rrms = mgr['rrms']
    chi2 = mgr['chi2']
    fig, ax = plt.subplots(figsize=(8, 3))
    _, cb = pg.show(mgr['paraDomain'],mgr['model'],ax=ax,
                    coverage=1, **kw)
    ax.plot(np.array(pg.x(data)), np.array(pg.z(data)), 'kd',markersize = 3)
    title_str = 'Inverted Resistivity Profile at {}number of data={:.0f}, rrms={:.2f}%, $\chi^2$={:.3f}, $\lambda$={:.0f}'.format(
        datetime.strptime(urf_file_name[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00")+'\n',len(data['rhoa']),rrms,chi2,lam)
    ax.set_title(title_str)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

    cb_ytick_label = np.round(cb.ax.get_yticks(),decimals = 0)
    cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick_label])

    plt.show()

    return fig

def plot_difference_profile(mgr1, mgr2, urf_file_name1, urf_file_name2,cmap):
    model1 = mgr1['model']
    model2 = mgr2['model']
    data = mgr1['data']
    left = min(pg.x(data))
    right = max(pg.x(data))
    depth = mgr1['investg_depth']
    diff = (np.log10(model2) - np.log10(model1))/np.log10(model1)*100
    def red_colormap():
        colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 256)] 
        return LinearSegmentedColormap.from_list("red_only", colors)

    def blue_colormap():
        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 256)] 
        return LinearSegmentedColormap.from_list("blue_only", colors[::-1])
    fig, ax = plt.subplots(figsize=(8, 3))
    if cmap == 1:
        kw_diff = dict(cMin=0, cMax=10,logScale=False,
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
        _, cb = pg.show(mgr1['paraDomain'],diff,ax = ax, cMap=red_colormap(), **kw_diff)
    else:
        kw_diff = dict(cMin=-10, cMax=0,logScale=False,
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
        _, cb = pg.show(mgr1['paraDomain'],diff,ax = ax, cMap=blue_colormap(), **kw_diff)
    ax.set_aspect('equal')
    title_str = 'Resistivity Difference Profile\n{} vs {}'.format(datetime.strptime(urf_file_name1[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"),
                                                                  datetime.strptime(urf_file_name2[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"))
    ax.set_title(title_str)
    ax.set_xlim(left,right)
    ax.set_ylim(-depth,0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))

    plt.show()

    return fig

def plot_difference_contour(mgr1, mgr2, urf_file_name1, urf_file_name2,cmap):
    model1 = mgr1['model']
    model2 = mgr2['model']
    data = mgr1['data']
    left = min(pg.x(data))
    right = max(pg.x(data))
    depth = mgr1['investg_depth']
    mesh_x = np.linspace(left, right, 250)
    mesh_y = np.linspace(-depth, 0, 150)
    grid = pg.createGrid(x=mesh_x, y= mesh_y)
    X,Y = np.meshgrid(mesh_x, mesh_y)
    one_line_diff = (np.log10(model2) - np.log10(model1))/np.log10(model1)*100
    diff_grid = np.reshape(pg.interpolate(mgr1['paraDomain'], one_line_diff, grid.positions()), (len(mesh_y), len(mesh_x)))
    fig, ax = plt.subplots(figsize=(8, 6))
    def red_colormap():
        colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 256)] 
        return LinearSegmentedColormap.from_list("red_only", colors)

    def blue_colormap():
        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 256)] 
        return LinearSegmentedColormap.from_list("blue_only", colors[::-1])
    if cmap == 1:
        kw_diff = dict(cMin=0, cMax=10,logScale=False,
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=red_colormap())
    else:
        kw_diff = dict(cMin=-10, cMax=0,logScale=False,
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=blue_colormap())

    ax.contourf(X,Y, diff_grid, cmap=kw_diff['cMap'], levels=32,
                vmin=kw_diff['cMin'],vmax=kw_diff['cMax'],antialiased=True)
    ax.set_aspect('equal')
    ax.set_xlim(left, right)
    ax.set_ylim(-depth, 0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="4.5%", pad=.1)
    m = plt.cm.ScalarMappable(cmap=kw_diff['cMap'])
    m.set_array(diff_grid)
    m.set_clim(kw_diff['cMin'],kw_diff['cMax'])
    cb = plt.colorbar(m, boundaries=np.linspace(kw_diff['cMin'],kw_diff['cMax'], 50),cax=cbaxes)
    cb.ax.set_yticks(np.linspace(kw_diff['cMin'],kw_diff['cMax'],5))
    cb.ax.set_yticklabels(['{:.2f}'.format(x) for x in cb.ax.get_yticks()])
    cb.ax.set_ylabel(kw_diff['label'])
    title_str = 'Resistivity Difference Profile\n{} vs {}'.format(datetime.strptime(urf_file_name1[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"),
                                                                  datetime.strptime(urf_file_name2[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"))
    ax.set_title(title_str)
    ax.set_xlabel(kw_diff['xlabel']+' max_abs:{:.2f}'.format(max(abs(one_line_diff))))
    ax.set_ylabel(kw_diff['ylabel'])

    plt.show()
    
    return fig

def draw_second_fig( all_mgrs, date_folder_name,cmap, **kw):
    global fig2  # 使用全局變量來引用 fig2
    if fig2 is not None:
        plt.close(fig2)  # 關閉先前的圖
    # fig2 = plot_inverted_profile(all_mgrs[-1], date_folder_name, **kw)  # 調用自定義的函數繪製新圖
    fig2 = plot_difference_contour(all_mgrs[-2], all_mgrs[-1], date_folder_name[-2], date_folder_name[-1]
                                   ,cmap = cmap)  
    


# 連接點擊事件
fig.canvas.mpl_connect('button_press_event', onclick)
# 獲取 Tkinter 的圖形窗口並設置位置
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+0+0")
plt.show()  # 顯示圖形

# 在圖形窗口關閉後打印所有選取的點
print("All selected points:")
for point in selected_points:
    print(f'x = {point[0]}, y = {point[1]:.2f}')


# %%



















# %%
# time_values = [int(time.strip('t=h')) for time in time_points]
# plt.close('all')
# fig,ax = plt.subplots(figsize=(15, 6))
# ax.plot(time_values, [(rhoa - median_rhoa_5[0]) for rhoa in median_rhoa_5],color='b', marker='o', linestyle='-', label='Ks=1e-5')
# ax.plot(time_values, [(rhoa - median_rhoa_6[0]) for rhoa in median_rhoa_6],color='r', marker='o', linestyle='-', label='Ks=1e-6')
# ax.plot(time_values, [(rhoa - median_rhoa_7[0]) for rhoa in median_rhoa_7],color='k', marker='o', linestyle='-', label='Ks=1e-7')

# ax.set_xlabel('Time after irrigation (hours)')
# ax.set_ylabel('$\Delta$ Median rhoa (Ohm m)')
# ax.set_title('Time Series of $\Delta$ Median rhoa Values')
# ax.grid(True)
# ax.set_xticks(time_values)
# # xtick rotation
# plt.xticks(rotation=45)
# plt.tight_layout()


# def plot_real_rhoa(begin_index,end_index):
#     time_segment = dates_E3[begin_index:end_index]
#     time_diffs_in_hours = [(time - time_segment[0]).total_seconds() / 3600 for time in time_segment]
#     rhoa_diffs = [(rhoa - median_rhoa_E3[begin_index:end_index][0]) for rhoa in median_rhoa_E3[begin_index:end_index]]
#     ax.plot(time_diffs_in_hours, rhoa_diffs, marker='.', linestyle='-', label='observed data: '+str(time_segment[0].date())+'~'+str(time_segment[-1].date()))

# # begin_index = dates_E3.index(datetime(2024, 3, 18, 13, 0))
# # end_index = dates_E3.index(datetime(2024, 3, 20, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 3, 21, 13, 0))
# # end_index = dates_E3.index(datetime(2024, 3, 22, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 3, 26, 11, 0))
# end_index = dates_E3.index(datetime(2024, 3, 27, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 3, 27, 13, 0))
# end_index = dates_E3.index(datetime(2024, 3, 28, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 3, 28, 13, 0))
# end_index = dates_E3.index(datetime(2024, 3, 29, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 3, 29, 13, 0))
# end_index = dates_E3.index(datetime(2024, 3, 30, 1, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 3, 30, 15, 0))
# end_index = dates_E3.index(datetime(2024, 4, 1, 1, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 4, 1, 11, 0))
# end_index = dates_E3.index(datetime(2024, 4, 4, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 4, 8, 11, 0))
# end_index = dates_E3.index(datetime(2024, 4, 11, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 4, 15, 13, 0))
# end_index = dates_E3.index(datetime(2024, 4, 17, 3, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 4, 17, 11, 0))
# end_index = dates_E3.index(datetime(2024, 4, 18, 1, 0))
# plot_real_rhoa(begin_index,end_index+1)

# begin_index = dates_E3.index(datetime(2024, 4, 18, 13, 0))
# end_index = dates_E3.index(datetime(2024, 4, 22, 11, 0))
# plot_real_rhoa(begin_index,end_index+1)
# # ----------------------------------------
# # begin_index = dates_E3.index(datetime(2024, 5, 7, 15, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 10, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 10, 13, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 13, 11, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 13, 23, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 16, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 16, 19, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 19, 13, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 19, 15, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 21, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 21, 21, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 24, 11, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 24, 15, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 26, 17, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 26, 19, 0))
# # end_index = dates_E3.index(datetime(2024, 5, 28, 3, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 5, 29, 11, 0))
# # end_index = dates_E3.index(datetime(2024, 6, 1, 11, 0))
# # plot_real_rhoa(begin_index,end_index+1)

# # begin_index = dates_E3.index(datetime(2024, 6, 2, 19, 0))
# # end_index = dates_E3.index(datetime(2024, 6, 6, 21, 0))
# # plot_real_rhoa(begin_index,end_index+1)
# # Put a legend to the right of outside the current axis
# ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1), ncol=1)

# plt.tight_layout(rect=[0, 0, 1, 1])  # 調整圖形布局

# # 使用 mplcursors 來啟用游標選取功能
# cursor = mplcursors.cursor(ax, hover=True)

# @cursor.connect("add")
# def on_add(sel):
#     # 設定自定義的 x 座標格式
#     sel.annotation.set(text=f'x: {sel.target[0]:.2f}\ny: {sel.target[1]:.2f}', backgroundcolor='white')
#     sel.annotation.draggable(True)

# plt.show()  # 顯示圖形


# %%

# %%

# # output to csv file
# df = pd.DataFrame({'dates': dates_E3, 'median_rhoa_E3': median_rhoa_E3})
# df.to_csv('median_rhoa_E3.csv', index=False)