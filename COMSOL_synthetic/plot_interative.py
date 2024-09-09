# -*- coding: big5 -*-
# %%
import pandas as pd
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'#'Microsoft Sans Serif'
import matplotlib
matplotlib.use('TkAgg')  # 設置後端為 TkAgg
import pickle
import matplotlib.dates as mdates
from datetime import datetime
from os import listdir
from os.path import isdir, join
from matplotlib.widgets import Cursor
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import timedelta
import pyperclip
from io import BytesIO
import io
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
import sys

output_path = r'D:\R2MSDATA\TARI_E3_test\output_second_timelapse_inveriosn'
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

def load_inversion_results(save_ph):
    output_ph = join(save_ph,'ERTManager')
    para_domain = pg.load(join(output_ph,'resistivity-pd.bms'))
    # mesh_fw = pg.load(join(output_ph,'resistivity-mesh.bms'))
    # Load data file
    data_path = join(output_ph,'data.dat')
    data = ert.load(data_path)
    investg_depth = (max(pg.x(data))-min(pg.x(data)))*0.2
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


pkl_ph = r'C:\Users\Git\TARI_research\COMSOL_synthetic'
process_data =  False
if process_data:
    output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]

    median_rhoa_E3 = []
    allall_mgrs = []

    for i,output_folder_name in enumerate(output_folders):
        print(output_folder_name)
        data_path = join(output_path,output_folder_name,'ERTManager','inverison_data.ohm')
        data = ert.load(data_path)
        allall_mgrs.append(load_inversion_results(join(output_path,output_folder_name)))
        median_rhoa_E3.append(np.median(data['rhoa']))


    with open(join(pkl_ph,'median_rhoa_E3output_second_timelapse_inveriosn.pkl'), 'wb') as f:
        pickle.dump(median_rhoa_E3, f)

else:
    # read field data from pickle file
    with open(join(pkl_ph,'median_rhoa_E3output_second_timelapse_inveriosn.pkl'), 'rb') as f:
        median_rhoa_E3 = pickle.load(f)

# %%
    
def read_hydro_data(data_path):
    df = pd.read_excel(data_path, sheet_name='彰化竹塘水田')
    # Pre-process and correct the TIMESTAMP column
    # df['TIMESTAMP'] = df['TIMESTAMP'].apply(correct_timestamp)

    numeric_columns = df.columns.drop('TIMESTAMP')
    # Set the TIMESTAMP column as the DataFrame index
    df.set_index('TIMESTAMP', inplace=True)
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # Resample the data to hourly averages
    hourly_avg = df.resample('h').sum()
    if 'Rain(mm)' in hourly_avg.columns:
        # Summing the rain data to the daily rainfall
        daily_rainfall = hourly_avg['Rain(mm)'].resample('d').sum()
    # Delete the RECORD, BattV column
    if 'Rain(mm)' in hourly_avg.columns:
        return hourly_avg, daily_rainfall
    else:
        return hourly_avg, None
    
_, daily_rainfall = read_hydro_data(r'C:\Users\Git\TARI_research\data\external\水文站資料彙整_20240731.xlsx')
# # write to pickle file
with open(r'C:\Users\Git\TARI_research\COMSOL_synthetic\daily_rainfall.pkl', 'wb') as f:
    pickle.dump(daily_rainfall, f)
with open(r'C:\Users\Git\TARI_research\COMSOL_synthetic\daily_rainfall.pkl', 'rb') as f:
    daily_rainfall = pickle.load(f)

# %%
# median_rhoa_df = pd.DataFrame({'Date': dates_E3, 'Median Apparent Resistivity': median_rhoa_E3})
# print(median_rhoa_df)
# median_rhoa_df.to_csv(r'C:\Users\Git\TARI_research\COMSOL_synthetic\median_rhoa_E3.csv', index=False)
# daily_rainfall_df = pd.DataFrame({'Rainfall': daily_rainfall})
# print(daily_rainfall_df)
# daily_rainfall_df.to_csv(r'C:\Users\Git\TARI_research\COMSOL_synthetic\daily_rainfall.csv', index=True)
# %%
fig, ax = plt.subplots(figsize=(20, 7))
median_rhoa_plot, = ax.plot(dates_E3, median_rhoa_E3, 'ro-',markersize=3 )
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# Rotate dates for better readability
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()  # Adjust layout to make room for the rotated date labels

begin_index = 1#dates_E3.index(datetime(2024, 3, 18, 1, 0))
end_index = -1#dates_E3.index(datetime(2024, 6, 6, 21, 0))
ax.set_xlim(dates_E3[begin_index], dates_E3[end_index])
ax.grid(True)
ax.set_title('Median Apparent Resistivity of E3')
ax.set_xlabel('Date')
ax.set_ylabel('Apparent Resistivity ($\Omega m$)')

ax2 = ax.twinx()  # Create a second Y-axis sharing the same X-axis
ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='Rainfall')
ax2.set_ylabel('Rainfall (mm)', color='c')  # Set label for the secondary Y-axis
ax2.tick_params(axis='y', labelcolor='c')  # Set ticks color for the secondary Y-axis
ax2.set_zorder(-100)  # Set the secondary Y-axis on bottom of the primary Y-axis
ax2.set_ylim([0,50])
# 使用 matplotlib.widgets.Cursor 來顯示游標
cursor = Cursor(ax, useblit=True, color='gray', linewidth=1)

app = QApplication(sys.argv)

selected_points = []  # 用於存儲選取的座標
all_mgrs = []
date_folder_names = []
fig2 = None  # 用於存儲第二張圖的引用
clicked_annotations = [] # 變量來追?點擊的狀態

annotation = ax.annotate("", xy=(0,0), xytext=(20,20),
                         textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                         arrowprops=dict(arrowstyle="->"))
annotation.set_visible(False)

def motion_hover(event):
    if event.inaxes == ax:  # 確保點擊發生在 ax 上
        if plt.get_current_fig_manager().toolbar.mode != '':
                return
        annotation_visbility = annotation.get_visible()
        if event.inaxes == ax:
            is_contained, annotation_index = median_rhoa_plot.contains(event)
            if is_contained:
                data_point_location = median_rhoa_plot.get_data()
                x_data, y_data = data_point_location
                x_data = mdates.date2num(x_data)  # 確保 x_data 是數值類型
                annotation.xy = (x_data[annotation_index['ind'][0]], y_data[annotation_index['ind'][0]])

                text_label = f'({mdates.num2date(x_data[annotation_index["ind"][0]]).strftime("%Y/%m/%d %H:%M")}, {y_data[annotation_index["ind"][0]]:.2f})'
                annotation.set_text(text_label)

                annotation.get_bbox_patch().set_facecolor('lightblue')  # 設定 box 顏色為淡藍色
                annotation.get_bbox_patch().set_alpha(0.6)  # 設定透明度為 0.6

                annotation.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annotation_visbility:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', motion_hover)

# 啟用滾輪縮放功能
def on_scroll(event):
    curr_xlim = ax.get_xlim()
    curr_ylim = ax.get_ylim()
    xdata = event.xdata  # 滾輪的 x 座標
    ydata = event.ydata  # 滾輪的 y 座標
    if event.button == 'up':
        scale_factor = 1 / 1.1
    elif event.button == 'down':
        scale_factor = 1.1
    else:
        scale_factor = 1
    
    new_width = (curr_xlim[1] - curr_xlim[0]) * scale_factor
    new_height = (curr_ylim[1] - curr_ylim[0]) * scale_factor

    relx = (curr_xlim[1] - xdata) / (curr_xlim[1] - curr_xlim[0])
    rely = (curr_ylim[1] - ydata) / (curr_ylim[1] - curr_ylim[0])

    ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
    ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
    plt.draw()

fig.canvas.mpl_connect('scroll_event', on_scroll)

def onclick(event):
    if event.inaxes == ax:  # 確保點擊發生在 ax 上
        if plt.get_current_fig_manager().toolbar.mode != '':
            return
        x_click = event.xdata
        y_click = event.ydata
        # 設置一個合理的距離閾值
        threshold_x = 0.01
        threshold_y = 0.01
        for x, y in zip(mdates.date2num(dates_E3), median_rhoa_E3):
            if abs(x - x_click) < threshold_x and abs(y - y_click) < threshold_y:

                if len(selected_points)%2 == 0:
                    for clicked_annotation in clicked_annotations:
                        clicked_annotation.set_visible(False)

                x_value = mdates.num2date(x).strftime("%Y/%m/%d %H:00")
                selected_points.append((x_value, y))
                print(f'Selected point: x = {x_value}, y = {y:.2f}')

                # 創建一個新的 annotation 並使其在點擊位置可見
                click_annotation = ax.annotate(
                    text=f'({x_value}, {y:.2f})',
                    xy=(x_click, y_click),
                    xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.6),
                    arrowprops=dict(arrowstyle="->")
                )
                click_annotation.set_visible(True)
                clicked_annotations.append(click_annotation)
                fig.canvas.draw_idle()

                # date: 2024-03-18 01:00:00 to '24031801_m_E3'
                date_folder_names.append(selected_points[-1][0].replace('/','').replace(':','').replace(' ','') [2:-2]+'_m_E3')
                save_ph = join(output_path,date_folder_names[-1])
                all_mgrs.append(load_inversion_results(save_ph))
                
                kw = dict(label='Resistivity $\Omega m$',
                        logScale=True,cMap='jet',cMin=10,cMax=316,
                        xlabel="x (m)", ylabel="z (m)",
                        orientation = 'vertical')
                # draw_second_fig(all_mgrs, date_folder_name, **kw)    
                if len(selected_points)%2 == 0:  
                    if selected_points[-1][1] - selected_points[-2][1] > 0:
                        cmap = +1
                    else:
                        cmap = -1
                    print(selected_points[-2][0],'→',selected_points[-1][0])
                    draw_second_fig(all_mgrs, date_folder_names,cmap = cmap)

                break


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
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    def red_colormap():
        colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 256)] 
        return LinearSegmentedColormap.from_list("red_only", colors)

    def blue_colormap():
        colors = [(0, 0, 1, i) for i in np.linspace(0, 1, 256)] 
        return LinearSegmentedColormap.from_list("blue_only", colors[::-1])
    # if cmap == 1:
    #     kw_diff = dict(cMin=0, cMax=10,logScale=False,
    #             label='Relative resistivity difference \n(%)',
    #             xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=red_colormap())
    # else:
    #     kw_diff = dict(cMin=-10, cMax=0,logScale=False,
    #             label='Relative resistivity difference \n(%)',
    #             xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=blue_colormap())
        
    # kw_diff = dict(cMin=-10, cMax=10,logScale=False,
    #             label='Relative resistivity difference \n(%)',
    #             xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap='bwr')
    colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
    nodes = [0, 0.95, 1]  # 範圍從0到-1是白色，-1到-10是白色到藍色的漸變
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    kw_diff = dict(cMin=-5, cMax=0,logScale=False,
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)
    
    # class StretchOutNormalize(plt.Normalize):
    #     def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
    #         self.low = low
    #         self.up = up
    #         plt.Normalize.__init__(self, vmin, vmax, clip)

    #     def __call__(self, value, clip=None):
    #         x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
    #         return np.ma.masked_array(np.interp(value, x, y))

    # midnorm=StretchOutNormalize(vmin=kw_diff['cMin'], vmax=kw_diff['cMax'], low=-1)

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
    ax.set_xlabel(kw_diff['xlabel'])#+' max_abs:{:.2f}'.format(max(abs(one_line_diff))))
    ax.set_ylabel(kw_diff['ylabel'])
    ax.grid(linestyle='--', linewidth=0.5,alpha = 0.5)

    plt.subplots_adjust(right=0.87,left=0.1)
    plt.show()
    
    return fig

def draw_second_fig( all_mgrs, date_folder_name,cmap, **kw):
    global fig2  # 使用全局變量來引用 fig2
    if fig2 is not None:
        plt.close(fig2)  # 關閉先前的圖
    # fig2 = plot_inverted_profile(all_mgrs[-1], date_folder_name, **kw)  # 調用自定義的函數繪製新圖
    fig2 = plot_difference_contour(all_mgrs[-2], all_mgrs[-1], date_folder_name[-2], date_folder_name[-1]
                                   ,cmap = cmap)  
    
    # 添加 ctrl+c 功能以複製圖像到剪貼簿
    def clipboard_handler(event):
        if event.key == 'ctrl+c':
            buf = io.BytesIO()
            fig2.savefig(buf, format='png',dpi=200)
            buf.seek(0)
            img = QImage.fromData(buf.getvalue())
            QApplication.clipboard().setImage(img)
            buf.close()
            print('Figure copied to clipboard.')

    fig2.canvas.mpl_connect('key_press_event', clipboard_handler)

# 添加 ctrl+c 功能以複製圖像到剪貼簿
def fig_clipboard_handler(event):
    if event.key == 'ctrl+c':
        buf = io.BytesIO()
        fig.savefig(buf, format='png',dpi=200)
        buf.seek(0)
        img = QImage.fromData(buf.getvalue())
        QApplication.clipboard().setImage(img)
        buf.close()
        print('First figure copied to clipboard.')

fig.canvas.mpl_connect('key_press_event', fig_clipboard_handler)

# 連接點擊事件
fig.canvas.mpl_connect('button_press_event', onclick)

# 獲取 Tkinter 的圖形窗口並設置位置
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+0+0")
plt.subplots_adjust(right=0.95,left=0.05,top=0.95,bottom=0.15)
plt.show()  # 顯示圖形



# 在圖形窗口關閉後打印所有選取的點
print("All selected points:")
for point in selected_points:
    print(f'x = {point[0]}, y = {point[1]:.2f}')