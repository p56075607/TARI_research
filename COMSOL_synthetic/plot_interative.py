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
import os
from os import listdir
from os.path import isdir, join
from matplotlib.widgets import Cursor
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import timedelta
from io import BytesIO
import io
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
import sys

linename='E3'
pkl_path = r'D:\R2MSDATA\TARI_E3_test'
output_path = join(pkl_path,'output_second_timelapse_inveriosn')
pkl_files = [f for f in listdir(pkl_path) if f.endswith('.pkl')]


# for i, pkl_file in enumerate(pkl_files):
#     with open(join(pkl_path, pkl_file), 'rb') as f:
#         if i == 0:
#             dates_E3 = pickle.load(f)
#             median_RHOA_E3 = pickle.load(f)
#         else:
#             dates_E3 += pickle.load(f)
#             median_RHOA_E3 += pickle.load(f)

# # %%
# # Save the dates and median_RHOA_E1 to a pickle file
# with open('median_RHOA_E3_and_date.pkl', 'wb') as f:
#     pickle.dump(dates_E3, f)
#     pickle.dump(median_RHOA_E3, f)

# %%
# read the dates and median_RHOA_E1 from the pickle file
with open('median_RHOA_E3_and_date.pkl', 'rb') as f:
    pickled_dates_E3 = pickle.load(f)
    pickled_median_RHOA_E3 = pickle.load(f)

dates_E3 = pickled_dates_E3
median_RHOA_E3 = pickled_median_RHOA_E3
# %%
# To pandas DataFrame
# dates_E3 = pd.to_datetime(pickled_dates_E3)
# median_RHOA_E3 = np.array(pickled_median_RHOA_E3)
# rhoa_df = pd.DataFrame({'Date': dates_E3, 'Median Apparent Resistivity': median_RHOA_E3})
# # output the rhoa_df to csv file
# rhoa_df.to_csv('median_rhoa_E3.csv', index=False)
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
        return hourly_avg, daily_rainfall, df
    else:
        return hourly_avg, None
    
_, daily_rainfall, hydr_df = read_hydro_data(r'C:\Users\Git\TARI_research\data\external\水文站資料彙整_20240909.xlsx')

# %%
# plot SWC timeseries hydr_df.columns: '10cm', '20cm', '30cm', '40cm', '50cm', '60cm', '80cm', '100cm','150cm', '200cm', '300cm', '400cm', '500cm', '600cm', '700cm', '800cm','900cm'
# fig, ax = plt.subplots(figsize=(8, 8))
# for column in hydr_df.columns:
#     if column in ['10cm', '50cm', '100cm','150cm', '200cm', '300cm', '400cm', '500cm', '600cm', '700cm', '800cm','900cm']:
#         print(column)
#         # ax.scatter(hourly_avg.index, hourly_avg[column], s=1, marker='o', label=column)
#         ax.plot(hydr_df.index, hydr_df[column],linewidth=1, label=column)
#         ax.set_title('Hourly Averages of Hydrological Data and Rainfall')
#         ax.set_xlabel('Time (YYYY-MM-DD)')
#         ax.set_ylabel('Soil Moisture (%)')
#         ax.set_ylim(0, 50)
#         # set xlim from 2024-04-13 to 2024-04-20
#         ax.set_xlim(datetime(2024, 4, 13, 0, 0), datetime(2024, 4, 20, 0, 0))
#     ax.legend(loc='upper right')
#     ax.grid(linestyle='--', linewidth=0.5)

# plt.show()

# %%
# median_rhoa_df = pd.DataFrame({'Date': dates_E3, 'Median Apparent Resistivity': median_rhoa_E3})
# print(median_rhoa_df)
# median_rhoa_df.to_csv(r'C:\Users\Git\TARI_research\COMSOL_synthetic\median_rhoa_E3.csv', index=False)
# daily_rainfall_df = pd.DataFrame({'Rainfall': daily_rainfall})
# print(daily_rainfall_df)
# daily_rainfall_df.to_csv(r'C:\Users\Git\TARI_research\COMSOL_synthetic\daily_rainfall.csv', index=True)
# %%
fig, ax = plt.subplots(figsize=(25, 8))
median_rhoa_plot, = ax.plot(dates_E3, median_RHOA_E3, 'ro',markersize=3 ,zorder=2)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# set xy ticks label fontsize 
fz_minor = 25
plt.xticks(fontsize=fz_minor,rotation=45, ha='right', rotation_mode='anchor')
plt.yticks(fontsize=fz_minor)
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
plt.tight_layout()  # Adjust layout to make room for the rotated date labels

begin_index = 1#dates_E3.index(datetime(2024, 3, 18, 1, 0))
end_index = -1#dates_E3.index(datetime(2024, 6, 6, 21, 0))
ax.set_xlim(dates_E3[begin_index], dates_E3[end_index])
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
fz_major = 30
# ax.set_title('Median Apparent Resistivity',fontsize=fz_major,fontweight='bold')
ax.set_xlabel('Date',fontsize=fz_major)
ax.set_ylabel('Apparent Resistivity ($\Omega m$)',fontsize=fz_major)

# instrument events: 4/6, 4/26, 5/1, 7/25, 7/29
instrument_events = [datetime(2024, 4, 6, 0, 0), datetime(2024, 4, 26, 0, 0), datetime(2024, 5, 1, 0, 0), datetime(2024, 7, 25, 0, 0), datetime(2024, 7, 29, 0, 0)]
for event in instrument_events:
    ax.axvline(event, color='k', linestyle='-', linewidth=3)

# irregration events: 
irregration_events = [datetime(2024, 3, 18, 13, 0),
                      datetime(2024, 3, 21, 13, 0),
                      datetime(2024, 3, 26, 11, 0),
                        datetime(2024, 3, 27, 13, 0),
                      datetime(2024, 3, 28, 13, 0),
                      datetime(2024, 3, 29, 13, 0),
                      datetime(2024, 3, 30, 15, 0),
                      datetime(2024, 4, 1, 11, 0),
                      datetime(2024, 4, 8, 11, 0),
                      datetime(2024, 4, 15, 13, 0),
                      datetime(2024, 4, 17, 11, 0),
                      datetime(2024, 4, 18, 13, 0),
                      datetime(2024, 5, 7, 15, 0),
                      datetime(2024, 5, 10, 13, 0),
                      datetime(2024, 5, 13, 23, 0),
                      datetime(2024, 5, 16, 19, 0),
                      datetime(2024, 5, 19, 15, 0),
                      datetime(2024, 5, 21, 21, 0),
                      datetime(2024, 5, 24, 15, 0),
                      datetime(2024, 5, 26, 19, 0),
                    #   datetime(2024, 5, 29, 11, 0),
                    #   datetime(2024, 6, 2, 19, 0),
                      datetime(2024, 6, 8, 21, 0),
                      datetime(2024, 6, 15,13, 0),
                      datetime(2024, 6, 20,11, 0),
                      datetime(2024, 6, 25,23, 0),
                      datetime(2024, 7, 13,11, 0),
                      datetime(2024, 7, 14,13, 0),
                      datetime(2024, 7, 17,13, 0),
                    datetime(2024, 7, 20, 13, 0),
                        datetime(2024, 7, 21, 11, 0),
                        datetime(2024, 7, 22, 11, 0),
                        datetime(2024, 7, 23, 11, 0),
                        datetime(2024, 7, 30, 11, 0),
                        datetime(2024, 8, 1, 15, 0),
                        datetime(2024, 8, 3, 11, 0),
                        datetime(2024, 8, 4, 15 ,0),
                        datetime(2024, 8, 10, 11, 0),
                        datetime(2024, 8, 11, 17, 0),
                        datetime(2024, 8, 13, 11, 0),
                        datetime(2024, 8, 15, 15, 0),
                        datetime(2024, 8, 19, 3, 0),
                        datetime(2024, 8, 20, 15, 0),
                        datetime(2024, 8, 21, 15, 0),
                        datetime(2024, 8, 22, 11, 0),
                        datetime(2024, 8, 23, 11, 0),
                        datetime(2024, 8, 24, 11, 0),
                        datetime(2024, 8, 26, 17, 0),
                        datetime(2024, 8, 29, 21, 0),
                        datetime(2024, 8, 31, 1, 0),
                        datetime(2024, 9, 3, 11, 0),
                      ]
# for event in irregration_events:
#     ax.axvline(event, color='g', linestyle='-', linewidth=3)



ax2 = ax.twinx()  # Create a second Y-axis sharing the same X-axis
ax2.bar(daily_rainfall.index, daily_rainfall, width=1,align='edge', alpha=1,color=[0.3010, 0.7450, 0.9330], label='Rainfall',zorder=1)
ax2.set_ylabel('Rainfall (mm/day)', color=[0.3010, 0.7450, 0.9330],fontsize=fz_major)  # Set label for the secondary Y-axis
ax2.tick_params(axis='y', labelcolor=[0.3010, 0.7450, 0.9330], length=10,width=3, direction='in')  # Set ticks color for the secondary Y-axis
# set y ticks label fontsize
plt.yticks(fontsize=fz_minor)
# ax2.set_ylim([0,50])
ax.set_zorder(ax2.get_zorder()+1)
ax.set_frame_on(False)
width = 3
ax2.spines['top'].set_linewidth(width)
ax2.spines['right'].set_linewidth(width)
ax2.spines['bottom'].set_linewidth(width)
ax2.spines['left'].set_linewidth(width)

fig.savefig('TARI_E3_timeseries.png', dpi=300, bbox_inches='tight')
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

def onclick(event):
    if event.inaxes == ax:  # 確保點擊發生在 ax 上
        if plt.get_current_fig_manager().toolbar.mode != '':
            return
        x_click = event.xdata
        y_click = event.ydata
        # 設置一個合理的距離閾值
        threshold_x = 0.1
        threshold_y = 0.1
        for x, y in zip(mdates.date2num(dates_E3), median_RHOA_E3):
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

# %%
# %%
# Plot time lapse resistivity different profile with for loop
def plot_difference_contour(mgr1, mgr2, urf_file_name1, urf_file_name2,save_folder, **kw_diff):
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
    m.set_clim(kw['cMin'],kw['cMax'])
    cb = plt.colorbar(m, boundaries=np.linspace(kw['cMin'],kw['cMax'], 50),cax=cbaxes)
    cb.ax.set_yticks(np.linspace(kw['cMin'],kw['cMax'],5))
    cb.ax.set_yticklabels(['{:.2f}'.format(x) for x in cb.ax.get_yticks()])
    cb.ax.set_ylabel(kw['label'])
    title_str = 'Resistivity Difference Profile\n{} vs {}'.format(datetime.strptime(urf_file_name1[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"),
                                                                  datetime.strptime(urf_file_name2[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"))
    ax.set_title(title_str)
    ax.set_xlabel(kw['xlabel'])#+' max_abs:{:.2f}'.format(max(abs(one_line_diff))))
    ax.set_ylabel(kw['ylabel'])
    ax.grid(linestyle='--', linewidth=0.5,alpha = 0.5)
    fig.savefig(join(save_folder, urf_file_name1[:8]+'_vs_'+urf_file_name2[:8]+'_contour.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return X,Y, diff_grid

all_mgrs = []
output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]
# begin_index = dates_E3.index(datetime(2024, 4, 14, 3, 0))
# end_index = dates_E3.index(datetime(2024, 4, 18, 3, 0))
begin_index = dates_E3.index(datetime(2024, 6, 2, 3, 0))
end_index = dates_E3.index(datetime(2024, 6, 6, 21, 0))
save_ph = join(output_path,output_folders[begin_index])
all_mgrs.append(load_inversion_results(save_ph))

colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 0.9, 1]  # 範圍從0到-1是白色，-1到-10是白色到藍色的漸變
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

kw = dict(cMin=-5, cMax=0,logScale=False,
            label='Relative resistivity difference \n(%)',
            xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)


save_folder = r'D:\R2MSDATA\TARI_E3_test\timelapsed_resistivity_difference'
if not os.path.exists(save_folder):
    # if not, create the folder
    os.makedirs(save_folder)
    print(f'Folder "{save_folder}" created.')
else:
    print(f'Folder "{save_folder}" already exists.')

count = 1
every_diff_grid = []
for j in range(begin_index+1,end_index+1,1):
    save_ph = join(output_path,output_folders[j])
    all_mgrs.append(load_inversion_results(save_ph))
    X,Y, diff_grid = plot_difference_contour(all_mgrs[0], all_mgrs[count], output_folders[begin_index], output_folders[j],save_folder, **kw)
    every_diff_grid.append(diff_grid)
    count += 1

# %%
begin_index = dates_E3.index(datetime(2024, 6, 2, 3, 0))
end_index = dates_E3.index(datetime(2024, 6, 6, 21, 0))

mgr1 = load_inversion_results(join(output_path,output_folders[begin_index]))
mgr2 = load_inversion_results(join(output_path,output_folders[end_index]))
# diff_mgrs = []
# for j in range(begin_index,end_index+1,1):
#     print(output_folders[j])
#     save_ph = join(output_path,output_folders[j])
#     diff_mgrs.append(load_inversion_results(save_ph))
# %%
# Plot intensity map
# pick up X index from 18~19 m
X_index = np.where((X[0] >= 10) & (X[0] <= 70))[0]
mean_diff = np.empty((len(Y[:,0]),0))
for i in range(len(every_diff_grid)):
    data = every_diff_grid[i][:, X_index]
    # Filter to include only negative values
    neg_data = np.where(data < 0, data, np.nan)
    # Compute mean of negative values along axis=1, ignoring NaNs
    diff_1819 = np.nanmean(neg_data, axis=1)
    # Reshape diff_1819 to 2D and append along axis=1
    mean_diff = np.append(mean_diff, diff_1819[:, np.newaxis], axis=1)

colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 0.65, 1]  # 範圍從0到-1是白色，-1到-10是白色到藍色的漸變
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
kw = dict(cMin=-1.5, cMax=0,logScale=False,
            label='Relative resistivity difference \n(%)',
            xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)

Tmesh,Ymesh = np.meshgrid(dates_E3[begin_index:end_index],Y[:,0] )
fig,ax = plt.subplots(figsize=(27.5,12))

# pc = ax.pcolor(Tmesh,Ymesh,(mean_diff),cmap=custom_cmap,vmin=-5, vmax=0)
levels = np.linspace(kw['cMin'], kw['cMax'], 32)  # Adjust the number of levels as needed

pc = ax.contourf(Tmesh, Ymesh, (mean_diff), cmap=kw['cMap'], levels=levels, vmin=kw['cMin'], vmax=kw['cMax'],zorder=2)
# ax.set_ylim([-8,0])
ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=1))
ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
fz_minor = 40
plt.xticks(fontsize=fz_minor,fontweight='bold')
plt.yticks(fontsize=fz_minor,fontweight='bold')

xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()
ax.set_xticks(xticks)
ax.tick_params(axis='both', which='major', length=10,width=3)
ax.tick_params(axis='both', which='minor', length=5,width=1.5)
plt.draw()
fz_major = fz_minor
ax.set_xticklabels(xticklabels,rotation = 45, ha='right',rotation_mode="anchor")
ax.grid(linestyle='--',color='w',linewidth=0.5)
ax.set_ylabel('Depth (m)',fontsize=fz_major,fontweight='bold')
ax.set_xlabel('Date (2024/mm/dd)',fontsize=fz_major,fontweight='bold')
# divider = make_axes_locatable(ax)
# cbaxes = divider.append_axes("right", size="4.5%", pad=10)
# m = plt.cm.ScalarMappable(cmap=kw['cMap'])
# m.set_array(mean_diff)
# m.set_clim(kw['cMin'],kw['cMax'])
# cb = plt.colorbar(m, boundaries=np.linspace(kw['cMin'],kw['cMax'], 50),cax=cbaxes)

cb = fig.colorbar(pc, pad=0.08)
cb.ax.set_yticks(np.linspace(kw['cMin'],kw['cMax'],6))
cb.ax.set_ylabel(kw['label'],fontsize=fz_major,fontweight='bold')
cb.ax.yaxis.set_tick_params(labelsize=fz_minor)
for label in cb.ax.yaxis.get_ticklabels():
    label.set_fontweight('bold')


title_str = 'Spacial & Temporal Intensity Plot of the '+kw['label'][:-5]
ax.set_title(title_str,fontsize=fz_major,fontweight='bold')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)

ax2 = ax.twinx()
ax2.bar(daily_rainfall.index, daily_rainfall, width=1,align='edge', alpha=1,color=[0.3010, 0.7450, 0.9330], label='Rainfall',zorder=1)
ax2.set_ylabel('Rainfall (mm/day)', color=[0.3010, 0.7450, 0.9330],fontsize=fz_major,fontweight='bold')  # Set label for the secondary Y-axis
ax2.tick_params(axis='y', labelcolor=[0.3010, 0.7450, 0.9330], length=10,width=3)  # Set ticks color for the secondary Y-axis
ax2.set_xlim(dates_E3[begin_index], dates_E3[end_index])
ax2.yaxis.set_tick_params(labelsize=fz_minor)
ax2.set_ylim([0, 100])
plt.yticks(fontsize=fz_minor,fontweight='bold')
plt.show()
fig.savefig(join(save_folder, 'E3_intensity.png'), dpi=300, bbox_inches='tight')
plt.close()
# %%
# %%
# Plot time series resistivity
def plot_timeseries(median_RHOA, daily_rainfall, dates, begin_index, end_index,current_index):
    fig, ax = plt.subplots(figsize=(30, 10))
    median_rhoa_plot, = ax.plot(dates[begin_index:end_index], median_RHOA[begin_index:end_index], 'ko',markersize=15 ,zorder=2)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    # set xy ticks label fontsize 
    fz_minor = 50
    plt.xticks(fontsize=fz_minor,rotation=45, ha='right', rotation_mode='anchor',fontweight='bold')
    plt.yticks(fontsize=fz_minor,fontweight='bold')
    ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
    ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
    plt.tight_layout()  # Adjust layout to make room for the rotated date labels

    # xy lim have some space
    # 6 hours before and after
    ax.set_xlim(dates[begin_index] - timedelta(hours=6), dates[end_index] + timedelta(hours=6))
    # 10% of the range above and below the data
    ax.set_ylim([min(median_RHOA[begin_index:end_index]) - 0.1 * (max(median_RHOA[begin_index:end_index]) - min(median_RHOA[begin_index:end_index])),
                 max(median_RHOA[begin_index:end_index]) + 0.1 * (max(median_RHOA[begin_index:end_index]) - min(median_RHOA[begin_index:end_index]))])
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    fz_major = 50
    # ax.set_title('Median Apparent Resistivity',fontsize=fz_major,fontweight='bold')
    ax.set_xlabel('Date (2024/mm/dd)',fontsize=fz_major,fontweight='bold')
    ax.set_ylabel('Apparent Resistivity ($\Omega m$)',fontsize=fz_major,fontweight='bold')
    ax2 = ax.twinx()  # Create a second Y-axis sharing the same X-axis
    ax2.bar(daily_rainfall.index, daily_rainfall, width=1,align='edge', alpha=1,color=[0.3010, 0.7450, 0.9330], label='Rainfall',zorder=1)
    ax2.set_ylabel('Rainfall (mm/day)', color=[0.3010, 0.7450, 0.9330],fontsize=fz_major,fontweight='bold')  # Set label for the secondary Y-axis
    ax2.tick_params(axis='y', labelcolor=[0.3010, 0.7450, 0.9330], length=10,width=3, direction='in')  # Set ticks color for the secondary Y-axis
    ax2.set_ylim([0, 100])
    # set y ticks label fontsize
    plt.yticks(fontsize=fz_minor,fontweight='bold')
    # ax2.set_ylim([0,50])
    ax.set_zorder(ax2.get_zorder()+1)
    ax.set_frame_on(False)
    width = 3
    ax2.spines['top'].set_linewidth(width)
    ax2.spines['right'].set_linewidth(width)
    ax2.spines['bottom'].set_linewidth(width)
    ax2.spines['left'].set_linewidth(width)

    fig.savefig(join(save_folder,'TARI_E3_timeseries.png'), dpi=300, bbox_inches='tight')


output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]
# begin_index = dates_E3.index(datetime(2024, 4, 14, 3, 0))
# end_index = dates_E3.index(datetime(2024, 4, 18, 3, 0))
current_index = 44
save_folder = r'D:\R2MSDATA\TARI_E3_test\timelapsed_resistivity_difference'

plot_timeseries(median_RHOA_E3, daily_rainfall, dates_E3, begin_index-2, end_index,current_index)
# %%
print((datetime(2024, 6, 5,11, 0)-datetime(2024, 6,2,3, 0)).total_seconds()/3600)
