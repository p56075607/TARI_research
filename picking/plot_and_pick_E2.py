# %%
import pandas as pd
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft JhengHei'
import matplotlib
matplotlib.use('TkAgg')
import pickle
import matplotlib.dates as mdates
from datetime import datetime
from os import listdir
from os.path import isdir, join
import os
from matplotlib.widgets import Cursor
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import timedelta
from io import BytesIO
import io
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
import sys
from scipy.stats import pearsonr, linregress
from os.path import join
# %%
# Import ERT data
# read the dates and median_RHOA_E1 from the pickle file
with open(join(r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor','median_RHOA_E2_and_date.pkl'), 'rb') as f:
    pickled_dates_E1 = pickle.load(f)
    pickled_median_RHOA_E1 = pickle.load(f)

dates = pickled_dates_E1
median_RHOA = pickled_median_RHOA_E1
# %%
rain_df = pd.read_csv(r'C:\Users\Git\TARI_research\data\external\G2F820\merged_data.csv')
rain_df['Time'] = pd.to_datetime(rain_df['Time'])
rain_df.set_index('Time', inplace=True)
daily_rainfall = rain_df['Precp']
# %%
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

def onclick(event):
    if event.inaxes == ax:  # 確保點擊發生在 ax 上
        if plt.get_current_fig_manager().toolbar.mode != '':
            return
        x_click = event.xdata
        y_click = event.ydata
        # 設置一個合理的距離閾值
        threshold_x = 0.01
        threshold_y = 0.1
        for x, y in zip(mdates.date2num(dates), median_RHOA):
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
    return selected_points

fig, ax = plt.subplots(figsize=(25, 8))

median_rhoa_plot, = ax.plot(dates, median_RHOA, 'ro',markersize=3 ,zorder=2)
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

begin_index = 1#dates_E1.index(datetime(2024, 3, 18, 1, 0))
end_index = -1#dates_E1.index(datetime(2024, 6, 6, 21, 0))
ax.set_xlim(dates[begin_index], dates[end_index])
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
fz_major = 30
# ax.set_title('Median Apparent Resistivity',fontsize=fz_major,fontweight='bold')
ax.set_xlabel('Date',fontsize=fz_major)
ax.set_ylabel('Apparent Resistivity ($\Omega m$)',fontsize=fz_major)

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

#使用 matplotlib.widgets.Cursor 來顯示游標
cursor = Cursor(ax, useblit=True, color='gray', linewidth=1)
# 檢查 QApplication 是否已初始化
app = QApplication.instance()
if app is None:
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
fig.canvas.mpl_connect('motion_notify_event', motion_hover)

# 啟用滾輪縮放功能
fig.canvas.mpl_connect('scroll_event', on_scroll)

# 連接點擊事件
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
# %%
df_RHOA = pd.DataFrame({'date': dates, 'RHOA': median_RHOA})
df_RHOA.set_index('date', inplace=True)
# %%
# Read x,y from csv
df = pd.read_csv(r'C:\Users\Git\TARI_research\picking\E2.csv')
df['x'] = pd.to_datetime(df['x'])
# df.set_index('x', inplace=True)
# fig, ax = plt.subplots(figsize=(5, 5))
rhoa_ini = []
rhoa_sat = []
x1s = []
slopes = []
maxx = []
for i in range(len(df)):
    if i%3 == 0:
        rhoa_ini.append(df['y'].iloc[i])
    elif i%3 == 1:
        rhoa_sat.append(df['y'].iloc[i])
        x1 = df['x'].iloc[i]
        x1s.append(x1)
    elif i%3 == 2:
        x2 = df['x'].iloc[i]
        # find df_RHOA['RHOA'] from x1 to x2
        time_seg = df_RHOA['RHOA'].iloc[df_RHOA.index.get_loc(x1):df_RHOA.index.get_loc(x2)+1].index
        x = (time_seg - time_seg.min()).total_seconds() / 3600
        maxx.append(max(x))
        y = np.log10(df_RHOA['RHOA'].iloc[df_RHOA.index.get_loc(x1):df_RHOA.index.get_loc(x2)+1].values)
        coefficients = np.polyfit(x, y, 1)
        slope, intercept = coefficients
        slopes.append(slope)
#         ax.plot(x,y-y[0])
# plt.show()
# %%
df_picking = pd.DataFrame({'xs': x1s, 'rhoa_ini': rhoa_ini, 'rhoa_sat': rhoa_sat,
                           'diff_rhoa': np.array(rhoa_ini)-np.array(rhoa_sat), 
                           'xrange':maxx,'slope': slopes})
# df_picking = df_picking[df_picking['xrange']>24*3]
plt.rcParams['font.family'] = 'Microsoft YaHei'
# plot the picking slope histogram
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df_picking['xs'],df_picking['slope'],'o',color='k',markersize=10)
fz_minor = 25
plt.yticks(fontsize=fz_minor,fontweight='bold')
plt.xticks(fontsize=fz_minor,rotation=45, ha='right', rotation_mode='anchor',fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax.yaxis.get_offset_text().set_fontsize(fz_minor)
ax2 = ax.twinx()
ax2.plot(df_RHOA.index, np.log10(df_RHOA['RHOA']), 'ro',markersize=3 ,zorder=2)
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
fontsize = 20
ax.set_ylabel('乾燥斜率'+'\n'+r'($\Delta log(\rho_a)/\Delta hr$)', fontsize=fontsize+5,fontweight='bold')
ax2.set_ylabel('視電阻率'+'\n'+r'$log(\rho_a)$', fontsize=fontsize+5,fontweight='bold')
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
# plt.show()
fig.savefig(r'C:\Users\Git\TARI_research\picking\E2_picking.png', dpi=300, bbox_inches='tight')

# %%
# # plot the picking slope histogram
# fig, ax = plt.subplots(figsize=(5, 5))

# # 假設 df_picking 是您的資料框
# x = df_picking['slope']
# y = df_picking['rhoa_sat']

# # 計算皮爾森相關係數和 p 值
# r, p_value = pearsonr(x, np.log10(y))

# print(f"皮爾森相關係數 (r): {r}")
# print(f"p 值: {p_value}")
# ax.scatter(df_picking['slope'],np.log10(df_picking['rhoa_sat']))
# ax.set_title('Correlation coefficient:{:.2f}, p values:{:.3f}'.format(r, p_value))
# ax.set_xlabel(r'Drying slope ($\Delta log(\rho_a)/\Delta hr$)')
# ax.set_ylabel(r'Wet state $log(\rho_a)$')
# ax.grid(True, which='major', linestyle='--', linewidth=0.5)
# plt.show()
df_picking.to_csv(r'C:\Users\Git\TARI_research\picking\E2_picking.csv', index=False)
# %%
df_picking_E1 = pd.read_csv(r'C:\Users\Git\TARI_research\picking\E1_picking.csv')
df_picking_E2 = pd.read_csv(r'C:\Users\Git\TARI_research\picking\E2_picking.csv')
df_picking_E3 = pd.read_csv(r'C:\Users\Git\TARI_research\picking\E3_picking.csv')
# merge 
# df_picking = pd.concat([df_picking_E1, df_picking_E2], axis=0)
# plot the picking slope histogram
fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams['font.family'] = 'Microsoft YaHei'

def linear_regression(x, y,data_name):
    # 計算線性回歸
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # 計算 CR 值
    r, p_value = pearsonr(x, np.log10(y))
    ax.scatter(x,y,label=data_name,edgecolors='k')
    y_pred = intercept + slope * x
    ax.plot(x, y_pred,alpha=1,linewidth=2, label='相關係數:{:.2f}'.format(r))
    # ax.text(max(x),min(y) ,f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.2f}', fontsize=12)
    
    return slope, intercept


# E1 
slope, intercept = linear_regression( np.log10(df_picking_E1['rhoa_sat']),df_picking_E1['slope'],'水田')
print(slope)
# E2
slope, intercept = linear_regression( np.log10(df_picking_E2['rhoa_sat']),df_picking_E2['slope'],'旱田')
print(slope)
# E3
# slope, intercept = linear_regression(np.log10(df_picking_E3['rhoa_sat']), df_picking_E3['slope'], '竹塘')
# print(slope)
# ax.semilogy(df_picking['slope'], df_picking['rhoa_sat'], 'o', color='k', markersize=10)
fontsize = 25
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax.yaxis.get_offset_text().set_fontsize(fz_minor)
ax.set_ylabel(r'乾燥斜率 ($\Delta log(\rho_a)/\Delta hr$)', fontsize=fontsize,fontweight='bold')
ax.set_xlabel(r'電阻率 $log(\rho_a)$', fontsize=fontsize,fontweight='bold')
font = matplotlib.font_manager.FontProperties(size=fontsize-5, weight='bold')
ax.legend( prop=font)
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)
ax.grid(True, which='major', linestyle='--', linewidth=1)
plt.yticks(fontsize=fz_minor,fontweight='bold')
plt.xticks(fontsize=fz_minor,fontweight='bold')
plt.show()
fig.savefig(join('drying_slope_vs_wet_state.png'), dpi=300, bbox_inches='tight')
# %%
t_max = 1000
t = np.linspace(0, t_max, 100)
K = (slope*np.log10(np.min(rhoa_sat))+intercept)
rho = (K*np.exp(slope*t)-intercept)/slope

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(t,rho, '-', color='k', markersize=10)
ax.set_xlabel('Time (hr)')
ax.set_ylabel(r'Apparent Resistivity $log(\rho_a)$')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.show()
fig.savefig(join('drying_curve_E2.png'), dpi=300, bbox_inches='tight')
# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rho[1:], np.diff(rho)/np.diff(t), '-', color='k', markersize=10)
plt.show()
# %%
rho_min = np.log10(np.min(rhoa_sat))
rho_max = np.log10(np.max(rhoa_ini))
rho = np.linspace(rho_min, rho_max, 100)
K = (slope*np.log10(np.min(rhoa_sat))+intercept)
t_drain = np.log((slope*rho+intercept)/K)/slope

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rho,t_drain, '-', color='k', markersize=10)
ax.set_ylabel('Time to drain (hr)')
ax.set_xlabel(r'Apparent Resistivity $log(\rho_a)$')

theta_c = np.exp(((10**rho_min)-1464.81)/-348.74)
theta =   abs(np.exp(((10**rho)-1464.81)/-348.74)-theta_c)
ax2 = ax.twinx()
ax2.plot(rho,theta, '-', color='b', markersize=10)
ax2.set_ylabel('Water needed (%)',color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.show()
fig.savefig(join('drying_X_curve_E2.png'), dpi=300, bbox_inches='tight')
# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(rho,(90*24/t_drain)*theta, '-', color='k', markersize=10)
ax.set_ylabel('Total Water Usage (mm)')
ax.set_xlabel(r'Apparent Resistivity $log(\rho_a)$')
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.show()
fig.savefig(join('demand_water_curve_E2.png'), dpi=300, bbox_inches='tight')