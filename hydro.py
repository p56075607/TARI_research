# %% 
# -*- coding: utf-8 -*-
# To read and process the provided hydrological data, which is formatted as CSV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"
from os.path import join
from datetime import timedelta
import matplotlib.dates as mdates
from datetime import datetime
import os

# Function to correct "24:00:00" timestamps
def correct_timestamp(ts):
    if ts.endswith("24:00:00"):
        # Replace "24:00:00" with "00:00:00" and add a day to the date
        corrected_ts = pd.to_datetime(ts.replace("24:00:00", "00:00:00")) + timedelta(days=1)
        return corrected_ts
    else:
        return pd.to_datetime(ts)

def read_hydro_data(data_path):
    df = pd.read_csv(data_path)
    # Pre-process and correct the TIMESTAMP column
    df['TIMESTAMP'] = df['TIMESTAMP'].apply(correct_timestamp)

    # Set the TIMESTAMP column as the DataFrame index
    df.set_index('TIMESTAMP', inplace=True)

    # Resample the data to hourly averages
    hourly_avg = df.resample('H').mean()
    if 'Rain_mm_Tot' in hourly_avg.columns:
        # Summing the rain data to the daily rainfall
        daily_rainfall = hourly_avg['Rain_mm_Tot'].resample('D').sum()
    # Delete the RECORD, BattV column
    hourly_avg.drop('RECORD', axis=1, inplace=True)
    hourly_avg.drop('BattV', axis=1, inplace=True)

    if 'Rain_mm_Tot' in hourly_avg.columns:
        return hourly_avg, daily_rainfall
    else:
        return hourly_avg, None


def check_files_in_directory(directory_path):
    # 存儲解析出來的日期
    dates = []

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(directory_path):
        # 檢查檔案名稱是否符合特定格式
        if filename.endswith('.urf'):
            date_str = filename[:8]  # 提取日期部分
            try:
                # 轉換日期格式從 'YYMMDDHH' 到 datetime 對象
                date = datetime.strptime(date_str, '%y%m%d%H')
                dates.append(date)
            except ValueError:
                # 如果日期格式不正確，忽略此檔案
                continue

    return dates



def plot_hydro_data(hourly_avg, daily_rainfall, plot_target,dates):
    fig, ax1 = plt.subplots(figsize=(20, 8))
    for column in hourly_avg.columns:
        if column in plot_target:
            print(column)
            # ax1.scatter(hourly_avg.index, hourly_avg[column], s=1, marker='o', label=column)
            ax1.plot(hourly_avg.index, hourly_avg[column],linewidth=1, label=column)
            ax1.set_title('Hourly Averages of Hydrological Data and Rainfall')
            ax1.set_xlabel('Time (YYYY-MM-DD)')
            ax1.set_ylabel('Soil Moisture (%)')
            ax1.set_ylim(0, 100)        
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.grid(linestyle='--',linewidth=0.5)
    ax1.plot(dates,[3]*len(dates), 'bo')
    ax1.set_xlim([dates[0]-timedelta(days=1),dates[-1]+timedelta(days=1)])
    # Rotate dates for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for the rotated date labels

    if 'Rain_mm_Tot' in hourly_avg.columns:
        ax2 = ax1.twinx()  # Create a second Y-axis sharing the same X-axis
        ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='Rainfall')
        ax2.set_ylabel('Rainfall (mm)', color='c')  # Set label for the secondary Y-axis
        ax2.tick_params(axis='y', labelcolor='c')  # Set ticks color for the secondary Y-axis

    fig.legend(loc="upper right", bbox_to_anchor=(0.9,0.9))
    return fig


# Read the hydrological data
data_path = join("data","external","農試所水田_1120829.dat")
hourly_avg, daily_rainfall = read_hydro_data(data_path)
dates = check_files_in_directory(r'C:\Users\R2MSDATA\TARI_E1\urf')
plot_target = ['Result_10cm_Avg', 'Result_20cm_Avg', 'Result_30cm_Avg',
    'Result_40cm_Avg', 'Result_50cm_Avg', 'Result_60cm_Avg',
    'Result_80cm_Avg', 'Result_100cm_Avg',
    'Result_150cm_Avg',
       'Result_200cm_Avg', 'Result_300cm_Avg', 'Result_400cm_Avg',
       'Result_500cm_Avg', 'Result_600cm_Avg', 'Result_700cm_Avg',
       'Result_800cm_Avg', 'Result_900cm_Avg']
fig = plot_hydro_data(hourly_avg, daily_rainfall, plot_target,dates)
# fig.savefig(join("output","hydro_data_10.png"),dpi=300, bbox_inches='tight')

# %%
data_path = join("data","external","農試所竹塘站_1120818.dat")
hourly_avg, daily_rainfall = read_hydro_data(data_path)
dates = check_files_in_directory(r'C:\Users\R2MSDATA\TARI_E3\urf')
plot_target = ['Result_10cm_Avg', 'Result_20cm_Avg', 'Result_30cm_Avg',
    'Result_40cm_Avg', 'Result_50cm_Avg', 'Result_60cm_Avg',
    'Result_80cm_Avg', 'Result_100cm_Avg',
    'Result_150cm_Avg',
       'Result_200cm_Avg', 'Result_300cm_Avg', 'Result_400cm_Avg',
       'Result_500cm_Avg', 'Result_600cm_Avg', 'Result_700cm_Avg',
       'Result_800cm_Avg', 'Result_900cm_Avg']
fig = plot_hydro_data(hourly_avg, daily_rainfall, plot_target,dates)

# %%
data_path = join("data","external","農試所旱田_1121222.dat")
hourly_avg, daily_rainfall = read_hydro_data(data_path)
dates = check_files_in_directory(r'C:\Users\R2MSDATA\TARI_E2\urf')
plot_target = ['Result_10cm_Avg', 'Result_20cm_Avg', 'Result_30cm_Avg',
    'Result_40cm_Avg', 'Result_50cm_Avg', 'Result_60cm_Avg',
    'Result_80cm_Avg', 'Result_100cm_Avg']
fig = plot_hydro_data(hourly_avg, daily_rainfall, plot_target,dates)
# %%
# plot_target = ['Result_150cm_Avg',
#        'Result_200cm_Avg', 'Result_300cm_Avg', 'Result_400cm_Avg',
#        'Result_500cm_Avg', 'Result_600cm_Avg', 'Result_700cm_Avg',
#        'Result_800cm_Avg', 'Result_900cm_Avg']
# fig = plot_hydro_data(hourly_avg, daily_rainfall, plot_target)
# fig.savefig(join("output","hydro_data_150.png"),dpi=300, bbox_inches='tight')