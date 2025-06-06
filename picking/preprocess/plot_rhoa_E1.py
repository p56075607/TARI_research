# %%
# Import moduals
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from inverison_util import convertURF
from ridx_analyse import ridx_analyse
from urf2ohm import urf2ohm

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"
# %config InlineBackend.figure_format='svg' # Setting figure format for this notebook
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import pygimli.meshtools as mt
from datetime import datetime
from os.path import join
from os import listdir
from datetime import timedelta
import matplotlib.dates as mdates
import pickle

from datetime import datetime
from collections import Counter

urf_path = r'D:\R2MSDATA_2024\TARI_E1_test\urf'
urffiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.urf')])
ohmfiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.ohm')])

def get_datetime_list_and_count(directory):
    # Initialize an empty list to store datetime objects
    datetime_list = []
    urffiles = sorted([_ for _ in listdir(directory) if _.endswith('.urf')])
    # Loop through all files in the directory
    for filename in urffiles:
        # Check if the filename matches the expected format
        if len(filename) > 8 and filename[:8].isdigit():
            # Extract the date-time part from the filename
            date_time_str = filename[:8]
            # Convert the string to a datetime object
            date_time_obj = datetime.strptime(date_time_str, '%y%m%d%H')
            # Append the datetime object to the list
            datetime_list.append(date_time_obj)

    # Count the occurrence of each date
    date_count = Counter([dt.date() for dt in datetime_list])

    return datetime_list, date_count

dates, date_count = get_datetime_list_and_count(urf_path)
print(len(date_count))
# %%
ridx_urf_path = r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor\E1_check\urf_E1_ridx'
unsorted_quality_info = ridx_analyse(ridx_urf_path, formula_choose='C')

ridx = unsorted_quality_info/100
rest = 50000
t3 = np.argsort(ridx)[rest:]
remove_index = np.full((len(unsorted_quality_info)), False)
for i in range(len(t3)):
    remove_index[t3[i]] = True

# %%
def read_ohm_r_column(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the number of electrodes and their positions
    num_electrodes = int(lines[0].strip().split()[0])
    electrodes = []
    idx = 2  # Start reading electrode positions after the header line

    # Skip electrode positions
    while len(electrodes) < num_electrodes and idx < len(lines):
        line = lines[idx].strip()
        if line and not line.startswith('#'):
            electrodes.append([float(x) for x in line.split()])
        idx += 1

    # Extract the number of data points
    while idx < len(lines) and not lines[idx].strip().split()[0].isdigit():
        idx += 1

    if idx >= len(lines):
        raise ValueError("Data points line not found")

    num_data = int(lines[idx].strip().split()[0])
    idx += 2  # Skip the data count line and the header line

    r_values = []
    while len(r_values) < num_data and idx < len(lines):
        line = lines[idx].strip()
        if line and not line.startswith('#'):
            data = line.split()
            r_values.append(float(data[4]))  # Assuming 'r' is the 5th column
        idx += 1

    if len(r_values) != num_data:
        raise ValueError("Mismatch between number of data points and extracted data")

    return r_values

median_RHOA = []
Q1_RHOA = []
Q3_RHOA = []
for i,urf_file_name in enumerate(urffiles):
    if urf_file_name[:-4]+'.ohm' in ohmfiles: # 檢查是否有 ohm 檔案，若有就視為做過不跑反演
        print(urf_file_name[:-4]+'.urf is already processed. Skip it!')
        ohm_file_name = join(urf_path,urf_file_name[:-4]+'.ohm')
    else:
        print('Processing: '+urf_file_name)
        ohm_file_name = urf2ohm(join(urf_path,urf_file_name),has_trn = False)

    if i == 0:
        data = ert.load(ohm_file_name)
        data.remove(remove_index)
        print(data) 
        data['k'] = ert.createGeometricFactors(data, numerical=True) # 以數值方法計算幾何因子，較耗費電腦資源
        rhoa = data['k'] * data['r']

    else:
        r_values = read_ohm_r_column(ohm_file_name)
        removed_r = np.delete(r_values, t3)
        rhoa = data['k'] * removed_r
    # DATA.append(data)
    median_RHOA.append(np.median(rhoa))
    Q1_RHOA.append(np.percentile(rhoa, 25))
    Q3_RHOA.append(np.percentile(rhoa, 75))

# %%
import winsound
duration = 10000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
# %%
# Save the processed data including the median apparent resistivity values and the corresponding dates
with open('median_RHOA_E1_and_date.pkl', 'wb') as f:
    pickle.dump(dates, f)
    pickle.dump(median_RHOA, f)
    pickle.dump(Q1_RHOA, f)
    pickle.dump(Q3_RHOA, f)

# %%
# # Load the processed data
# with open('median_RHOA_E1_and_date.pkl', 'rb') as f:
#     read_dates = pickle.load(f)
#     read_median_RHOA = pickle.load(f)
# # %%
# import pandas as pd
# def read_hydro_data(data_path):
#     df = pd.read_excel(data_path, sheet_name='農試所(霧峰)雨量資料')

#     # 將'TIME'列轉換為日期時間格式
#     df['TIME'] = pd.to_datetime(df['TIME'])
#     df.set_index('TIME', inplace=True)

#     # 將'Rain(mm)'列轉換為數字，無法轉換的設置為NaN，然後丟棄NaN值
#     df['Rain(mm)'] = pd.to_numeric(df['Rain(mm)'], errors='coerce')
#     df.dropna(subset=['Rain(mm)'], inplace=True)
    
#     daily_rainfall = df['Rain(mm)']
#     return daily_rainfall

# daily_rainfall = read_hydro_data(r'水文站資料彙整_20240731.xlsx')
# # %%
# # plt.plot(dates,mean_rhoa,'o',markersize=1)
# fig, ax1 = plt.subplots(figsize=(25, 6))
# ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
# # ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


# ax1.plot(dates,median_RHOA, 'bo',markersize=1.5)
# # ax1.set_xlim(dates[dates.index(datetime(2024,2,29,0,0))], dates[-1])
# ax1.set_xlim(dates[0], dates[-1])
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Median apparent resistivity ($\Omega m$)'  )
# ax1.grid(linestyle='--',linewidth=0.5)
# # Rotate dates for better readability
# plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
# plt.tight_layout()  # Adjust layout to make room for the rotated date labels
# ax2 = ax1.twinx()  # Create a second Y-axis sharing the same X-axis
# ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='Rainfall')
# ax2.set_ylabel('Rainfall (mm)', color='c')  # Set label for the secondary Y-axis
# ax2.tick_params(axis='y', labelcolor='c')  # Set ticks color for the secondary Y-axis
# ax2.set_ylim(0, daily_rainfall.max())  # Set limits for the secondary Y-axis
# # %%
