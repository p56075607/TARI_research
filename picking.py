# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isdir, join
from datetime import datetime
import pygimli as pg
from scipy.optimize import curve_fit
from pygimli.physics import ert  # the module
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
# %%
# input csv file 
df = pd.read_csv("data\external\水田_1112.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
df.set_index('Timestamp', inplace=True)
df_resampled = df.resample('H').mean()
# export to csv file
df_resampled.to_csv('data\external\水田_1112_hourly.csv')
# %%
fig, ax = plt.subplots(figsize=(12, 6))
# relative change with respect to the first data point
# index = df.index.get_loc(datetime(2024, 10, 31, 12, 0))

ax.plot(df.index, (df['10cm'] ), label='10cm')
ax.plot(df.index, (df['50cm'] ), label='50cm')
ax.plot(df.index, (df['100cm']), label='100cm')
ax.set_xlim(datetime(2024, 10, 8, 0, 0), datetime(2024, 10, 9, 0, 0))
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
xticklabels = ax.get_xticklabels()
ax.set_xticklabels(xticklabels,rotation = 45, ha='right',rotation_mode="anchor")
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
ax.legend()
# save png with transparent background
# fig.savefig('data\external\旱田_1112_hourly.png', dpi=300, bbox_inches='tight', transparent=True)


















# %%
# input csv file 
df = pd.read_csv("data\external\旱田_1112.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
df.set_index('Timestamp', inplace=True)
df_resampled = df.resample('H').mean()
# export to csv file
df_resampled.to_csv('data\external\旱田_1112_hourly.csv')
# %%
fig, ax = plt.subplots(figsize=(12, 6))
# relative change with respect to the first data point
index = df.index.get_loc(datetime(2024, 10, 31, 12, 0))

ax.plot(df.index, (df['10cm'] ), label='10cm')
ax.plot(df.index, (df['50cm'] ), label='50cm')
ax.plot(df.index, (df['100cm']), label='100cm')
ax.set_xlim(datetime(2024, 10, 31, 12, 0), datetime(2024, 11, 2, 0, 0))
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
xticklabels = ax.get_xticklabels()
ax.set_xticklabels(xticklabels,rotation = 45, ha='right',rotation_mode="anchor")
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
ax.legend()
# save png with transparent background
fig.savefig('data\external\旱田_1112_hourly.png', dpi=300, bbox_inches='tight', transparent=True)
# %%
# 然後再執行 .dt 存取器來轉換為 datetime list
df = pd.read_csv("data\external\旱田_1112_hourly.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
time_list = df['Timestamp'].dt.to_pydatetime().tolist()

# Define depths and interpolate the data
depths = [-1, -0.5, -0.1]  # Original depths in m
interp_depths = np.linspace(-1, -0.1, 50)  # Interpolated depths in m

# Interpolating data along depth axis for each time point
interpolated_values = np.array([
    np.interp(interp_depths, depths, row) for row in df[['100cm', '50cm', '10cm']].values
])

# Prepare the meshgrid again with the new datetime list
X, Y = np.meshgrid(time_list, interp_depths)

# Plotting the contour
fig, ax = plt.subplots(figsize=(12, 6))
contour = ax.contour(X, Y, interpolated_values.T, levels=20, colors='black', linewidths=0.8)
ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f")  # Label the contour lines with values





# Formatting the plot
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)
ax.set_xlabel('Time')
ax.set_ylabel('Depth (cm)')
ax.set_title('Contour Plot of Values Over Time and Depth')
# set xlim from 10/31 12:00 to 11/5 7:00
ax.set_xlim(mdates.date2num(datetime(2024, 10, 31, 12, 0)), mdates.date2num(datetime(2024, 11, 5, 7, 0)))

# %%
# export X, Y, interpolated_values to pkl
import pickle
with open('interpolated_values.pkl', 'wb') as f:
    pickle.dump([X, Y, interpolated_values], f)
    f.close()

# %%
# load interpolated_values.pkl
with open('interpolated_values.pkl', 'rb') as f:
    X, Y, interpolated_values = pickle.load(f)