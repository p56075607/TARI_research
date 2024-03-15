# %% To read and process the provided hydrological data, which is formatted as CSV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from datetime import timedelta
import matplotlib.dates as mdates

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

    return hourly_avg, daily_rainfall

def plot_hydro_data(hourly_avg, daily_rainfall, plot_target):
    fig, ax1 = plt.subplots(figsize=(20, 8))
    for column in hourly_avg.columns:
        if column in plot_target:
            print(column)
            # ax1.scatter(hourly_avg.index, hourly_avg[column], s=1, marker='o', label=column)
            ax1.plot(hourly_avg.index, hourly_avg[column],linewidth=1, label=column)
            ax1.set_title('Hourly Averages of Hydrological Data and Rainfall')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Average Value')
            ax1.set_ylim(0, 100)        
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.grid(linestyle='--',linewidth=0.5)
    # Rotate dates for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for the rotated date labels

    if 'Rain_mm_Tot' in hourly_avg.columns:
        ax2 = ax1.twinx()  # Create a second Y-axis sharing the same X-axis
        ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='Rainfall')
        ax2.set_ylabel('Rainfall (mm)', color='c')  # Set label for the secondary Y-axis
        ax2.tick_params(axis='y', labelcolor='c')  # Set ticks color for the secondary Y-axis

    fig.legend(loc="upper right", bbox_to_anchor=(0.9,0.9))

# %%
# Read the CHUT hydrological data
data_path = join("data","external","CHUT_1120818.dat")
hourly_avg, daily_rainfall = read_hydro_data(data_path)
plot_target = ['Result_10cm_Avg', 'Result_20cm_Avg', 'Result_30cm_Avg',
    'Result_40cm_Avg', 'Result_50cm_Avg', 'Result_60cm_Avg',
    'Result_80cm_Avg', 'Result_100cm_Avg']
plot_hydro_data(hourly_avg, daily_rainfall, plot_target)
# %%
plot_target = ['Result_150cm_Avg',
       'Result_200cm_Avg', 'Result_300cm_Avg', 'Result_400cm_Avg',
       'Result_500cm_Avg', 'Result_600cm_Avg', 'Result_700cm_Avg',
       'Result_800cm_Avg', 'Result_900cm_Avg']
plot_hydro_data(hourly_avg, daily_rainfall, plot_target)