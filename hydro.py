# %% To read and process the provided hydrological data, which is formatted as CSV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from datetime import timedelta

# Function to correct "24:00:00" timestamps
def correct_timestamp(ts):
    if ts.endswith("24:00:00"):
        # Replace "24:00:00" with "00:00:00" and add a day to the date
        corrected_ts = pd.to_datetime(ts.replace("24:00:00", "00:00:00")) + timedelta(days=1)
        return corrected_ts
    else:
        return pd.to_datetime(ts)
# %%
data_path = join("data","external","TARIwet_1120829.dat")
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
# %%
# Plot setup
fig, ax1 = plt.subplots(figsize=(20, 8))
# Plotting other columns on the primary Y-axis
for column in hourly_avg.columns:
    if column != 'Rain_mm_Tot':
        ax1.scatter(hourly_avg.index, hourly_avg[column], s=1, marker='o', label=column)


if 'Rain_mm_Tot' in hourly_avg.columns:
    ax2 = ax1.twinx()  # Create a second Y-axis sharing the same X-axis
    ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='Rainfall')
    ax2.set_ylabel('Rainfall (mm)', color='c')  # Set label for the secondary Y-axis
    ax2.tick_params(axis='y', labelcolor='c')  # Set ticks color for the secondary Y-axis

# Set x-axis one week interval
ax1.xaxis.set_major_locator(plt.MaxNLocator(35))
ax1.set_title('Hourly Averages of Hydrological Data and Rainfall')
ax1.set_xlabel('Time')
ax1.set_ylabel('Average Value')
ax1.set_ylim(0, 100)  # Set y-axis limits for the primary Y-axis
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()