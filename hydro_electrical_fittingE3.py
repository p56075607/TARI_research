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
from datetime import datetime, timedelta
# %%
# input csv file 
df = pd.read_csv("data\external\竹塘水田.csv")

# plot the data x: "Timestamp", y:10cm	50cm	100cm
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["100cm"])

# %%
# aveage the data hourly
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.set_index("Timestamp", inplace=True)
df = df.resample("h").mean()
df['date_time'] = df.index
# average 10cm	50cm	100cm data to df['mean_1m']
df["mean_1m"] = df["100cm"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["mean_1m"], label="mean_1m")

# %%
# urf_path = join(r'D:\R2MSDATA\TARI_E3_test','urf')
# ohmfiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.ohm')])

# all_data = []
# dates = []
# for i,output_folder_name in enumerate(ohmfiles):
#     print(output_folder_name)
#     data = pg.load(join(urf_path, output_folder_name))
#     data.remove(~((data['a']==57)*(data['b']==54)*(data['m']==55)*(data['n']==56)))
#     if i == 0:
#         k = ert.createGeometricFactors(data, numerical=True)
#     data['rhoa'] = k * data['r']
#     all_data.append(data['rhoa'][0])
#     dates.append(pd.to_datetime(datetime.strptime(output_folder_name[:8] , '%y%m%d%H')))

# save all_data and dates to a pickle file

# with open(join('picking', 'RHOA_E3_local.pickle'), 'wb') as f:
#     pickle.dump([all_data, dates], f)
# load all_data and dates from a pickle file
# with open(join('picking', 'RHOA_E3_local.pickle'), 'rb') as f:
#     all_data, dates = pickle.load(f)

# %%
import pickle
with open(join(r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor','median_RHOA_E3_and_date.pkl'), 'rb') as f:
    dates = pickle.load(f)
    all_data = pickle.load(f)
# %%
# rho_100_cm = []
# mesh_filter = (para_domain.cellCenters()[:,1]>-1)
# for i,output_folder_name in enumerate(output_folders):
#     rho_100_cm.append(np.median(all_mgrs[i]['model'][mesh_filter]))


# Extract df data from dates
filterd_hydro_data = pd.DataFrame( columns=df.columns )
for i in range(len(dates)):
    mask = (df['date_time'] == dates[i])
    filterd_hydro_data = pd.concat([filterd_hydro_data, df.loc[mask]])


# find index of filterd_hydro_data['date_time'] in dates
index = []
for i in range(len(filterd_hydro_data)):
    index.append(dates.index(filterd_hydro_data['date_time'][i]))
# extract rho_100_cm from df['date_time']
rhoa = [all_data[i] for i in index]
filterd_hydro_data['rhoa'] = rhoa

# remove rhoa < 10
filterd_hydro_data = filterd_hydro_data[filterd_hydro_data['rhoa'] > 10]

# filterd_hydro_data = pd.read_csv('filterd_hydro_data_E3.csv')
filterd_hydro_data['date_time'] = pd.to_datetime(filterd_hydro_data['date_time'])
# extract filterd_hydro_data from 2024/6/26 to 2024/7/13
filterd_hydro_data = filterd_hydro_data[(filterd_hydro_data['date_time'] > '2024-06-08 00:00:00')]
filterd_hydro_data = filterd_hydro_data[(filterd_hydro_data['date_time'] < '2024-07-05 00:00:00')]
x_data = filterd_hydro_data['mean_1m']
y_data = filterd_hydro_data['rhoa']


fig, ax = plt.subplots(figsize=(21,8))
ax.scatter(filterd_hydro_data['date_time'], filterd_hydro_data['mean_1m'],s=2, label="mean_1m")
ax2 = ax.twinx()
ax2.scatter(filterd_hydro_data['date_time'], filterd_hydro_data['rhoa'],c='r',s=2, label="RHOA")  
ax.grid(True, which='major', linestyle='--', linewidth=0.5)

# Extracting the relevant columns


# Defining the logarithmic model y = a * ln(x) + b
def log_model(x, a, b):
    return a * np.log(x) + b

# Curve fitting
params, params_covariance = curve_fit(log_model, x_data, y_data,maxfev = 100000)
a, b = params

# Predicted values
xlim = [10,40]
x_fit = np.linspace(xlim[0],xlim[1], 100)
y_fit = log_model(x_fit, a, b)

# Calculating the residuals for R-squared
residuals = y_data - log_model(x_data, a, b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)

# Calculating the confidence interval (±5%)
y_fit_upper = y_fit * 1.05
y_fit_lower = y_fit * 0.95

#  Plotting
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(x_data, y_data, c='black'#np.linspace(1,len(filterd_hydro_data),len(filterd_hydro_data))#
           , label='Data points',s=3)
ax.plot(x_fit, y_fit, color='red', label=f'Fit: y = {a:.2f}ln(x) + {b:.2f}')
ax.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.1, label='±5% Confidence Interval')
# ax.set_xlim(xlim)
# # ylim = [14, 27]
# ax.set_ylim(ylim)

fontsize=18
# # Adding equation and R-squared to the plot
plt.text(xlim[0],27 ,f'    $y = {a:.2f} \ln(x) + {b:.2f}$\n    $R^2 = {r_squared:.2f}$\n', fontsize=fontsize)

# Labels and legend
plt.xlabel('Water content (%)',fontsize=fontsize,fontweight='bold')
plt.ylabel('Apparent Resistivity (ohm-m)',fontsize=fontsize,fontweight='bold')
# plt.title('Logarithmic Fit with Confidence Interval',fontsize=fontsize,fontweight='bold')
plt.legend(fontsize=fontsize)
ax.grid(linestyle='--',color='gray',linewidth=0.5,alpha = 0.5)
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)
plt.xticks(fontsize=fontsize,fontweight='bold')
plt.yticks(fontsize=fontsize,fontweight='bold')
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
# plt.show()

# export filterd_hydro_data to csv
# filterd_hydro_data.to_csv('filterd_hydro_data.csv', index=False, columns= ['date_time', 'mean_1m', 'rho_100_cm'])
# %%
date_start = '2024-06-02 01:00:00'#'2024-05-13 03:00:00'
date_end  =  '2024-06-08 16:00:00'#'2024-05-16 11:00:00'
filterd_data = filterd_hydro_data[(filterd_hydro_data['date_time'] >= date_start)]
filterd_data =       filterd_data[(filterd_hydro_data['date_time'] <= date_end  )]
SWC_diff = []
def log_model_inverse(y, a, b):
    return np.exp((y - b) / a)

for j in range(len(filterd_data)-1):
    diff = (log_model_inverse(filterd_data['rhoa'][j+1], a, b) - log_model_inverse(filterd_data['rhoa'][0], a, b))/100

    SWC_diff.append(diff)

# Plot water_m2 time-series
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(filterd_data['date_time'][1:], SWC_diff, '-o',color='k', linewidth=3, markersize=10, label='ERT')

index = filterd_data['date_time'] == datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')
filterd_data['SWChange_hydro'] = (filterd_data['mean_1m']-list(filterd_data['mean_1m'][index])[0])/100
ax.plot(filterd_data['date_time'], filterd_data['SWChange_hydro'], 
        '-or', linewidth=3, markersize=10, label='Contact Sensor')
ax.legend(fontsize=20)
# ax.set_ylim([0,0.06])
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
# set xy ticks label fontsize 
fz_minor = 30
plt.xticks(fontsize=fz_minor,rotation=45, ha='right', rotation_mode='anchor',fontweight='bold')
plt.yticks(fontsize=fz_minor,fontweight='bold')

ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
ax.set_xlabel('Date (2024/mm/dd)', fontsize=fz_minor, fontweight='bold')
ax.set_ylabel('Water Change ($m^3$)', fontsize=fz_minor, fontweight='bold')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
ax.grid(True, which='major', linestyle='-', linewidth=1)

# Extract filterd_hydro_data['SWChange_hydro'] from 2024/10/10 10:00 to 2024/10/14 14:00
start_time = pd.Timestamp('2024-06-03 11:00')
end_time =   pd.Timestamp('2024-06-18 13:00')
# start_time = pd.Timestamp('2024-11-02 06:00')
# end_time = pd.Timestamp('2024-11-07 00:00')
filtered_df = filterd_data[(filterd_data['date_time'] >= start_time) & (filterd_data['date_time'] <= end_time)]
filtered_df['time_delta'] = (filterd_data['date_time'] - filterd_data['date_time'].min()).dt.total_seconds() / 3600
x = filtered_df['time_delta'].values
y = filtered_df['SWChange_hydro'].values
coefficients = np.polyfit(x, y, 1)
slope, intercept = coefficients
# Define the start datetime
start_datetime = datetime.strptime('2024-06-03 11:00', '%Y-%m-%d %H:%M')
# start_datetime = datetime.strptime('2024/11/02 06:00', '%Y/%m/%d %H:%M')
# Initialize an empty list to store the datetime objects
THMC_time = []
num = 60
x_delta = np.linspace(0,2*(num-1),num)

# Loop to generate 200 datetime objects with 1-hour intervals
for i in range(len(x_delta)):
    # Append the current datetime to the list
    THMC_time.append(start_datetime + timedelta(hours=i*2))
ax.plot(THMC_time, slope * x_delta + intercept, 
        '--k',alpha=0.2, linewidth=3, markersize=10, label='Water Change from Contact Sensor')