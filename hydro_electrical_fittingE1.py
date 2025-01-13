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
# %%
# # input csv file 
# df = pd.read_csv("data\external\水田_1112.csv")

# # plot the data x: "Timestamp", y:10cm	50cm	100cm
# fig, ax = plt.subplots()
# ax.plot(df.index, df["10cm"], label="10cm")
# ax.plot(df.index, df["50cm"], label="50cm")
# ax.plot(df.index, df["100cm"], label="100cm")

# # %%
# # aveage the data hourly
# df["Timestamp"] = pd.to_datetime(df["Timestamp"])
# df.set_index("Timestamp", inplace=True)
# df = df.resample("h").mean()
# df['date_time'] = df.index
# # average 10cm	50cm	100cm data to df['mean_1m']
# df["mean_1m"] = (df["10cm"] + df["50cm"] + df["100cm"]) / 3
# fig, ax = plt.subplots()
# ax.plot(df.index, df["mean_1m"], label="mean_1m")

# # %%
# urf_path = join(r'D:\R2MSDATA\TARI_E1_test','urf')
# ohmfiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.ohm')])

# all_data = []
# dates = []
# for i,output_folder_name in enumerate(ohmfiles):
#     print(output_folder_name)
#     data = pg.load(join(urf_path, output_folder_name))
#     data.remove(~((data['a']==18)*(data['b']==15)*(data['m']==16)*(data['n']==17)))
#     if i == 0:
#         k = ert.createGeometricFactors(data, numerical=True)
#     data['rhoa'] = k * data['r']
#     all_data.append(data['rhoa'][0])
#     dates.append(pd.to_datetime(datetime.strptime(output_folder_name[:8] , '%y%m%d%H')))
# # load mesh geometry
# # para_domain = pg.load(join(output_path,output_folder_name,'ERTManager','resistivity-pd.bms'))
# # %%

# # %%
# # rho_100_cm = []
# # mesh_filter = (para_domain.cellCenters()[:,1]>-1)
# # for i,output_folder_name in enumerate(output_folders):
# #     rho_100_cm.append(np.median(all_mgrs[i]['model'][mesh_filter]))


# # Extract df data from dates
# filterd_hydro_data = pd.DataFrame( columns=df.columns )
# for i in range(len(ohmfiles)):
#     mask = (df['date_time'] == dates[i])
#     filterd_hydro_data = pd.concat([filterd_hydro_data, df.loc[mask]])


# # find index of filterd_hydro_data['date_time'] in dates
# index = []
# for i in range(len(filterd_hydro_data)):
#     index.append(dates.index(filterd_hydro_data['date_time'][i]))
# # extract rho_100_cm from df['date_time']
# rhoa = [all_data[i] for i in index]
# filterd_hydro_data['rhoa'] = rhoa
# %%
filterd_hydro_data = pd.read_csv('filterd_hydro_data_E1.csv')
# Extracting the relevant columns
x_data = filterd_hydro_data['mean_1m']
y_data = filterd_hydro_data['rhoa']

# Defining the logarithmic model y = a * ln(x) + b
def log_model(x, a, b):
    return a * np.log(x) + b

def log_model_inverse(y, a, b):
    return np.exp((y - b) / a)
# Curve fitting
params, params_covariance = curve_fit(log_model, x_data, y_data)
a, b = params

# Predicted values
xlim = [10,45]
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
ax.scatter(x_data, y_data, color='black', label='Data points',s=3)
ax.plot(x_fit, y_fit, color='red', label=f'Fit: y = {a:.2f}ln(x) + {b:.2f}')
ax.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.1, label='±5% Confidence Interval')
ax.set_xlim(xlim)
ylim = [20, 150]
ax.set_ylim(ylim)

fontsize=18
# Adding equation and R-squared to the plot
plt.text(xlim[0],ylim[0] ,f'    $y = {a:.2f} \ln(x) + {b:.2f}$\n    $R^2 = {r_squared:.2f}$\n', fontsize=fontsize)

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

# %%
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

    mgr_dict = {'Name': save_ph.split('\\')[-1],
                'paraDomain': para_domain, 
                'data': data, 
                # 'response': response, 
                'model': model, 'coverage': coverage, 
                'investg_depth': investg_depth, 
                'rrms': rrms, 'chi2': chi2, 'lam': lam,
                'rrmsHistory': rrmsHistory, 'chi2History': chi2History}

    return mgr_dict

output_path = r'D:\R2MSDATA\TARI_E1_test\output_second_inversion'
output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]
# print(output_folders) # ['24022917_m_E1', '24022921_m_E1', '24030215_m_E1', '24030221_m_E1',...]
all_mgrs = []
begin_index = output_folders.index('24100806_m_E1')
end_index = output_folders.index('24101500_m_E1')
for j in range(begin_index,end_index+1,1):
    print(output_folders[j])
    all_mgrs.append(load_inversion_results(join(output_path,output_folders[j])))

# %%
# from runTHMC import runTHMC
# val_1m = runTHMC(alpha=5,n=1.26)


# %%
single_point = (all_mgrs[0]['paraDomain'].cellCenters()[:,1]>-1) & (all_mgrs[0]['paraDomain'].cellCenters()[:,0]>18) & (all_mgrs[0]['paraDomain'].cellCenters()[:,0]<19)
left_side = (all_mgrs[0]['paraDomain'].cellCenters()[:,1]>-1) & (all_mgrs[0]['paraDomain'].cellCenters()[:,0]<18)
right_side = (all_mgrs[0]['paraDomain'].cellCenters()[:,1]>-1) & (all_mgrs[0]['paraDomain'].cellCenters()[:,0]>19) & ((all_mgrs[0]['paraDomain'].cellCenters()[:,0]>38) | (all_mgrs[0]['paraDomain'].cellCenters()[:,0]<25))
over_1m = (all_mgrs[0]['paraDomain'].cellCenters()[:,1]>-1) & ((all_mgrs[0]['paraDomain'].cellCenters()[:,0]>40) | (all_mgrs[0]['paraDomain'].cellCenters()[:,0]<25))
all_mgrs[0]['SWC'] = log_model_inverse(all_mgrs[0]['model'], a, b)
SWC_diff = []
SWC_diff_left = []
SWC_diff_right = []
SWC_diff_over_1m = []
water_m2 = []
for j in range(len(all_mgrs)-1):
    all_mgrs[j+1]['SWC'] = log_model_inverse(all_mgrs[j+1]['model'], a, b)
    all_mgrs[j+1]['SWC_diff'] = (all_mgrs[j+1]['SWC'] - all_mgrs[0]['SWC'])/100
    # SWC_diff.append(np.mean(all_mgrs[j+1]['SWC_diff'][single_point]))
    SWC_diff.append(sum(all_mgrs[j+1]['SWC_diff'][single_point]*all_mgrs[0]['paraDomain'].cellSizes()[single_point]))#/(max(all_mgrs[0]['paraDomain'].cellCenter()[:,0])-min(all_mgrs[0]['paraDomain'].cellCenter()[:,0])))#/sum(all_mgrs[0]['paraDomain'].cellSizes()[over_1m]))
    # SWC_diff_left.append(np.mean(all_mgrs[j+1]['SWC_diff'][left_side]/all_mgrs[0]['paraDomain'].cellSizes()[left_side]))
    # SWC_diff_right.append(np.mean(all_mgrs[j+1]['SWC_diff'][right_side]/all_mgrs[0]['paraDomain'].cellSizes()[right_side]))
    # SWC_diff_over_1m.append(np.mean(all_mgrs[j+1]['SWC_diff'][over_1m])*sum(all_mgrs[0]['paraDomain'].cellSizes()[over_1m]))
    # water_m2.append(np.mean(all_mgrs[j+1]['SWC_diff'][over_1m]))#*all_mgrs[0]['paraDomain'].cellSizes()[over_1m]))

# Plot water_m2 time-series
fig, ax = plt.subplots(figsize=(15, 8))
# Change folder name to datetime
dates = [datetime.strptime(folder_name[:8], '%y%m%d%H') for folder_name in output_folders[begin_index+1:end_index+1]]
# ax.plot(dates, SWC_diff_over_1m-SWC_diff_over_1m[0], '-sb',linewidth=3,markersize=10, label='SWC Change from ERT in x=0~43 m')
# ax.plot(dates, SWC_diff_left-SWC_diff_left[0], '-vg', linewidth=3, markersize=10, label='SWC Change from ERT in x=0~18 m')
# ax.plot(dates, SWC_diff_right-SWC_diff_right[0], '-vk', linewidth=3, markersize=10, label='SWC Change from ERT in x=19~43 m')
ma_SWC_diff = pd.Series(SWC_diff).rolling(window=3).mean()
ax.plot(dates, SWC_diff-SWC_diff[0], '-o',color='k', linewidth=3, markersize=10, label='ERT at x=18~19 m')
ax.set_xlim([datetime(2024, 10, 8, 0, 0), datetime(2024, 10, 14, 16, 0)])
# ax.set_xlim([datetime(2024, 10, 31, 0, 0), datetime(2024, 11, 7, 0, 0)])
# ax.set_ylim([-0.005, 0.021])
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

# Plot the filterd_hydro_data['mean_1m'] at ax2
filterd_hydro_data['date_time'] = pd.to_datetime(filterd_hydro_data['date_time'])
index = filterd_hydro_data['date_time'] == datetime(2024, 10, 8, 6, 0)#datetime(2024, 10, 31, 0, 0)
filterd_hydro_data['SWChange_hydro'] = (filterd_hydro_data['mean_1m']-list(filterd_hydro_data['mean_1m'][index])[0])/100*0.75
ax.plot(filterd_hydro_data['date_time'], filterd_hydro_data['SWChange_hydro'], 
        '-or', linewidth=3, markersize=10, label='Contact Sensor')
ax.legend(fontsize=20)


# ax2 = ax.twinx()
# ax2.plot(THMC_time,val_1m)
# ax2.set_ylim([0.3, 0.41])
from matplotlib.ticker import FuncFormatter
def y_tick_formatter(value, tick_number):
    """
    Format y-axis tick labels. If the value is zero, return '0'.
    For positive values, return '+value' with one decimal place.
    """
    if value == 0:
        return '0'
    elif value > 0:
        return f'+{value:.3f}'
    else:
        return f'{value:.3f}'
# ax.yaxis.set_major_formatter(FuncFormatter(y_tick_formatter))

# Extract filterd_hydro_data['SWChange_hydro'] from 2024/10/10 10:00 to 2024/10/14 14:00
start_time = pd.Timestamp('2024-10-10 06:00')
end_time = pd.Timestamp('2024-10-14 14:00')
# start_time = pd.Timestamp('2024-11-02 06:00')
# end_time = pd.Timestamp('2024-11-07 00:00')
filtered_df = filterd_hydro_data[(filterd_hydro_data['date_time'] >= start_time) & (filterd_hydro_data['date_time'] <= end_time)]
filtered_df['time_delta'] = (filtered_df['date_time'] - filtered_df['date_time'].min()).dt.total_seconds() / 3600
x = filtered_df['time_delta'].values
y = filtered_df['SWChange_hydro'].values
coefficients = np.polyfit(x, y, 1)
slope, intercept = coefficients

from datetime import datetime, timedelta

# Define the start datetime
start_datetime = datetime.strptime('2024/10/10 06:00', '%Y/%m/%d %H:%M')
# start_datetime = datetime.strptime('2024/11/02 06:00', '%Y/%m/%d %H:%M')
# Initialize an empty list to store the datetime objects
THMC_time = []
num = 60
x_delta = np.linspace(0,2*(num-1),num)

# Loop to generate 200 datetime objects with 1-hour intervals
for i in range(len(x_delta)):
    # Append the current datetime to the list
    THMC_time.append(start_datetime + timedelta(hours=i*2))

# ax.plot(THMC_time, slope * x_delta + intercept, 
#         '--k',alpha=0.2, linewidth=3, markersize=10, label='Water Change from Contact Sensor')

# Define the start datetime
start_datetime = datetime.strptime('2024/10/08 12:00', '%Y/%m/%d %H:%M')
# start_datetime = datetime.strptime('2024/10/31 20:00', '%Y/%m/%d %H:%M')
# Initialize an empty list to store the datetime objects
THMC_time = []
num = 60
x_delta = np.linspace(0,2*(num-1),num)

# Loop to generate 200 datetime objects with 1-hour intervals
for i in range(len(x_delta)):
    # Append the current datetime to the list
    THMC_time.append(start_datetime + timedelta(hours=i*2))

# ax.plot(THMC_time, slope * x_delta + intercept, 
#         '--b', linewidth=3, markersize=10, label='Water Change from Contact Sensor')


# %%
# start_time = pd.Timestamp('2024-10-08 08:00')
# end_time = pd.Timestamp('2024-10-14 14:00')
start_time = pd.Timestamp('2024-10-31 08:00')
end_time = pd.Timestamp('2024-11-07 00:00')
filtered_df = filterd_hydro_data[(filterd_hydro_data['date_time'] >= start_time) & (filterd_hydro_data['date_time'] <= end_time)]
filtered_df['time_delta'] = (filtered_df['date_time'] - filtered_df['date_time'].min()).dt.total_seconds() / 3600
# %%
Total_wu = np.trapz(y = filtered_df['SWChange_hydro'].values, x = filtered_df['time_delta'].values)
print('Total Water Usge: {:.1f}'.format(Total_wu))
Opt_wu = np.trapz(y = slope * x_delta + intercept, x = x_delta)
print('Optimized Water Usge: {:.1f}'.format(Opt_wu))
print('Toatal Saved Water: {:.1f}, ({:.1f} %)'.format((Total_wu - Opt_wu) , (Total_wu - Opt_wu)/Total_wu*100))
# %%
