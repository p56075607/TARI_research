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
import pickle
# %%
# %%
def read_hydro_data(data_path):
    df = pd.read_excel(data_path, sheet_name='農試所旱田')

    # 將'TIME'列轉換為日期時間格式
    df['TIME'] = pd.to_datetime(df['TIME'])
    df.set_index('TIME', inplace=True)

    return df
df = read_hydro_data(join(r'C:\Users\Git\TARI_research\data\external','水文站資料彙整_20241127.xlsx'))
df = df.resample("h").mean()
df['date_time'] = df.index
df["mean_1m"] = (df["30cm"] + df["50cm"]  + df["100cm"] + df["500cm"] + df["700cm"] + df["900cm"])/6
fig, ax = plt.subplots()
ax.plot(df.index, df["mean_1m"], label="mean_1m")
# %%
# # %%
# # input csv file 
# df = pd.read_csv("data\external\旱田_1217.csv")

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

# %%
# urf_path = join(r'D:\R2MSDATA\TARI_E2_test','test_urf')
# ohmfiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.ohm')])

# all_data = []
# dates = []
# for i,output_folder_name in enumerate(ohmfiles):
#     print(output_folder_name)
#     data = pg.load(join(urf_path, output_folder_name))
#     data.remove(~((data['a']==33)*(data['b']==30)*(data['m']==31)*(data['n']==32)))
#     if i == 0:
#         k = ert.createGeometricFactors(data, numerical=True)
#     data['rhoa'] = k * data['r']
#     all_data.append(data['rhoa'][0])
#     dates.append(pd.to_datetime(datetime.strptime(output_folder_name[:8] , '%y%m%d%H')))
# Import ERT data
# read the dates and median_RHOA_E1 from the pickle file
with open(join(r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor','median_RHOA_E2_and_date.pkl'), 'rb') as f:
    dates = list(pickle.load(f))
    all_data = list(pickle.load(f))

# %%
fig, ax = plt.subplots()
# ax.plot(df.index, df["mean_1m"], label="mean_1m")
ax2 = ax.twinx()
ax2.plot(dates, all_data,color='orange')
# %%
# rho_100_cm = []
# mesh_filter = (para_domain.cellCenters()[:,1]>-1)
# for i,output_folder_name in enumerate(output_folders):
#     rho_100_cm.append(np.median(all_mgrs[i]['model'][mesh_filter]))


# Extract df data from dates
filterd_hydro_data = pd.DataFrame( columns=df.columns )
for i in range(len(all_data)):
    mask = (df['date_time'] == dates[i])
    filterd_hydro_data = pd.concat([filterd_hydro_data, df.loc[mask]])


# find index of filterd_hydro_data['date_time'] in dates
index = []
for i in range(len(filterd_hydro_data)):
    index.append(dates.index(filterd_hydro_data['date_time'][i]))
# extract rho_100_cm from df['date_time']
rhoa = [all_data[i] for i in index]
filterd_hydro_data['rhoa'] = rhoa

# read filterd_hydro_data_E2.csv
# filterd_hydro_data = pd.read_csv('filterd_hydro_data_E2.csv')
filterd_hydro_data = filterd_hydro_data[(filterd_hydro_data['date_time'] > '2024-06-01 00:00:00')]
filterd_hydro_data = filterd_hydro_data[(filterd_hydro_data['date_time'] < '2024-09-01 00:00:00')]
fig, ax = plt.subplots(figsize=(15,8))
ax.plot(filterd_hydro_data['date_time'], filterd_hydro_data['mean_1m'],'o', label="mean_1m")
ax2 = ax.twinx()
ax2.plot(filterd_hydro_data['date_time'], filterd_hydro_data['rhoa'],'o',color='orange', label="rhoa")
# Extracting the relevant columns
x_data = filterd_hydro_data['mean_1m']
y_data = filterd_hydro_data['rhoa']

# Defining the logarithmic model y = a * ln(x) + b
def log_model(x, a, b):
    return a * np.log(x) + b

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
ax.scatter(x_data, y_data, c='k'#filterd_hydro_data.index
           , label='Data points',s=3)
ax.plot(x_fit, y_fit, color='red', label=f'Fit: y = {a:.2f}ln(x) + {b:.2f}')
ax.fill_between(x_fit, y_fit_lower, y_fit_upper, color='red', alpha=0.1, label='±5% Confidence Interval')
# ax.set_xlim(xlim)
ylim = [30, 250]
# ax.set_ylim(ylim)

fontsize=18
# Adding equation and R-squared to the plot
plt.text(xlim[0],100 ,f'    $y = {a:.2f} \ln(x) + {b:.2f}$\n    $R^2 = {r_squared:.2f}$\n', fontsize=fontsize)

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
filterd_hydro_data.to_csv('filterd_hydro_data_E2.csv', index=False, columns= ['date_time', 'mean_1m', 'rhoa'])
