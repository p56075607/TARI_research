# %%
# Import moduals
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from convertURF import convertURF

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

urf_path = r'C:\Users\R2MSDATA\TARI_E2\urf_test'
urffiles = [_ for _ in listdir(urf_path) if _.endswith('.urf')]
ohmfiles = [_ for _ in listdir(urf_path) if _.endswith('.ohm')]

def check_files_in_directory(directory_path):
    # 存儲解析出來的日期
    dates = []

    # 遍歷資料夾中的所有檔案
    for filename in listdir(directory_path):
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

dates = check_files_in_directory(urf_path)
# %%
mean_rhoa = []
for i,urf_file_name in enumerate(urffiles):
    if urf_file_name[:-4]+'.ohm' in ohmfiles: # 檢查是否有 ohm 檔案，若有就視為做過不跑反演
        print(urf_file_name[:-4]+'.urf is already processed. Skip it!')
    else:
        print('Processing: '+urf_file_name)
        ohm_file_name = convertURF(join(urf_path,urf_file_name),has_trn = False)
        data = ert.load(ohm_file_name)
        print(data) 
        data['k'] = ert.createGeometricFactors(data, numerical=True) # 以數值方法計算幾何因子，較耗費電腦資源
        data['rhoa'] = data['k'] * data['r']
        mean_rhoa.append(np.mean(data['rhoa']))

# %%
# plt.plot(dates,mean_rhoa,'o',markersize=1)
fig, ax1 = plt.subplots(figsize=(20, 8))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.grid(linestyle='--',linewidth=0.5)
ax1.plot(dates,mean_rhoa, 'bo')
ax1.set_xlim([dates[0]-timedelta(days=1),dates[-1]+timedelta(days=1)])
# ax1.set_ylim(40,60)
# Rotate dates for better readability
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to make room for the rotated date labels