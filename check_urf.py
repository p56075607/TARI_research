# -*- coding: utf-8 -*-
# %%
# Import moduals
import sys
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"
# %config InlineBackend.figure_format='svg' # Setting figure format for this notebook
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import pygimli.meshtools as mt
from datetime import datetime
import os
from os.path import join
from os import listdir
from datetime import timedelta
import matplotlib.dates as mdates
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

root_path = r'D:\R2MSDATA\TARI_E1_test'
output_ph = join(root_path,'output')
# check if the folder exists
if not os.path.exists(output_ph):
    # if not, create the folder
    os.makedirs(output_ph)
    print(f'Folder "{output_ph}" created.')
else:
    print(f'Folder "{output_ph}" already exists.')

urf_path = join(root_path,'urf')
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
date_lim = [datetime(2024,10,31,0,0),datetime(2024,11,12,0,0)]
picked_date = [date for date in dates if date >= date_lim[0] and date <= date_lim[1]]
picked_date_index = [dates.index(date) for date in picked_date]
picked_output_folders = [ohmfiles[i] for i in picked_date_index]

# %%
for i, ohm_fname in enumerate(picked_output_folders[::24]):
    data = pg.load(join(urf_path, ohm_fname))
    data.remove(abs(data['a']-data['b'])!=42)
    qua = np.array(list(zip(data['a'], data['b'], data['m'], data['n'],data['r'])))
    plt.plot(qua[71:104+1,4])
    plt.xticks(np.arange(0, 35, 1))
    plt.grid(visible=True,which='both',axis='both')