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
    # �s�x�ѪR�X�Ӫ����
    dates = []

    # �M����Ƨ������Ҧ��ɮ�
    for filename in listdir(directory_path):
        # �ˬd�ɮצW�٬O�_�ŦX�S�w�榡
        if filename.endswith('.urf'):
            date_str = filename[:8]  # �����������
            try:
                # �ഫ����榡�q 'YYMMDDHH' �� datetime ��H
                date = datetime.strptime(date_str, '%y%m%d%H')
                dates.append(date)
            except ValueError:
                # �p�G����榡�����T�A�������ɮ�
                continue

    return dates

dates = check_files_in_directory(urf_path)
# %%
mean_rhoa = []
for i,urf_file_name in enumerate(urffiles):
    if urf_file_name[:-4]+'.ohm' in ohmfiles: # �ˬd�O�_�� ohm �ɮסA�Y���N�������L���]�Ϻt
        print(urf_file_name[:-4]+'.urf is already processed. Skip it!')
    else:
        print('Processing: '+urf_file_name)
        ohm_file_name = convertURF(join(urf_path,urf_file_name),has_trn = False)
        data = ert.load(ohm_file_name)
        print(data) 
        data['k'] = ert.createGeometricFactors(data, numerical=True) # �H�ƭȤ�k�p��X��]�l�A���ӶO�q���귽
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