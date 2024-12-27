# %%
import pygimli as pg
from pygimli.physics import ert
import numpy as np
import sys
import os
from os.path import join
from os import listdir 
from os.path import isfile
from os.path import isdir
sys.path.append(r'E:\研究室電腦E槽\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\PYGIMLY\field_data\SANSIN')
from stg2ohm_pp import stg2ohm
from pygimli.viewer import pv
import matplotlib.pyplot as plt
# %%
date_mother_path = r'C:\Users\Git\TARI_research\data\AGI\ERT3D\output'
date_folders = [f for f in listdir(date_mother_path) if isdir(join(date_mother_path,f)) and f.startswith('1')]

DATA = []
# survey_name = '111801'
for survey_name in date_folders:
    # stg_fname = join(survey_name,survey_name+'.stg')#r'0910\091011.stg'
    # cmd_path = r'0910\091011.cmd'

    ohm_file_name = join(date_mother_path, survey_name,'repeat_3','ERTManager','inverison_data.ohm')#stg2ohm(cmd_path,stg_fname,113,114,is_3D=True)

    data = ert.load(ohm_file_name)
    # data['k'] = ert.createGeometricFactors(data, numerical=True)
    # data['rhoa'] = data['r'] * data['k']
    DATA.append(data)
    

# %%
datum = []
for i in range(len(DATA)):
    datum.append(np.mean(DATA[i]['rhoa']))
# plot the datum
plt.plot(datum)