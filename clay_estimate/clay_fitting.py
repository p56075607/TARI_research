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
import matplotlib
matplotlib.use('TkAgg')
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
                'model': model, 
                }

    return mgr_dict

# output_path = r'D:\R2MSDATA\TARI_E3_test\output_second_inveriosn'
output_path = r'D:\R2MSDATA\BIGBIRD_AC_test\output_A_second_inversion_091313_092413'
output_folders = [f for f in sorted(listdir(output_path)) if isdir(join(output_path,f))]
# print(output_folders) # ['24022917_m_E1', '24022921_m_E1', '24030215_m_E1', '24030221_m_E1',...]
all_mgrs = []
# begin_index = output_folders.index('24091506_m_E1')
# end_index = output_folders.index('24101500_m_E1')
# begin_index = output_folders.index('24050119_m_E3')
# end_index = output_folders.index('24053123_m_E3')
begin_index = output_folders.index('24091313_m_A')
end_index = output_folders.index('24091913_m_A')
for j in range(begin_index,end_index+1,1):
    print(output_folders[j])
    all_mgrs.append(load_inversion_results(join(output_path,output_folders[j])))

# %%
single_point = (all_mgrs[0]['paraDomain'].cellCenters()[:,1]>-1.5
                ) #& (all_mgrs[0]['paraDomain'].cellCenters()[:,0]>70) & (all_mgrs[0]['paraDomain'].cellCenters()[:,0]<78)
all_cond = []
for i in range(len(all_mgrs)):
    all_cond.append(np.array(1/all_mgrs[i]['model']#[single_point]
                            ))

all_cond = np.array(all_cond)
all_cond_mean = np.mean(all_cond, axis=0)
all_cond_var = np.var(all_cond, axis=0)

fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(all_cond_var,all_cond_mean,c=np.linspace(0,1,len(all_cond_mean)),s=3)
# linear fit 
def linear_fit(x, a, b):
    return a * x + b
params, params_covariance = curve_fit(linear_fit, all_cond_var, all_cond_mean)
a, b = params
ax.plot(all_cond_var, linear_fit(all_cond_var, a, b), 'r-')
ax.text(0.05, 0.95, f'y = {a:.2f}x + {b:.2f}', transform=ax.transAxes, fontsize=20, verticalalignment='top')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(all_mgrs[0]['paraDomain'].cellCenters()[:, 0], all_mgrs[0]['paraDomain'].cellCenters()[:, 1],
           c=np.linspace(0,1,len(all_cond_mean)),s=3)
# ax.plot(a*
plt.show()