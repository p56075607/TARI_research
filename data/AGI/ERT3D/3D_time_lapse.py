# -*- coding: utf-8 -*-
# %%
# Import moduals
import sys
# from inverison_util import data_filtering, plot_inverted_profile, plot_inverted_contour, plot_convergence, crossplot, data_misfit, export_inversion_info
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"
# %config InlineBackend.figure_format='svg' # Setting figure format for this notebook
import numpy as np
import pygimli as pg
from pygimli.viewer import pv
from pygimli.physics import ert  # the module
import pygimli.meshtools as mt
from datetime import datetime
import os
from os.path import join
from os import listdir 
from os.path import isfile
from os.path import isdir
from datetime import timedelta
import matplotlib.dates as mdates
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pvista
#
# %%
date_mother_path = r'C:\Users\Git\TARI_research\data\AGI\ERT3D\output'
date_folders = [f for f in listdir(date_mother_path) if isdir(join(date_mother_path,f)) and f.startswith('1')]


def load_inversion_result(save_ph,output_folder):
    output_ph = join(save_ph,output_folder,'repeat_3','ERTManager')
    # Load paraDomain
    para_domain = pg.load(join(output_ph,'resistivity-pd.bms'))
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

    mgr_dict = {'Name':output_folder,
                'paraDomain': para_domain, 
                'data': ert.load(join(output_ph,'inverison_data.ohm')),
                'rrms': rrms, 'chi2': chi2, 'lam': lam,
                'rrmsHistory': rrmsHistory, 'chi2History': chi2History
                }
    return mgr_dict

mgrs = []
for date_folder in date_folders:
    date_path = join(date_mother_path, date_folder)
    mgrs.append(load_inversion_result(join(date_mother_path), date_folder))

# %%
# Interpolate grid 
mesh_x = np.linspace(min(pg.x(mgrs[-1]['data'])), max(pg.x(mgrs[-1]['data'])), 70)
mesh_y = np.linspace(min(pg.y(mgrs[-1]['data'])), max(pg.y(mgrs[-1]['data'])), 140)
mesh_z = 1.0 - np.logspace(np.log10(1.0), np.log10(6.0),50 )

grid = pg.createGrid(x=mesh_x, y= mesh_y, z= mesh_z)
# %%
for i,mgr in enumerate(mgrs[:1]):
    print('diff_'+mgrs[-1]['Name']+'_'+mgr['Name']+'_grid.vtk')
    model1 = mgrs[-1]['paraDomain']['Resistivity']
    model2 = mgrs[i]['paraDomain']['Resistivity']
    data = mgrs[-1]['data']
    diff = mgrs[-1]['paraDomain']
    diff['diff'] = (np.log10(model2) - np.log10(model1))/np.log10(model1)*100
    # Interpolate grid 
    Diff = pg.interpolate(diff, 
                            diff['diff'],
                            grid.cellCenter())
    grid['diff'] = Diff
    grid.exportVTK('diff_'+mgrs[-1]['Name']+'_'+mgr['Name']+'_grid.vtk')

# # %%
# diff.exportVTK('diff_22_20.vtk')
# # %%


# kw = dict(cMin=-2.5, cMax=0,logScale=False,cMap=custom_cmap)
# pl, _ = pg.show(diff, label="diff_22_18", style="surface", hold=True, **kw,
#                 filter={"slice": dict(normal="x", origin=[3.5, 0, 0])})
# pv.drawMesh(pl, diff, label="diff_22_18", style="surface", **kw,
#             filter={"slice": dict(normal="y", origin=[0, 7, 0])})
# pl.camera_position = [
#     [-20, -20, 20],   # camera_position
#     [3.5, 7, -5],      # view point
#     [0, 0, 1]       # 向上的方向
# ]
# light = pvista.Light(position = [-20, -20, 20],intensity = 1, color='white', light_type='headlight')  
# pl.add_light(light)
# pl.background_color = 'white'
# # pl.add_axes(xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis', label_font_size=12)
# pl.show()

# %%
datum = []
dates = []
for i,mgr in enumerate(mgrs):
    datum.append(np.mean(mgr['paraDomain']['Resistivity'][(mgr['paraDomain'].cellCenters()[:,2] > -2) & (mgr['paraDomain'].cellCenters()[:,0] < 3)]))
    print(mgr['Name'])
    if mgr['Name'][-1] == '1':
        time_str = '24'+mgr['Name'][-6:-1]+'9'
    elif mgr['Name'][-1] == '2':
        time_str = '24'+mgr['Name'][-6:-2]+'13'
    dates.append(datetime.strptime(time_str, '%y%m%d%H'))

plt.plot(datum)


# %%
import matplotlib.animation as animation
fig, ax = plt.subplots(figsize=(10, 5))
ax.axvline(dates[-1], color='k', linestyle='-', linewidth=2)
ax.plot(dates[-1], datum[-1], 'ko', linewidth=2, markersize=10)
    # datum.append(np.mean(mgr['data']['rhoa'][mgr['data']['a'] == 14]))
ax.set_xlim(datetime(2024,11,18,0),datetime(2024,11,22,23))
# set major xtick to 00:00
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
# ax.plot(dates, datum,'bo')
fz_minor = 18
plt.xticks(fontsize=fz_minor,rotation=45, ha='right', rotation_mode='anchor')
plt.yticks(fontsize=fz_minor)
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
plt.tight_layout()  # Adjust layout to make room for the rotated date labels
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
ax.set_ylabel('Mean Resistivity ($\Omega - m$)',fontsize=fz_minor)
ax.set_ylim([14.4,15])
def update(frame):
    #ax.lines = ax.lines[:1]  # 保留原始資料線，移除先前的垂直線
    current_date = dates[frame]
    # ax.axvline(current_date, color='r', linestyle='-', linewidth=2)
    ax.plot(current_date, datum[frame],'bo',markersize=10)
    return ax.lines

ani = animation.FuncAnimation(
    fig, update, frames=len(dates)-1, interval=1000, blit=True
)
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)
plt.show()
ani.save('date_res_animation.gif', writer='Pillow', fps=1)
# %%
colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 0.95, 1]  # 範圍從0到-1是白色，-1到-10是白色到藍色的漸變
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
kw = dict(cMin=-3, cMax=0,logScale=False,cMap=custom_cmap)
pl, _ = pg.show(grid, label="diff", style="surface", hold=True, **kw,
                filter={"slice": dict(normal="x", origin=[2, 0, 0])})
pv.drawMesh(pl, grid, label="diff", style="surface", **kw,
            filter={"slice": dict(normal="y", origin=[0, 7, 0])})
pl.camera_position = [
    [-20, -20, 20],   # camera_position
    [3.5, 7, -5],      # view point
    [0, 0, 1]       # 向上的方向
]
light = pvista.Light(position = [-20, -20, 20],intensity = 1, color='white', light_type='headlight')  
pl.add_light(light)
pl.background_color = 'white'
# pl.add_axes(xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis', label_font_size=12)
pl.show()