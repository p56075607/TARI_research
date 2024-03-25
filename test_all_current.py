# %%
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from convertURF import convertURF
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
# %%
# Read the URF file
# urf_data_name = r'data\electrodes43.urf'
# ohm_file_name = convertURF(urf_data_name,has_trn = False)
# %%
# Loading the .ohm file
ohm_file_name = r'data\electrodes20.ohm'
data = ert.load(ohm_file_name)
print(data) 
# %%
# filter the data
# t5 = (data['a'] > 20) | (data['b'] > 20) | (data['m'] > 20) | (data['n'] > 20)
# index = [i for i, x in enumerate(t5) if x]
# print(r'remove electrode position bigger than 20 {:d}'.format(len(index)))
# data.remove(index)
# %%
ert.show(data,data['rhoa'])
# data.save(join('data','electrode20.ohm'))
# %%
# Create a mesh for the forward modeling
kw = dict(cMin=50, cMax=500, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
left = 0
right = 21
depth = 4
# DATA = []
# inf_depths = np.linspace(-1,0,11)#-0.1
# for n,inf_depth in enumerate(inf_depths):
#     world = mt.createWorld(start=[left, 0], end=[right, -depth],
#                         layers=[inf_depth], 
#                         worldMarker=True)
#     mesh = mt.createMesh(world, quality=34, area=0.01)
#     # ax, cb = pg.show(mesh, markers=True,**kw)
#     if (n == 8) | (n == 9):
#         rhomap = [[0, 50.], [2, 500.]]
#     elif inf_depth == 0:
#         rhomap = [[2, 500.]]
#     else:
#         rhomap = [[1, 50.], [2, 500.]]

#     ax, cb = pg.show(mesh, 
#         data=rhomap, 
#         showMesh=True,**kw)
#     ax.set_ylim(-0.5, 0)
#     ax.set_xlim(0, 1)
#     data_i = ert.simulate(mesh=mesh, scheme=data, res=rhomap, noiseLevel=1,
#                             noiseAbs=1e-6, 
#                             seed=1337) 
#     def print_data_siminfo(data_sim):
#         pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
#         pg.info('Simulated data', data_sim)
#         pg.info('The data contains:', data_sim.dataMap().keys())
#         pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
#         pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random seed for repeatable noise in synthetic experiments 
#     print_data_siminfo(data_i)
#     DATA.append(data_i)
# # %%
# tlmgr = ert.TimelapseERT(DATA)


# %%
def print_data_siminfo(data_sim):
    pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
    pg.info('Simulated data', data_sim)
    pg.info('The data contains:', data_sim.dataMap().keys())
    pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random see

world = mt.createWorld(start=[left, 0], end=[right, -depth],
                    layers=[-1], 
                    worldMarker=True)
mesh = mt.createMesh(world, quality=34, area=0.01)
rhomap = [[1, 50.], [2, 500.]]
# ax, cb = pg.show(mesh, 
#     data=rhomap, 
#     showMesh=True,**kw)
# ax.set_ylim(-2, 0)
# ax.set_xlim(0, 1)
scheme = ert.createData(elecs=np.linspace(start=1, stop=21, num=21),
                           schemeName='dd')
data_sim = ert.simulate(mesh=mesh, scheme=data, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337) 
print_data_siminfo(data_sim)
t5 = (data_sim['rhoa'] > 500) 
index = [i for i, x in enumerate(t5) if x]
print((len(index)))
data_sim.remove(index)
# ert.show(data_sim,data_sim['rhoa'])
skip_step = 5
remove_index = np.array([x for x in np.arange(len(data['rhoa']))])
remove_index = remove_index % skip_step != 0
data.remove(remove_index)
plt.scatter(np.arange(len(data_sim['rhoa'])),data_sim['rhoa'],s=1)
# data_sim.save(join('data','electrode20_sim.ohm'))
# %%
yDevide = np.linspace(-2,0,41)#np.concatenate((np.linspace(-3,-2,3),1.0 - np.logspace(np.log10(1.0), np.log10(2.5),31 )[::-1]))
xDevide = np.linspace(start=0, stop=21, num=41)
ERTDomain = pg.createGrid(x=xDevide,y=yDevide,marker=2)
grid = pg.meshtools.appendTriangleBoundary(ERTDomain, marker=1,
                                    xbound=200, ybound=200)
# ax,_ = pg.show(ERTDomain,markers=True, showMesh=True)
# ax.set_ylim(-2, 0)
# ax.set_xlim(0, 1)
# %%
mgr = ert.ERTManager(data_sim)
model = mgr.invert(data_sim,paraDX=0.25,paraMaxCellSize=0.1,#mesh=grid,  
                   lam=100, maxIter=10,
                #    limits = [40,600],
                   verbose=True)
mgr.showResultAndFit(cMap='jet')
# %%
# Varify the fitted and measured data cross plot
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(np.log10(mgr.data["rhoa"]),np.log10(mgr.inv.response),s=1)
xticks = ax.get_xlim()
yticks = ax.get_ylim()
lim = max(max(yticks,xticks)) + 0.5
ax.plot([0,lim],[0,lim],'k-',linewidth=1, alpha=0.2)
ax.set_xlim([0,lim])
ax.set_ylim([0,lim])
ax.set_xlabel('Log10 of Measured Apparent resistivity')
ax.set_ylabel('Log10 of Predicted Apparent resistivity')
# %%
pg.viewer.showMesh(mgr.paraDomain, mgr.model,#grid,data=rho_normal_grid,
                             **kw)