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
rho1 = 500
rho2 = 1000
kw = dict(cMin=rho1, cMax=rho2, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')

def print_data_siminfo(data_sim):
    pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
    pg.info('Simulated data', data_sim)
    pg.info('The data contains:', data_sim.dataMap().keys())
    pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random see
left = 0
right = 20
depth = 4
world = mt.createWorld(start=[left, 0], end=[right, -depth],
                    layers=[-1], 
                    worldMarker=True)
mesh = mt.createMesh(world, quality=34, area=0.01)

rhomap = [[1, rho1], [2, rho2]]
ax, cb = pg.show(mesh, 
    data=rhomap, 
    showMesh=True,**kw)
ax.set_ylim(-2, 0)
ax.set_xlim(0, 1)

# %%
# Create a mesh for the forward modeling
scheme = ert.createData(elecs=np.linspace(start=left, stop=right, num=21),
                           schemeName='dd')

DATA = []
inf_depths = np.linspace(-1,0,11)[::-1]#-0.1
for n,inf_depth in enumerate(inf_depths):
    world = mt.createWorld(start=[left, 0], end=[right, -depth],
                        layers=[inf_depth], 
                        worldMarker=True)
    mesh = mt.createMesh(world, quality=34, area=0.01)
    # ax, cb = pg.show(mesh, markers=True,**kw)
    if (n == 1) | (n == 2):
        rhomap = [[0, rho1], [2, rho2]]
    elif inf_depth == 0:
        rhomap = [[2, rho2]]
    else:
        rhomap = [[1, rho1], [2, rho2]]

    ax, cb = pg.show(mesh, 
        data=rhomap, 
        showMesh=True,**kw)
    ax.set_ylim(-3, 0)
    ax.set_xlim(0, 1)
    data_i = ert.simulate(mesh=mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                            noiseAbs=1e-6, 
                            seed=1337) 
    def print_data_siminfo(data_sim):
        pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
        pg.info('Simulated data', data_sim)
        pg.info('The data contains:', data_sim.dataMap().keys())
        pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
        pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random seed for repeatable noise in synthetic experiments 
    print_data_siminfo(data_i)
    DATA.append(data_i)
# %%
tlmgr = ert.TimelapseERT(DATA)
# %%
yDevide = np.linspace(-4,0,61)#np.concatenate((np.linspace(-3,-2,3),1.0 - np.logspace(np.log10(1.0), np.log10(2.5),31 )[::-1]))
xDevide = np.linspace(start=0, stop=20, num=41)
ERTDomain = pg.createGrid(x=xDevide,y=yDevide,marker=2)
grid = pg.meshtools.appendTriangleBoundary(ERTDomain, marker=1,
                                    xbound=100, ybound=100)
ax,_ = pg.show(grid,markers=True, showMesh=True)

# %%
tlmgr.invert(mesh=grid, lam=100, maxIter=40,verbose=True)

# %%
kw_appres = dict(cMin=rho1, cMax=rho2, cMap='jet', logScale=True, orientation='vertical',
                    label='appres', xlabel='Distance (m)', ylabel='Array type and \nElectrode separation (m)')
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label=r'$\Delta \rho$ (%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
fig, ax = plt.subplots(figsize=(3*10, 11*3), 
                       ncols=3, nrows=11,constrained_layout=False)
for n,inf_depth in enumerate(inf_depths):
    ert.show(DATA[n], ax=ax[n, 0],cMin=rho1, cMax=rho2, cMap='jet', logScale=True, orientation='vertical',
                    # label='appres', xlabel='Distance (m)', ylabel='Array type and \nElectrode separation (m)'
                    )#**kw_appres)
    pg.show(tlmgr.pd, tlmgr.models[n],ax=ax[n,1], **kw)

    if n == 0:
        pg.show(tlmgr.pd, tlmgr.models[n],ax=ax[n,2], **kw)
    else:
        pg.show(tlmgr.pd, 
                100*(tlmgr.models[n]-tlmgr.models[0])/tlmgr.models[0],ax=ax[n,2],
                **kw_compare)

# %%
tlmgr.showAllModels(**kw)
# %%
tlmgr.showAllModels(ratio=True,rMax=2)
# %%
mgr = ert.ERTManager(data_sim)
model = mgr.invert(data_sim,#paraDX=0.25,paraMaxCellSize=0.1,
                   mesh=grid,  
                   lam=100, maxIter=40,
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
ax,_ = pg.viewer.showMesh(mgr.paraDomain, mgr.model,#grid,data=rho_normal_grid,
                             **kw)
ax.set_ylim(-2, 0)
ax, cb = pg.show(mesh, 
    data=rhomap, 
    showMesh=True,**kw)
ax.set_ylim(-2, 0)