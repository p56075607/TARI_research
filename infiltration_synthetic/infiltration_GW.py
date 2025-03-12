# %%
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt
import numpy as np
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from ohm2urf import ohm2urf
from os.path import join

# %%
stg_fname = "test1202-12.stg"
data = ert.load(stg_fname)
print(data)
# %%
left = min(pg.x(data))
right = max(pg.x(data))
depth = 10
rho1 = 200
rho2 = 400
kw = dict(cMin=rho1, cMax=rho2, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')

inf_depths = np.linspace(-1,0,11)[::-1]#-0.1
DATA = []
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
    ax.set_ylim(-10, 0)
    ax.set_xlim(0, 1)

    data_i = ert.simulate(mesh=mesh, scheme=data, res=rhomap, noiseLevel=1,
                            noiseAbs=1e-6, 
                            seed=1337) 
    def print_data_siminfo(data_sim):
        pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
        pg.info('Simulated data', data_sim)
        pg.info('The data contains:', data_sim.dataMap().keys())
        pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
        pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random seed for repeatable noise in synthetic experiments 
    print_data_siminfo(data_i)
    data_i['r'] = data_i['rhoa']/data_i['k']
    DATA.append(data_i)
    data_i.save(join('syn_output', 'syn_{}.ohm'.format(n)))

# %%