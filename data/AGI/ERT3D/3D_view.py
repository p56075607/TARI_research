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
from pygimli.viewer import pv
import pyvista as pvista
# %%
def load_inversion_result(save_ph):
    output_ph = join(save_ph,'repeat_3','ERTManager')
    # Load paraDomain
    para_domain = pg.load(join(output_ph,'resistivity-pd.bms'))
    mgr_dict = {'paraDomain': para_domain, 
                'data': ert.load(join(output_ph,'inverison_data.ohm')),
                }
    return mgr_dict

mgr3d = load_inversion_result(r'output\091011')
# %%
kw = dict(
        logScale=True,cMap='jet',cMin=10,cMax=20,
        )

result3d = mgr3d['paraDomain']
result3d["res"] = mgr3d['paraDomain']['Resistivity']
pl, _ = pg.show(result3d, label="res", style="surface", hold=True, **kw,
                filter={"slice": dict(normal="x", origin=[3.5, 0, 0])})
                #filter={"threshold": dict(value=15, scalars="res", method="lower")})
pv.drawMesh(pl, result3d, label="res", style="surface", **kw,
            filter={"slice": dict(normal="y", origin=[0, 7, 0])})
pv.drawMesh(pl, result3d, label="res", style="surface", **kw,
            filter={"slice": dict(normal="z", origin=[0, 0, -3])})
pv.drawMesh(pl, result3d, label="res", style="surface", **kw,
            filter={"slice": dict(normal=[2,1, 0], origin=[3.5, 7, 0])})
pv.drawMesh(pl, result3d, label="res", style="surface", **kw,
            filter={"slice": dict(normal=[-2,1, 0], origin=[3.5, 7, 0])})
pv.drawSensors(pl, mgr3d['data'].sensors(), diam=0.1, color='black')

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

# %%
data = mgr3d['data']

mesh_x = np.linspace(min(pg.x(data)), max(pg.x(data)), 70)
mesh_y = np.linspace(min(pg.y(data)),max(pg.y(data)), 140)
mesh_z = 1.0 - np.logspace(np.log10(1.0), np.log10(6.0),50 )

grid = pg.createGrid(x=mesh_x, y= mesh_y, z= mesh_z)

Resistivity = pg.interpolate(mgr3d['paraDomain'], 
                             mgr3d['paraDomain']['Resistivity'],
                             grid.cellCenter())
grid['Resistivity'] = Resistivity
# rho_grid = np.reshape(Resistivity,(len(mesh_y),len(mesh_x),len(mesh_z)))

# %%
pl, _ = pg.show(grid, label="Resistivity", style="surface", hold=True, **kw,
                filter={"slice": dict(normal="x", origin=[3.5, 0, 0])})
pv.drawMesh(pl, grid, label="Resistivity", style="surface", **kw,
            filter={"slice": dict(normal="y", origin=[0, 7, 0])})
pv.drawMesh(pl, grid, label="Resistivity", style="surface", **kw,
            filter={"slice": dict(normal=[2,1, 0], origin=[3.5, 7, 0])})
pv.drawMesh(pl, grid, label="Resistivity", style="surface", **kw,
            filter={"slice": dict(normal=[-2,1, 0], origin=[3.5, 7, 0])})
# pv.drawMesh(pl, grid, label="Resistivity", style="surface", **kw,
#             filter={"threshold": dict(value=18, scalars="Resistivity", method="upper")})
# pv.drawMesh(pl, grid, label="Resistivity", style="surface", **kw,
#             filter={"threshold": dict(value=13, scalars="Resistivity", method="lower")})
pv.drawSensors(pl, mgr3d['data'].sensors(), diam=0.1, color='black')
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