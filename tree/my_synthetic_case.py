# %%
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = "Microsoft Sans Serif"

scheme = pg.getExampleFile('ert/hollow_limetree.ohm', load=True, verbose=True)
print(scheme)

# %%
plc = mt.createPolygon(scheme.sensors(), isClosed=True,
                       addNodes=3, interpolate='spline',boundaryMarker=-1)
mesh = mt.createMesh(plc, quality=34.3, area=8e-5, smooth=[10, 1])
print(mesh)
# ax, _ = pg.show(mesh,markers=True)
coor = []
for i, s in enumerate(scheme.sensors()):
    # ax.text(s.x(), s.y(), str(i+1), zorder=100)
    coor.append([0.5*s.x(), 0.5*s.y()])

# c2 = mt.createCircle(pos=(0.0, 0.0), radius=0.1, segments=100, marker=2)
c2 = mt.createPolygon(coor, isClosed=True, addNodes=3, interpolate='spline', marker=1)
mesh_constrain = mt.createMesh(plc+c2, quality=34.3, area=8e-5, smooth=[10, 1])
print(mesh_constrain)
pg.show(mesh_constrain,markers=True)


c2 = mt.createPolygon(coor, isClosed=True, addNodes=3, interpolate='spline', marker=2)
geom = plc+ c2
mesh_fwd = mt.createMesh(geom, area=8e-5, smooth=[10, 1])
pg.show(mesh_fwd,markers=True)
# %%
kw = dict(cMin=100, cMax=10000, cMap='jet', logScale=True, label='Resistivity ($\Omega$m)')
model_rho = np.array([100,10000])[mesh_fwd.cellMarkers()-1]
pg.show(mesh_fwd, data=model_rho,**kw)
# %%
data_sim = ert.simulate(mesh_fwd, scheme=scheme, res=model_rho, noiseLevel=0.01,
                    noiseAbs=1e-6, seed=1337)
ax, cb = ert.show(data_sim, "rhoa", circular=True)

# %%
mgr = ert.Manager(data_sim)
mgr.invert(mesh=mesh, verbose=True)


# %%

ax, _ = pg.show(mesh_constrain,markers=True)
for i, s in enumerate(scheme.sensors()):
    ax.text(s.x(), s.y(), str(i+1), zorder=100)

# %%
mgr_constrain = ert.Manager(data_sim)
mgr_constrain.invert(mesh=mesh_constrain, verbose=True)
# %%
ax, cb = pg.show(mgr.paraDomain,mgr.model,**kw)
ax, cb = pg.show(mgr_constrain.paraDomain,mgr_constrain.model,**kw)
pg.show(mesh_fwd, data=model_rho,**kw)

# %%
# def plot_compare_profile(mesh_fwd, model_rho, mgr_normal, mgr_constrain):
kw = dict(cMin=100, cMax=10000, cMap='jet', logScale=True, label='Resistivity ($\Omega$m)',orientation='vertical')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(12,12),constrained_layout=True)
ax2.axis('off')
# Subplot 1:Original resistivity model
pg.viewer.showMesh(mesh_fwd, model_rho,ax=ax1, **kw)
ax1.set_title('Original resistivity model profile',fontweight="bold", size=16)
ax1.plot(pg.x(data_sim),pg.y(data_sim),'ko')

# Subplot 3:Normal inversion result
pg.show(mgr.paraDomain, mgr.model, ax=ax3, **kw)
ax3.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'--w')
ax3.set_title('Normal inversion result', fontweight="bold", size=16)
ax3.plot(pg.x(data_sim),pg.y(data_sim),'ko')
# Subplot 5:Constrained inversion result
pg.show(mgr_constrain.paraDomain, mgr_constrain.model, ax=ax5, **kw)
ax5.set_title('Constrained inversion result', fontweight="bold", size=16)
ax5.plot(pg.x(data_sim),pg.y(data_sim),'ko')



# Comparesion of the results by the residual profile
# Re-interpolate the grid
mesh_x = np.linspace(ax1.get_xlim()[0],ax1.get_xlim()[1],300)
mesh_y = np.linspace(ax1.get_ylim()[0],ax1.get_ylim()[1],300)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh_fwd, model_rho, grid.cellCenters())
rho_normal_grid = pg.interpolate(mgr.paraDomain, mgr.model, grid.cellCenters())
rho_constrain_grid = pg.interpolate(mgr_constrain.paraDomain, mgr_constrain.model, grid.cellCenters())

def plot_residual_contour(ax, grid, data, title,mesh_x,mesh_y, **kw_compare):
    class StretchOutNormalize(plt.Normalize):
        def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
            self.low = low
            self.up = up
            plt.Normalize.__init__(self, vmin, vmax, clip)
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
            return np.ma.masked_array(np.interp(value, x, y))
    clim = [kw_compare['cMin'], kw_compare['cMax']]
    midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-1, up=1)
    X,Y = np.meshgrid(mesh_x,mesh_y)
    diff_pos = pg.interpolate(grid, data, grid.positions())
    mesh = np.reshape(diff_pos,(len(mesh_y),len(mesh_x)))
    ax.contourf(X,Y,mesh,
                levels = 128,
                cmap='bwr',
                norm=midnorm)
    ax.set_title(title, fontweight="bold", size=16)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="4%", pad=.15)
    m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
    m.set_array(mesh)
    m.set_clim(clim[0],clim[1])
    cb = plt.colorbar(m,
                    boundaries=np.linspace(clim[0],clim[1], 128),
                    cax=cbaxes)
    cb_ytick = np.linspace(clim[0],clim[1],5)
    cb.ax.set_yticks(cb_ytick)
    cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
    cb.ax.set_ylabel('Relative resistivity difference\n(%)')

kw_compare = dict(cMin=-10, cMax=10, cMap='bwr',
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
# Subplot 4:Normal inversion compare result
residual_normal_grid = ((np.log10(rho_normal_grid) - np.log10(rho_grid))/np.log10(rho_grid))*100
plot_residual_contour(ax4, grid, residual_normal_grid, 'Normal inversion compare result',mesh_x,mesh_y, **kw_compare)
ax4.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'--k')
# Subplot 6:Constrained inversion compare result
residual_constrain_grid = ((np.log10(rho_constrain_grid) - np.log10(rho_grid))/np.log10(rho_grid))*100
plot_residual_contour(ax6, grid, residual_constrain_grid, 'Constrained inversion compare result',mesh_x,mesh_y, **kw_compare)
ax6.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'-k')
