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

mesh = mt.createMesh(plc, quality=34.3, area=2e-4, smooth=[10, 1])
print(mesh)
ax, _ = pg.show(mesh,markers=True)
for i, s in enumerate(scheme.sensors()):
    ax.text(s.x(), s.y(), str(i+1), zorder=100)

c2 = mt.createCircle(pos=(0.0, 0.0), radius=0.1, segments=100, marker=2)
geom = plc+ c2
mesh_fwd = mt.createMesh(geom, area=2e-5, smooth=[10, 1])
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
mgr.showResultAndFit(**kw)
# %%

# %%
def generate_circle_points(radius=0.25, num_points=24):
    points = []
    angle_interval = 2 * np.pi / num_points

    for i in range(num_points):
        angle = angle_interval * i
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append((x, y))

    initial_point = (0, -radius)
    initial_angle = np.arctan2(initial_point[1], initial_point[0])

    rotated_points = []
    for x, y in points:
        rotated_x = x * np.cos(-initial_angle) - y * np.sin(-initial_angle)
        rotated_y = x * np.sin(-initial_angle) + y * np.cos(-initial_angle)
        rotated_points.append((round(rotated_x, 8), round(rotated_y, 8)))

    return np.array(rotated_points)

def find_closest_circle_points(target_points, circle_points):
    closest_points = []

    for target in target_points:
        distances = np.linalg.norm(circle_points[:, :2] - target[:2], axis=1)
        closest_index = np.argmin(distances)
        closest_points.append(circle_points[closest_index])

    return np.array(closest_points)

# Generate the 24 points on the circle
circle_points = generate_circle_points()

# Define the target points
target_points = np.array(scheme.sensors())

# Find the closest circle points
circle_points = find_closest_circle_points(target_points, circle_points)

print(circle_points)
# %%
data = data_sim.copy()
data.setSensorPositions(circle_points)

# scheme_circle = ert.DataContainer()
# scheme_circle.setSensorPositions(circle_points)
# scheme_circle['a'] = scheme['a']
# scheme_circle['b'] = scheme['b']
# scheme_circle['m'] = scheme['m']
# scheme_circle['n'] = scheme['n']

# plc_circle = mt.createPolygon(scheme_circle.sensors(), isClosed=True,
#                        addNodes=3, interpolate='spline',boundaryMarker=-1)
# geom_circle = plc_circle+ c2
# mesh_fwd_circle = mt.createMesh(geom_circle, area=2e-5, smooth=[10, 1])
# pg.show(mesh_fwd_circle,markers=True)
# model_rho = np.array([100,10000])[mesh_fwd_circle.cellMarkers()-1]
# data_sim_circle = ert.simulate(mesh_fwd_circle, scheme=scheme_circle, res=model_rho, noiseLevel=0.01,
#                     noiseAbs=1e-6, seed=1337)
# %%
plc = mt.createPolygon(data.sensors(), isClosed=True,
                       addNodes=3, interpolate='spline',boundaryMarker=-1)

mesh_circle = mt.createMesh(plc, quality=34.3, area=2e-4, smooth=[10, 1])
print(mesh_circle)
ax, _ = pg.show(mesh_circle,markers=True)
for i, s in enumerate(scheme.sensors()):
    ax.text(s.x(), s.y(), str(i+1), zorder=100)

# %%
mgr_circle = ert.Manager(data)
mgr_circle.invert(mesh=mesh_circle, verbose=True)
mgr_circle.showResultAndFit(**kw)

# %%
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
_, cb = pg.show(mgr_circle.paraDomain,mgr_circle.model,ax = ax1,**kw)
ax1.plot(pg.x(data),pg.y(data),'ko')
_, cb = pg.show(mgr.paraDomain,mgr.model,ax = ax2,**kw)
ax2.plot(pg.x(data_sim),pg.y(data_sim),'ko')
# %%
# ax, cb = ert.show(data_sim_circle, "rhoa", circular=True)
# fig, ax = plt.subplots(figsize=(5,5))
# ax.scatter(data_sim['rhoa'],data_sim_circle['rhoa'],s=1)
# xticks = ax.get_xlim()
# yticks = ax.get_ylim()
# lim = 220
# ax.plot([0,lim],[0,lim],'k-',linewidth=1, alpha=0.2)
# ax.set_xlim([0,lim])
# ax.set_ylim([0,lim])