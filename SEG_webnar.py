# %% [markdown]
# # ERT inversion - beyond the standard
# 
# Here, we exemplify the possibilities of pyGIMLi by treating an ERT timelapse data set. Most of the strategies are independent on the method and can be used for other types of data.
# 
# The data set here was published by H?bner et al. (2017) and describes a shallow infiltration experiment using a surface electrode layout.
# 
# ![Survey layout](slides/images/survey.png)

# %%
# We import some basic libraries like numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# %%
# Furthermore we import pygimli and three of its modules
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt
from pygimli.viewer import pv

# %% [markdown]
# From the whole timelapse set of data files we load a single one, show its properties.

# %%
data = pg.getExampleData("ert/Huebner2017/010.dat")
print(data)

# %% [markdown]
# The data contains 2849 measurements using 392 electrodes. In the `DataContainer`, there are AB-MN indices of the electrodes resistances (`'r'`). We first plot the electrode positions (H?bner et al. 2017, Fig. 1).

# %%
plt.plot(pg.x(data), pg.y(data), ".")
plt.gca().set_aspect(1.0);

# %% [markdown]
# We then compute the geometric factors and apparent resistivities and store both in the data container.

# %%
data["k"] = ert.geometricFactors(data)
data["rhoa"] = data["r"] * data["k"]

# %% [markdown]
# The data exhibit mostly dipole-dipole data measured with a 12-channel ABEM Terrameter LS2 instrument. To fill up the channels, some multi-gradient and some square arrays have been added. We can have a look at the data by using `data.show()`. The `style` argument generates a crossplot of the A over the M electrode. See [`ert.showERTData`](https://www.pygimli.org/pygimliapi/_generated/pygimli.physics.ert.html#pygimli.physics.ert.showERTData).

# %%
data.show(style="A-M");

# %% [markdown]
# We can already see the pseudosections on the main diagonal with some reciprocals mirrored at the diagonal, plus some measurements between the lines (off-diagonals).

# %% [markdown]
# ## Error estimation
# For inversion we need an error estimate to weight the individual data points. The analysis of normal vs. reciprocal data is a common strategy to quantify the data error. We have a very limited amount of reciprocal data pairs with which we can do a reciprocal analysis, i.e. a statistical analysis of the reciprocity deviations as a function of the measured resistance.
# For an example with more rigorous data and background of normal-reciprocal analysis see, based on Udphuay et al. (2011), [this example](https://dev.pygimli.org/_examples_auto/3_ert/plot_10_reciprocal_analysis.html). 
# 
# We generate an error model by assuming a relative error of 2%.

# %%
# data.estimateError(relativeError=0.02)
data["err"] = 0.02

# %% [markdown]
# For demonstration purposes, we will first work on 2D profiles. Therefore, we extract a single profile from the data using the `subset` command for a constant line along x=0. For 2D inversion, we exchange the x and the y coordinates.

# %%
x0 = data.subset(x=1)
print(x0)
x0.setSensors(np.column_stack([pg.y(x0), pg.x(x0)*0])) # auto for x/y
ax, cb = x0.show()

# %% [markdown]
# The upper part is a single multi-gradient using the outermost electrodes, the lower a dipole-dipole section. If we think one of the values is an outlier, we can remove it by logical operations.

# %%
x0.remove(x0["rhoa"] < 600)
ax, cb = x0.show()

# %%
mgr = ert.Manager(x0)
mgr.invert(verbose=True)
pg.setLogLevel('WARNING')

# %% [markdown]
# The data seem to be fitted well. First, we want to compare measured and modelled data by `showFit()` and in more detail by `showMisfit()`.

# %%
mgr.showFit();

# %% [markdown]
# For details on RMS and chi-square values see [this tutorial](https://www.pygimli.org/_tutorials_auto/3_inversion/plot_1-polyfit.html).
# 
# For a closer look at the differences, we show the misfit distribution:

# %%
mgr.showMisfit()

# %% [markdown]
# There is some limited systematics in the misfit distribution.
# Nevertheless, we feel qualified to have a look at the inversion result.

# %%
# We show the result using a predefined color-scale
kw = dict(cMin=500, cMax=3000, cMap="Spectral_r", 
          logScale=True, coverage=1,
          xlabel="x (m)", ylabel="z (m)")
ax, cb = mgr.showResult(**kw)
# we can  modify the figure using the axis and colorbar handles

# %% [markdown]
# ## Regular grids
# One can, of course, also work with regular meshes.
# Here, we create a regular grid with 10 cm grid spacing.
# In order to ensure accurate boundary conditions in the forward modelling, we append a triangle mesh around it.

# %%
grid = pg.createGrid(x=np.arange(-1, 4.0, 0.1),
                     y=np.arange(-1.5, 0.01, 0.1), 
                     marker=2, 
                     worldBoundaryMarker=True)
ax, cb = pg.show(grid, markers=True, showMesh=True)

# %% [markdown]
# The markers 1 (background) and 2 (inversion) define the default behaviour in the inversion.

# %%
mgr = ert.Manager(x0)
mgr.setMesh(grid)
mgr.invert()
ax, cb = mgr.showResult()
# We save the model for later 
x0result = mgr.paraDomain
x0result["res"] = mgr.model

# %% [markdown]
# We now choose another profile along the y axis.

# %%
y1 = data.subset(y=0.6)
y1.setSensors(y1.sensors() * np.array([1, 0, 0]))
print(y1)
y1.show();

# %%
mgr = ert.Manager(y1)
mgr.invert(zWeight=0.3)
ax, cb = mgr.showResult(**kw)

# %% [markdown]
# ## Mesh generation and regularization
# We now want to improve the mesh a bit, step by step, introducing
# a smaller boundary, a smaller surface discretization `paraDX`, a
# smaller depth, an improved quality and a maximum cell size.

# %%
plc = mt.createParaMeshPLC(y1, paraDX=0.25, paraDepth=1.3, 
                           boundary=0.2, paraMaxCellSize=0.03)
mesh = mt.createMesh(plc, quality=34.4, smooth=True)
ax, cb = pg.show(mesh, markers=True, showMesh=True)

# %%
mgr.setMesh(mesh)
mgr.invert()
ax, cb = mgr.showResult(**kw)

# %% [markdown]
# The default regularization scheme are smoothness constraints of first-order, i.e. a derivative across the model cells. We can also apply second order constraints by setting this inversion property.

# %%
mgr.inv.setRegularization(cType=2)
mgr.invert()
print(mgr.inv.chi2())
ax, cb = mgr.showResult(**kw)

# %% [markdown]
# Alternative to classical smoothness constraints, geostatistical operators can be used for regularization. For details on the method see the paper of Jordi et al. (2018) and [the corresponding tutorial](https://www.pygimli.org/_tutorials_auto/3_inversion/plot_6-geostatConstraints.html).

# %%
mgr.inv.setRegularization(correlationLengths=[1, 0.1])
mgr.invert()
ax, cb = mgr.showResult(**kw)

# %% [markdown]
# For a more extensive comparison of regularization methods, see [this tutorial](https://www.pygimli.org/_tutorials_auto/3_inversion/plot_5_Regularization.html).

# %% [markdown]
# ## Additional information
# ### Structural constraints
# Imagine we have knowledge on a geological interface, e.g. from seismic or GPR reflections. To include this structural information into the inversion, we create a line with a marker>0 and re-create the mesh. 

# %%
line = mt.createLine(start=[0.6, -0.1], 
                     end=[3.5, -0.1], marker=1)
mesh = mt.createMesh(plc+line, quality=34.4)
ax, cb = pg.show(mesh, markers=True, showMesh=True)

# %% [markdown]
# The interface is built into the mesh and deactivates smoothness constraints across it in inversion. For a more rigorous example see
# https://www.pygimli.org/_examples_auto/6_inversion/plot_4_structural_constraints.html
# 
# We create a new manager, set the created mesh and run inversion.

# %%
mgr = ert.Manager(y1)
mgr.setMesh(mesh)
mgr.invert()
ax, cb = mgr.showResult(**kw)

# %% [markdown]
# As a result, there is a sharp contrast in the area of the infiltration, whereas outside the effect of the constraint is lower. 

# %% [markdown]
# ### Region-specific inversion
# The regularization does not have to be the same for the whole modelling domain. We create a cube under the infiltration area for which we use different settings.

# %%
plc = mt.createParaMeshPLC(y1, paraDX=0.25, paraDepth=1.3, 
                           boundary=0.2, paraMaxCellSize=0.03)
cube = mt.createRectangle(start=[0.5, -0.3], end=[3, 0], 
                          marker=3, boundaryMarker=-1)
pg.show(plc+cube, markers=True);

# %%
mesh = mt.createMesh(plc+cube, quality=34.4, smooth=True)
pg.show(mesh, markers=True, showMesh=True);
mgr.setMesh(mesh)

# %%
mgr.invert()
mgr.showResult(**kw);

# %%
mgr = ert.Manager(y1)
mgr.setMesh(mesh)
mgr.inv.setRegularization(1, background=True)
mgr.inv.setRegularization(2, cType=2)
mgr.inv.setRegularization(3, cType=1, zWeight=0.3)
mgr.fop.regionManager().setInterRegionConstraint(2, 3, 1)
mgr.invert()
mgr.showModel(**kw)
# Store the model
y1result = mgr.paraDomain
y1result["res"] = mgr.model
# %% [markdown]
# ## 3D visualization
# Now we bring the 2D result of the y-directed profile  into 3D by changing the dimension, switching y (in 2D depth axis) and z coordinates and shifting the mesh into the correct y position.

# %%
y1result.setDimension(3)
y1result.swapCoordinates(1, 2) # make 2D depth to 3D depth
y1result.translate([0, 0.6, 0])

# %% [markdown]
# Similarly, we do so for the y-directed profile x0, for which we additionally have to exchange the x with the y coordinate

# %%
x0result.setDimension(3)
x0result.swapCoordinates(1, 2) # 2D depth to 3D
x0result.swapCoordinates(0, 1) # 2D x to 3D y
kw.pop('coverage')
kw.pop('xlabel')
kw.pop('ylabel')
# %%
# import matplotlib
# matplotlib.use('TkAgg')  # ³]¸m«áºÝ¬° TkAgg
# pv.set_jupyter_backend('ipyvtklink')  # or 'panel'
kw['off_screen'] = True  # Add this line
pl, _ = pg.show(x0result, "res", **kw, hold=True)
pv.drawMesh(pl, y1result, label="res", **kw, colorBar=False)
pl.show()
# %% [markdown]
# For a more thorough overview on the region options see [this tutorial](https://www.pygimli.org/_tutorials_auto/3_inversion/plot_8-regionWise.html).

# %% [markdown]
# ### Incorporation of petrophysics
# 
# Already in pyGIMLi 1.0 (R?cker et al., 2017), we demonstrated the incorporation of petrophysical relations in the inverse problem. In our case, we are interested in the water saturation that is described by Archie's law. To this end, there is an already defined transformation function [`transFwdArchieS`](https://www.pygimli.org/pygimliapi/_generated/pygimli.physics.petro.html#pygimli.physics.petro.transFwdArchieS). We import the function and transfer the modelled resistivity into saturation.
# For the Archie equation, we use values derived by H?bner et al. (2015).

# %%
from pygimli.physics.petro import transFwdArchieS
tS = transFwdArchieS(rFluid=66*(1-0.02*20), 
                     phi=0.4, m=1.3, n=1.83)
satKW = dict(cMin=0, cMax=0.5, logScale=0, cMap="Blues",
             label="saturation (-)", coverage=1)
mgr.showResult(tS.inv(mgr.model), **satKW);

# %% [markdown]
# Next, we combine it with the ERT forward operator using the `PetroModelling` framework.

# %%
mgr = ert.Manager(y1, verbose=False)
mesh = mt.createMesh(plc, quality=34.4, smooth=True)
mgr.setMesh(mesh)
fop = pg.frameworks.PetroModelling(mgr.fop, tS)

# %%
fop.setMesh(mesh)
inv = pg.Inversion(fop=fop)
# inv.setRegularization(1, background=True)
# inv.setRegularization(2, limits=[0, 1], zWeight=0.3)
model = inv.run(y1["rhoa"], relativeError=y1["err"], 
                startModel=0.2)
pg.show(mgr.paraDomain, model, **satKW);

# %% [markdown]
# ### Parameter constraints
# Besides from structural information, one may have parameter information, e.g. from a borehole or direct-push sounding. For an example, see
# https://www.pygimli.org/_examples_auto/6_inversion/plot_5_ert_with_priors.html
# 
# We define each four equidistant points in 10 cm depth and in 50 cm depth and assume a saturation of 0.5 for the upper ones and the lower ones.

# %%
xpos = [1, 1.5, 2, 2.5]
pos = np.array([[x, -0.1] for x in xpos] + 
               [[x, -0.5] for x in xpos])
pointSat = pg.cat(np.ones_like(xpos) * 0.5, 
                  np.ones_like(xpos) * 0.2)

# %% [markdown]
# We generate a prior modelling operator that indexes into a given mesh, here the inversion domain.

# %%
invmesh = mgr.paraDomain
invmesh["marker"] = 0
pointFop = pg.frameworks.PriorModelling(invmesh, pos)

# %% [markdown]
# Now we combine the ERT forward operator and the prior mapping operator by the `JointModelling` framework.
# We set the individual data for the two operators and run a (joint) inversion for the new modelling operator.

# %%
fopJoint = pg.frameworks.JointModelling([fop, pointFop])
fopJoint.setData([y1, pointSat])
invJoint = pg.Inversion(fopJoint)
invJoint.setRegularization(limits=[0, 1])
dataVec = pg.cat(y1["rhoa"], pointSat)
errorVec = pg.cat(y1["err"], 0.03/pointSat)
invJoint.setRegularization(cType=2) #correlationLengths=[1, .2])
modelJoint = invJoint.run(dataVec, relatativeError=errorVec, 
                          lam=300, startModel=0.2)
ax, cb = pg.show(invmesh, modelJoint, **satKW)
ax.scatter(pos[:, 0], pos[:, 1], c=pointSat, s=20, 
           vmin=satKW["cMin"], vmax=satKW["cMax"], 
           cmap=satKW["cMap"], ec="yellow");



# %% [markdown]
# ## 3D inversion
# For a 3D inversion of all data, one needs to create a 3D mesh. To do so, first the geometry is created and then meshed (just like in 2D).

# %%
plc = mt.createParaMeshPLC3D(data)
mesh = mt.createMesh(plc, quality=1.3)
print(mesh)
# pg.show(mesh, style="wireframe")

# %%
mgr3d = ert.Manager(data)
mgr3d.setMesh(mesh)
mgr3d.invert(verbose=True)

# %%
result3d = mgr3d.paraDomain
result3d["res"] = mgr3d.paraModel()
pl, _ = pg.show(result3d, label="res", style="surface", hold=True, **kw,
                filter={"threshold": dict(value=400, scalars="res", method="lower")})
pv.drawMesh(pl, result3d, label="res", style="surface", **kw,
            filter={"slice": dict(normal="y", origin=[0, 1, 0])})
pl.show()

# %% [markdown]
# ## Timelapse inversion
# 
# For this data set, we have a lot of data over a couple of days.
# Of course, we could invert these step by step and compare the
# individual models.
# For timelapse ERT, there is a specialized class `TimelapseERT`.
# It handles the data, e.g. filtering and masking, but also 
# exhibits several timelapse strategies, such as 4D inversion.
# 
# For more details, see the example on [time-lapse ERT](https://www.pygimli.org/_examples_auto/3_ert/plot_09_ert_timelapse.html).

# %%
DATA = []
for nr in [0, 1, 2, 4, 10, 40]:
    data = pg.getExampleData(f"ert/Huebner2017/{nr:03d}.dat")
    data2d = data.subset(y=1.4)
    data2d.setSensors(data2d.sensors() * np.array([1, 0, 0]))
    DATA.append(data2d)

tl = ert.TimelapseERT(DATA)
print(tl)

# %%
tl.invert(paraDepth=1.3, correlationLengths=[1, 0.4])
tl.showAllModels(ratio=True, rMax=2, orientation="vertical");

# %%
tl.fullInversion(C=tl.mgr.fop.constraints())
tl.showAllModels(ratio=True, rMax=2, orientation="vertical");

# %% [markdown]
# For a more rigorous explanation of time-lapse strategies see
# 
# https://www.pygimli.org/_examples_auto/3_ert/plot_09_ert_timelapse.html

# %%
# tl = ert.TimelapseERT("*.dat")
# print(tl)
# tl.fullInversion()

# %% [markdown]
# ## References
# * H?bner, R., G?nther, T., Heller, K., Noell, U. & Kleber, A. (2017): Impacts of a capillary barrier on infiltration and subsurface stormflow in layered slope deposits monitored with 3-D ERT and hydrometric measurements. Hydrol. Earth Syst. Sci. 21, 5181-5199, [doi:10.5194/hess-21-5181-2017](https://doi.org/10.5194/hess-21-5181-2017).
# * Jordi, C., Doetsch, J., G?nther, T., Schmelzbach, C. & Robertsson, J.O.A. (2018): Geostatistical regularisation operators for geophysical inverse problems on irregular meshes. Geophys. J. Int. 213, 1374-1386, [doi:10.1093/gji/ggy055](https://doi.org/10.1093/gji/ggy055).
# * Gr?nenbaum, N., G?nther, T., Greskowiak, J., Vienken, T., M?ller-Petke, M. & Massmann, G. (2023): Salinity distribution in the subterranean estuary of a meso-tidal high-energy beach characterized by Electrical Resistivity Tomography and Direct Push technology. J. of Hydrol. 617, 129074, [doi:10.1016/j.jhydrol.2023.129074](https://doi.org/10.1016/j.jhydrol.2023.129074).
# * R?cker, C., G?nther, T., Wagner, F.M. (2017): pyGIMLi: An open-source library for modelling and inversion in geophysics, Computers & Geosciences 109, 106-123, [doi:10.1016/j.cageo.2017.07.011](https://doi.org/10.1016/j.cageo.2017.07.011).
# * H?bner, R., Heller, K., G?nther, T. & Kleber, A. (2015): Monitoring hillslope moisture dynamics with sur- face ERT for enhancing spatial significance of hydrometric point measurements. Hydrology and Earth System Sciences 19(1), 225-240, [doi:10.5194/hess-19-225-2015](https://doi.org/10.5194/hess-19-225-2015).
# * Udphuay, S., G?nther, T., Everett, M.E., Warden, R.R. & Briaud, J.-L. (2011): Three-dimensional resistivity tomography in extreme coastal terrain amidst dense cultural signals: application to cliff stability assessment at the historic D-Day site. Geophys. J. Int. 185(1), 201-220, [doi:10.1111/j.1365-246X.2010.04915.x](https://doi.org/10.1111/j.1365-246X.2010.04915.x).
# 
# 
# 


