#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ERT inversion of data measured on a standing tree
=================================================

This example is about inversion of ERT data around a hollow lime tree.
It applies the BERT methodology of Günther et al. (2006) to trees,
first documented by Göcke et al. (2008) and Martin & Günther (2013).
It has already been used in the BERT tutorial (Günther & Rücker, 2009-2023)
in chapter 3.3 (closed geometries).

- generate circular geometry
- geometric factor generation
- circular data representation
- inversion
"""
# sphinx_gallery_thumbnail_number = 5

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

###############################################################################
# Get some example data with topography, typically by a call like
# data = ert.load("filename.dat")
# that supports various file formats
data = pg.getExampleFile('ert/hollow_limetree.ohm', load=True, verbose=True)
print(data)

###############################################################################
# It uses 24 electrodes around the circumference and applies a dipole-dipole
# (AB-MN) array. For each of the 24 dipoles being used as current injection
# (AB), 21 potential dipoles (from 3-4 to 23-24) can be measured of which 10
# are measured, so that we end up in a total of 24*11=264 measurements.
# Apart from current ('a' and 'b') and potential ('m', 'n') electrodes, the
# file contains current ('i'), voltage ('u') and resistance ('r') vectors for
# each of the 264 data.
#
# We first generate the geometry by creating a close polygon.
# Between each two electrodes, we place three additional nodes whose positions
# are interpolated using a spline.
plc = mt.createPolygon(data.sensors(), isClosed=True,
                       addNodes=3, interpolate='spline')
ax, cb = pg.show(plc)

###############################################################################
# From this geometry, we create a triangular mesh with a quality factor, a
# maximum triangle area and post-smoothing.
mesh = mt.createMesh(plc, quality=34.3, area=2e-4, smooth=[10, 1])
print(mesh)
ax, _ = pg.show(mesh)
for i, s in enumerate(data.sensors()):
    ax.text(s.x(), s.y(), str(i+1), zorder=100)

###############################################################################
# We first create the geometric factors to multiply the resistances to obtain
# "mean" resistivities for every quadrupol. We do this numerically using a
# refined mesh with quadratic shape functions using a constant resistivity of 1.
# The inverse of the modelled resistances is the geometric factor so that all
# apparent resistivities become 1.
# We then generate apparent resistivity, store it in the data and show it.
data["k"] = ert.createGeometricFactors(data, mesh=mesh, numerical=True)
data["rhoa"] = data["r"] * data["k"]
ax, cb = ert.show(data, "rhoa", circular=True)

###############################################################################
# Note that we use the option circular. The values have almost rotation symmetry
# with lower values for shallow penetrations and higher ones for larger
# current-voltage distances.
#
# We estimate a measuring error using default values (3%) and feed it
# into the ERT Manager.
data.estimateError()
mgr = ert.Manager(data)
mgr.invert(mesh=mesh, verbose=True)

###############################################################################
# We achieve a fairly good data fit with a chi-square value of 2 and compare
# data and model response, both are looking very similar.
ax = mgr.showFit(circular=True)

###############################################################################
#%% Finally, we have a look at the resulting resistivity distribution.
# It clearly shows cavity inside as high resistivity.
ax, cb = mgr.showResult(cMin=100, cMax=500, coverage=1,cMap='jet',logScale=True,sensors=False)
# %%
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Times New Roman'
ax,cb = pg.show(mgr.paraDomain, mgr.model, label=pg.unit('res'),
                 cMap='jet', logScale=True,cMin=100, cMax=500,
                 orientation='vertical')
for i, s in enumerate(data.sensors()):
    ax.plot(s.x(), s.y(), 'ko', zorder=100)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
###############################################################################
# References
# ----------
# - Günther, T., Rücker, C. & Spitzer, K. (2006): Three-dimensional modeling &
#   inversion of dc resistivity data incorporating topography – II: Inversion.
#   Geophys. J. Int. 166, 506-517, doi:10.1111/j.1365-246X.2006.03011.x
# - Göcke, L., Rust, S., Weihs, U., Günther, T. & Rücker, C. (2008): Combining
#   sonic and electrical impedance tomography for the nondestructive testing of
#   trees. - Western Arborist, Spring/2008: 1-11.
# - Martin, T. & Günther, T. (2013): Complex Resistivity Tomography (CRT) for
#   fungus detection on standing oak trees. European Journal of Forest Research
#   132(5), 765-776, doi:10.1007/s10342-013-0711-4
#%%
c1 = mt.createPolygon([[-0.15,0],[0,0.1],[0.1,0.1],[0.05,-0.1],[-0.05,-0.05],[-0.15,0]],isClosed=True,
                     marker = 2) #mt.createCircle(pos=(0.0, 0.0), radius=0.1, segments=25, marker=2)
geom = plc+ c1
# pg.show(geom)
mesh_fwd = mt.createMesh(geom, area=2e-5, smooth=[10, 1])
pg.show(mesh_fwd)
# %%
import numpy as np
model_rho = np.array([100,10000])[mesh_fwd.cellMarkers()-1]
data_sim = ert.simulate(mesh_fwd, scheme=data, res=model_rho, noiseLevel=0.01,
                    noiseAbs=1e-6, seed=1337)
# %%
mgr2 = ert.Manager(data_sim)
mgr2.invert(mesh=mesh, verbose=True)
# %%
ax,cb = pg.show(mgr2.paraDomain, mgr2.model, label=pg.unit('res'),
                 cMap='jet', logScale=True,cMin=100, cMax=1000,
                 orientation='vertical')
for i, s in enumerate(data.sensors()):
    ax.plot(s.x(), s.y(), 'ko', zorder=100)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# %%
