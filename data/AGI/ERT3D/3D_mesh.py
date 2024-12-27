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
import matplotlib.pyplot as plt
import pygimli.meshtools as mt

# %%
survey_name = '111801'
ohm_file_name = join(survey_name,survey_name+'.ohm')
data = ert.load(ohm_file_name)

# %%
tmp=np.asarray(data.sensors())
tmp[:,2]=tmp[:,2]-0.2

plc1 = mt.createParaMeshPLC3D(data, paraDepth=5, paraMaxCellSize=0.1,
                         surfaceMeshQuality=34)

plc2=mt.createParaMeshSurface(tmp,surfaceMeshQuality=34,surfaceMeshArea=0.1)

plc=plc1+plc2

pg.show(plc1,style="wireframe")