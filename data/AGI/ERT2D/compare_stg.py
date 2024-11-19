# %%
import pygimli as pg
from pygimli.physics import ert
import numpy as np
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from ohm2urf import ohm2urf
from os.path import join
import matplotlib.pyplot as plt

# %%
stg_fname = join(r"C:\Users\Git\TARI_research\data\AGI\ERT2D\110702\110702.stg")
data1 = ert.load(stg_fname)
print(data1)
stg_fname = join(r"C:\Users\Git\TARI_research\data\AGI\ERT2D\110501\110501.stg")
data2 = ert.load(stg_fname)
print(data2)

DATA = [data1,data2]
# Four electrode numbers for each data set
sets_of_quadruples = []

for data in DATA:
    quadruples = set(zip(data['a'], data['b'], data['m'], data['n']))
    sets_of_quadruples.append(quadruples)

# find the common quadruples
common_quadruples = set.intersection(*sets_of_quadruples)

# delete the data with quadruples not in common_quadruples
remove_indices_list = []
i = 0
filtered_DATA = DATA.copy()
for data in DATA:
    quadruples = list(zip(data['a'], data['b'], data['m'], data['n']))
    remove_indices = [quadruple not in common_quadruples for quadruple in quadruples]
    remove_indices_list.append(remove_indices)
    filtered_DATA[i].remove(remove_indices)
    print(filtered_DATA[i])
    i += 1

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(np.log10(filtered_DATA[0]['rhoa']),np.log10(filtered_DATA[1]['rhoa']),s=1,c=np.linspace(0,1,len(filtered_DATA[1]['rhoa'])),cmap='jet')
xticks = ax.get_xlim()
yticks = ax.get_ylim()
lim = max(max(yticks,xticks)) + 0.5
ax.plot([0,lim],[0,lim],'k-',linewidth=1, alpha=0.2)
ax.set_xlim([0,lim])
ax.set_ylim([0,lim])
