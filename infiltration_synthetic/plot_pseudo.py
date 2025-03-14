# %%
import os
import sys
from resipy import Project
import matplotlib.pyplot as plt
import numpy as np
# set font to mircosoft yahei
plt.rcParams['font.family'] = 'Microsoft YaHei'
# %%
# list the files in the dataset folder: C:\Users\Git\TARI_research\infiltration_synthetic\inverison_data_case1
files = os.listdir(r'C:\Users\Git\TARI_research\infiltration_synthetic\inverison_data_case1')

# %%
for file in files[-2:]:
    ERT = Project(typ='R2')  # create a Project object in a working directory
    ERT.createSurvey(os.path.join('inverison_data_case1', file), ftype='BERT')  # read the survey file
    print(f'File: {file}')
    print('Total data numbers: ', len(ERT.surveys[0].df))
    
    # Plot pseudo section
    xpos, _, ypos = ERT.surveys[0]._computePseudoDepth()
    resist = np.log10(ERT.surveys[0].df['app'])
    vmin = 2
    vmax = 3
    fig, ax = plt.subplots(figsize=(10,5))
    scatter = ax.scatter(xpos,-ypos,c=resist,s=10,cmap='jet',vmin=vmin,vmax=vmax)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Pseudo section - {file}')
    ax.set_aspect('equal')
    # plot the electrode location ERT.surveys[0].elec
    ax.plot(ERT.surveys[0].elec['x'],ERT.surveys[0].elec['y'],'k.',markersize=10)

    plt.show()
# %%
