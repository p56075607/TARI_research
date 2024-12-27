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
sys.path.append(r'E:\研究室電腦E槽\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\PYGIMLY\field_data\SANSIN')
from stg2ohm_pp import stg2ohm
from pygimli.viewer import pv
import matplotlib.pyplot as plt
import pygimli.meshtools as mt

# %%
survey_name = '111801'
stg_fname = join(survey_name,survey_name+'.stg')#r'0910\091011.stg'
cmd_path = r'091011\091011.cmd'
output_ph = join('output',survey_name)
if not os.path.exists(output_ph):
    os.makedirs(output_ph)
ohm_file_name = stg2ohm(cmd_path,stg_fname,113,114,is_3D=True)

# %%
data = ert.load(ohm_file_name)
data['k'] = ert.createGeometricFactors(data, numerical=True)
data['rhoa'] = data['r'] * data['k']
data['err'] = ert.estimateError(data, relativeError=0.02)
# data.save(stg_fname[:-4]+'.ohm')
print(data)
# %%
t2 = data['a'] == 89
index = [i for i, x in enumerate(t2) if x]
print(r'remove a == ch90 {:d}'.format(len(index)))
data.remove(t2)

t2 = data['m'] == 89
index = [i for i, x in enumerate(t2) if x]
print(r'remove a == ch90 {:d}'.format(len(index)))
data.remove(t2)
print(data)
# %%

# data.show(style="A-M")
plt.plot(pg.x(data), pg.y(data), ".")

# %%

plc = mt.createParaMeshPLC3D(data, paraDX=1/3.5, paraDepth=10, paraMaxCellSize=10,
                             surfaceMeshQuality=30)
mesh = mt.createMesh(plc, quality=1.3)
print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarker() == 2) if x]))
# pg.show(mesh, style="wireframe")

# %%
def plot_convergence(mgr, output_ph):
    rrmsHistory = mgr.inv.rrmsHistory
    chi2History = mgr.inv.chi2History
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(np.linspace(1,len(rrmsHistory),len(rrmsHistory)),rrmsHistory, linestyle='-', marker='o',c='black')
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('rRMS (%)')
    ax.set_title('Convergence Curve of Resistivity Inversion')
    ax2 = ax.twinx()
    ax2.plot(np.linspace(1,len(rrmsHistory),len(rrmsHistory)),chi2History, linestyle='-', marker='o',c='blue')
    ax2.set_ylabel('$\chi^2$',c='blue')
    ax.grid()
    fig.savefig(join(output_ph,'CONV.png'),dpi=300, bbox_inches='tight')
    plt.close(fig)

    return rrmsHistory, chi2History

def crossplot(mgr,output_ph):
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
    ax.set_title(r'Crossplot of Measured vs Predicted Resistivity $\rho_{apparent}$')
    fig.savefig(join(output_ph,'CROSP.png'),dpi=300, bbox_inches='tight')
    plt.close(fig)

def data_misfit(mgr, output_ph):
    mgr.data['misfit'] = np.abs((mgr.inv.response - mgr.data["rhoa"])/mgr.data["rhoa"])*100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(mgr.data['misfit'],np.linspace(0,100,21))
    ax.set_xticks(np.linspace(0,100,21))
    ax.set_xlabel('Relative Data Misfit (%)')
    ax.set_ylabel('Number of Data')
    ax.set_title('Data Misfit Histogram for Removal of Poorly-Fit Data')
    fig.savefig(join(output_ph, 'HIST.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def export_inversion_info(mgr, output_ph, lam, rrmsHistory, chi2History):
    information_ph = join(output_ph,'ERTManager','inv_info.txt')
    with open(information_ph, 'w') as write_obj:
        write_obj.write('## Final result ##\n')
        write_obj.write('rrms:{}\n'.format(mgr.inv.relrms()))
        write_obj.write('chi2:{}\n'.format(mgr.inv.chi2()))

        write_obj.write('## Inversion parameters ##\n')
        write_obj.write('use lam:{}\n'.format(lam))

        write_obj.write('## Iteration ##\n')
        write_obj.write('Iter.  rrms  chi2\n')
        for iter in range(len(rrmsHistory)):
            write_obj.write('{:.0f},{:.2f},{:.2f}\n'.format(iter,rrmsHistory[iter],chi2History[iter]))

# %%
repeat_time = 3

for repeat in range(repeat_time):
    lam = 1000
    mgr3d = ert.ERTManager(data)

    mgr3d.invert(data,mesh=mesh,
                lam=lam  ,zWeight=1,
                maxIter = 10,
                verbose=True)

    rrms = mgr3d.inv.relrms()
    chi2 = mgr3d.inv.chi2()
    pg.boxprint('rrms={:.2f}%, chi^2={:.3f}'.format(rrms, chi2))

    # Creat output folder
    out_folder = join(output_ph,'repeat_'+str(repeat+1))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f'Folder "{out_folder}" created.')

    mgr3d.saveResult(out_folder)
    rrmsHistory, chi2History = plot_convergence(mgr3d, out_folder)
    crossplot(mgr3d, out_folder)
    data_misfit(mgr3d, out_folder)
    export_inversion_info(mgr3d, out_folder, lam, rrmsHistory, chi2History)
    # Export data used in this inversion 
    mgr3d.data.save(join(out_folder,'ERTManager','inverison_data.ohm'))
    # Export model response in this inversion 
    pg.utils.saveResult(join(out_folder,'ERTManager','model_response.txt'),
                        data=mgr3d.inv.response, mode='w')

    # Repeat and delete poor-fit data
    if (repeat_time>1): # 準備要第二次反演時
        remain_per = 0.95
        t1 = np.argsort(data['misfit'])[int(np.round(remain_per*len(data['rhoa']),0)):]
        remove_index = np.full((len(data['rhoa'])), False)
        for j in range(len(t1)):
            remove_index[t1[j]] = True
        print(r'remove {:d}% worst misfit data, rest data {:d}'.format(int(100*(1-remain_per)),len(data['rhoa'])-len(t1)))
        print(r'remove {:d} bad-fitting data, from misfit {:.2f}% above'.format(len(t1),data['misfit'][t1[0]]))
        data.remove(remove_index)
# %%
