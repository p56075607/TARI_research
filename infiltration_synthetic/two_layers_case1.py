# %%
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt
import numpy as np
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from matplotlib.colors import LinearSegmentedColormap
import os
import matplotlib.pyplot as plt
from plot_util import  plot_inverted_profile, plot_inverted_contour, plot_convergence, crossplot, data_misfit, export_inversion_info
from datetime import datetime, timedelta
from os import listdir
from os.path import isdir, join
output_path = r"inverison_results_case1"
# %%
ohm_file = r"D:\R2MSDATA\TARI_E1_test\output_second_inversion\24030215_m_E1\ERTManager\inverison_data.ohm"
data = ert.load(ohm_file)
# %%
# save path check and create
save_path = r"inverison_data_case1"
if not os.path.exists(save_path):
    os.makedirs(save_path)
left = min(pg.x(data))
right = max(pg.x(data))
depth = 10

rho1 = 200
rho2 = 1000
rho1w = 100
rho2w = 400
kw = dict(cMin=rho1w, cMax=rho2, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')

inf_depths = np.concatenate((np.linspace(-2, 0, 11)[::-1], np.linspace(-10, -2, 9)[::-1][1:])).round(1) # 合併並且刪除重複的-2

DATA = []
for n,inf_depth in enumerate(inf_depths):
    # check the data if exist continue
    if os.path.exists(join(save_path, f'data_case1_infdepth_{inf_depth:.1f}.ohm')):
        continue

    if (inf_depth > -2) & (inf_depth != 0):
        world = mt.createWorld(start=[left, 0], end=[right, -depth],
                            layers=[inf_depth,-2], 
                            worldMarker=True)
    elif (inf_depth == -2) | (inf_depth == 0) | (inf_depth == -10):
        world = mt.createWorld(start=[left, 0], end=[right, -depth],
                            layers=[-2], 
                            worldMarker=True)
    else:
        world = mt.createWorld(start=[left, 0], end=[right, -depth],
                            layers=[-2,inf_depth], 
                            worldMarker=True)
    mesh = mt.createMesh(world, quality=34, area=0.01)

    if inf_depth == 0: # 2,3
        rhomap = [[1, rho1], [2, rho2]]
    elif inf_depth == -0.2: #023
        rhomap = [[0, rho1w],[2, rho1], [3, rho2]]
    elif (inf_depth <-0.2) & (inf_depth > -2): #123
        rhomap = [[1, rho1w], [2, rho1], [3, rho2]]
    elif inf_depth == -2: #12
        rhomap = [[1, rho1w], [2, rho2]]
    elif (inf_depth < -2) & (inf_depth > -10): #123
        rhomap = [[1, rho1w], [2, rho2w], [3, rho2]]
    else: #12
        rhomap = [[1, rho1w], [2, rho2w]]

    ax, cb = pg.show(mesh, 
        data=rhomap, 
        showMesh=True,
        markers=True,
        **kw)
    ax.set_ylim(-10, 0)
    ax.set_xlim(19, 21)

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
    
    # save data
    data_i.save(join(save_path, f'data_case1_infdepth_{inf_depth:.1f}.ohm'))
# %%
# Inversion mesh
left = min(pg.x(data))
right = max(pg.x(data))
length = right - left
depth = length/4
plc = mt.createParaMeshPLC(data,quality=34,paraDX=1/3,paraMaxCellSize=1
                            ,paraDepth=depth)
mesh = mt.createMesh(plc)
print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarker() == 2) if x]))
# %%
# Inversion
# output_ph check and create

if not os.path.exists(output_path):
    os.makedirs(output_path)
for n,inf_depth in enumerate(inf_depths):
    time = (datetime(2025, 1, 1, 0, 0) + timedelta(hours=n)).strftime("%y%m%d%H")
    print(f'Inversion {n+1} of {len(inf_depths)}, time: {time}')
    if os.path.exists(join(output_path, f'{time}')):
        continue
    
    lam = 1000
    mgr = ert.ERTManager(DATA[0])
    model = mgr.invert(DATA[0],mesh=mesh,
                        lam=lam  ,zWeight=1,
                        maxIter = 10,
                        verbose=True)
    rrms = mgr.inv.relrms()
    chi2 = mgr.inv.chi2()
    pg.boxprint('rrms={:.2f}%, chi^2={:.3f}'.format(rrms, chi2))
    
    
    path, fig, ax = mgr.saveResult(join(output_path,f'{time}'))
    plt.close(fig)
    save_path = join(output_path,f'{time}')
    # Export the information about the inversion
    export_inversion_info(mgr, save_path, lam, mgr.inv.rrmsHistory, mgr.inv.chi2History)

    # Export data used in this inversion 
    mgr.data.save(join(save_path,'ERTManager','{inf_depth:.1f}_inv.ohm'))
    # Export model response in this inversion 
    pg.utils.saveResult(join(save_path,'ERTManager','model_response.txt'),
                        data=mgr.inv.response, mode='w')  
    # Compute Model resolution
    pg.utils.saveResult(join(save_path,'ERTManager','model_resolution.vector'),
                        data=pg.frameworks.resolution.resolutionMatrix(mgr.inv).diagonal(), mode='w')  
    # Plot the inverted profile

    urf_file_name = f'{time}.urf'
    fig = plot_inverted_profile(save_path, urf_file_name, lam, rrms, chi2, **kw)

    # Plot inverted contour profile
    plot_inverted_contour(save_path, urf_file_name, lam, rrms, chi2, fig, **kw)

    # Convergence Curve of Resistivity Inversion
    rrmsHistory, chi2History = plot_convergence(save_path)

    # Varify the fitted and measured data cross plot
    crossplot(save_path)

    # Data Misfit Histogram for Removal of Poorly-Fit Data
    data_misfit(save_path)

# %%
# read the results file 
output_folders = sorted([f for f in listdir(output_path) if isdir(join(output_path,f))])
def load_1inversion_data(save_ph):
    output_ph = join(save_ph,'ERTManager')

    para_domain = pg.load(join(output_ph,'resistivity-pd.bms'))
    model = pg.load(join(output_ph,'resistivity.vector'))

    mgr_dict = {'paraDomain': para_domain, 
                'model': model}

    return mgr_dict

all_mgrs = []
for i,output_folder_name in enumerate(output_folders):
    print(output_folder_name)
    all_mgrs.append(load_1inversion_data(join(output_path,output_folder_name)))
# %%
# Calculate the difference, the first one is the reference
# find the value between x = 19.5 and x = 20.5, and y = -10 and y = 0, x_coord=mgr['paraDomain'].cellCenter()[:,0] and y_coord=mgr['paraDomain'].cellCenter()[:,1]
x_coord = all_mgrs[0]['paraDomain'].cellCenter()[:,0]
y_coord = all_mgrs[0]['paraDomain'].cellCenter()[:,1]

target_x = (x_coord >= 19.5) & (x_coord <= 20.5)


# 同時考慮x和y的條件
target_cells = target_x

x_coord_target = x_coord[target_cells]
y_coord_target = y_coord[target_cells]

# 先對y座標進行排序，獲取排序索引
sort_idx = np.argsort(y_coord_target)
y_coord_sorted = y_coord_target[sort_idx]

DIFF_1D_target = []
for i in range(len(all_mgrs)):
    DIFF = (100*(np.log10(all_mgrs[i]['model'])-np.log10(all_mgrs[0]['model']))/np.log10(all_mgrs[0]['model']))
    # 選取目標區域的差異值
    DIFF_target = DIFF[target_cells]
    # 使用相同的排序索引排序差異值
    DIFF_1D_target.append(DIFF_target[sort_idx])

DIFF_1D_target = np.array(DIFF_1D_target).T

# 繪製圖形
time_axis = np.linspace(0, 10, 11)
time_axis = np.append(time_axis, np.linspace(15, 50, 8)) 


colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 1-(-3/-13), 1]  
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))     
kw = dict(cMin=-13, cMax=0,logScale=False,
            label='Relative resistivity difference \n(%)',
            xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)       
Tmesh, Ymesh = np.meshgrid(time_axis, y_coord_sorted)
fig, ax = plt.subplots(figsize=(27.5, 12))
contour = ax.contourf(Tmesh, Ymesh, DIFF_1D_target, levels=20, cmap=kw['cMap'], vmin=kw['cMin'], vmax=kw['cMax'])
ax.set_ylim(-10,0)
ax.set_xlim(0,50)
fontsize = 30
ax.set_xlabel('Time (hrs)', fontsize=fontsize)
ax.set_ylabel('Depth (m)', fontsize=fontsize)

cb = fig.colorbar(contour, pad=0.03)
cb.ax.set_ylim(kw['cMin'],kw['cMax'])
cb.ax.set_yticks(np.linspace(kw['cMin'],kw['cMax'],6))
cb.ax.set_ylabel(kw['label'],fontsize=fontsize,fontweight='bold')
cb.ax.yaxis.set_tick_params(labelsize=fontsize)
for label in cb.ax.yaxis.get_ticklabels():
    label.set_fontweight('bold')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)

# y ticks each 1 m
ax.set_yticks(np.arange(-10, 0, 1))

ax.tick_params(labelsize=fontsize)
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')

ax.grid(True, which='major', linestyle='--', linewidth=0.5)

# add vertical line for each time
for i in range(len(time_axis)):
    ax.axvline(time_axis[i], color='black', linewidth=1, linestyle='--')

# add the ture line slope = -10/45
ax.plot(time_axis, -10/50*time_axis, color='red', linewidth=2)

plt.show()









