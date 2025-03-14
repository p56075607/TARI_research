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
from plot_util import  plot_inverted_profile, plot_inverted_contour,Clip_cornor, plot_convergence, crossplot, data_misfit, export_inversion_info
from datetime import datetime, timedelta
from os import listdir
from os.path import isdir, join
from matplotlib import animation
import matplotlib.collections
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
    # if os.path.exists(join(save_path, f'data_case1_infdepth_{inf_depth:.1f}.ohm')):
    #     continue

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
    mesh = mt.createMesh(world, quality=34, area=1)

    if inf_depth == 0: # 2,3
        rhomap = [[1, rho1], [2, rho2]]
    elif inf_depth == -0.2: #023
        rhomap = [[1, rho1w],[2, rho1], [3, rho2]]
    elif (inf_depth <-0.2) & (inf_depth > -2): #123
        rhomap = [[1, rho1w], [2, rho1], [3, rho2]]
    elif inf_depth == -2: #12
        rhomap = [[1, rho1w], [2, rho2]]
    elif (inf_depth < -2) & (inf_depth > -10): #123
        rhomap = [[1, rho1w], [2, rho2w], [3, rho2]]
    else: #12
        rhomap = [[1, rho1w], [2, rho2w]]
    fig, ax = plt.subplots(figsize=(3,6))
    ax, cb = pg.show(mesh,ax=ax, 
        data=rhomap, 
        showMesh=False,
        markers=False,
        **kw)
    ax.set_ylim(-10, 0)
    ax.set_xlim(20,25)
    fontsize = 13
    ax.set_title(f'Infiltration depth: {-inf_depth:.1f} m', fontsize=fontsize)
    ax.set_xlabel('Distance (m)', fontsize=fontsize)
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    # ax.set_yticks(np.arange(-10, 0, 1))
    ax.tick_params(labelsize=fontsize)
    ax.tick_params(axis='both', which='major', length=5,width=1, direction='in')
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel(kw['label'],fontsize=fontsize,fontweight='bold')
    ax.set_aspect('equal')



    # data_i = ert.simulate(mesh=mesh, scheme=data, res=rhomap, noiseLevel=1,
    #                         noiseAbs=1e-6, 
    #                         seed=1337) 
    # def print_data_siminfo(data_sim):
    #     pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
    #     pg.info('Simulated data', data_sim)
    #     pg.info('The data contains:', data_sim.dataMap().keys())
    #     pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
    #     pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random seed for repeatable noise in synthetic experiments 
    # print_data_siminfo(data_i)
    # data_i['r'] = data_i['rhoa']/data_i['k']
    # DATA.append(data_i)
    
    # # save data
    # data_i.save(join(save_path, f'data_case1_infdepth_{inf_depth:.1f}.ohm'))
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
            label='Relative resistivity \ndifference (%)',
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
cb.ax.set_yticks(np.linspace(kw['cMin'],kw['cMax'],14))
cb.ax.set_ylabel(kw['label'],fontsize=fontsize)
cb.ax.yaxis.set_tick_params(labelsize=fontsize)

width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)

# y ticks each 1 m
ax.set_yticks(np.arange(-10, 0, 1))

ax.tick_params(labelsize=fontsize)
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
# x ticks [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50])

# add horizontal line for depth = [-0.5, -1.5, -2.5, -4]
ax.axhline(-0.5, color='black', linewidth=1, linestyle='--')
ax.axhline(-1.5, color='black', linewidth=1, linestyle='--')
ax.axhline(-2.5, color='black', linewidth=1, linestyle='--')
ax.axhline(-4, color='black', linewidth=1, linestyle='--')

# add vertical line for each time
for i in range(len(time_axis)):
    ax.axvline(time_axis[i], color='black', linewidth=1, linestyle='--')

# add the ture line slope = -10/45
ax.plot(time_axis, -10/50*time_axis, color='red', linewidth=2)
# points each times and depth
for i in range(len(time_axis)):
    ax.scatter(time_axis[i], -10/50*time_axis[i], color='red', marker='o', s=100)
plt.show()
fig.savefig(join('two_layers_case1_diff.png'),dpi=300,bbox_inches='tight')

# %%
# plot the resistivity difference profile
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'Microsoft YaHei'
def plot_difference_contour(mgr1, mgr2, urf_file_name1, urf_file_name2, **kw_diff):
    model1 = mgr1['model']
    model2 = mgr2['model']
    mesh_x = np.linspace(left, right, 250)
    mesh_y = np.linspace(-depth, 0, 150)
    X,Y = np.meshgrid(mesh_x, mesh_y)
    grid = pg.createGrid(x=mesh_x, y= mesh_y)
    one_line_diff = (np.log10(model2) - np.log10(model1))/np.log10(model1)*100
    diff_grid = np.reshape(pg.interpolate(mgr1['paraDomain'], one_line_diff, grid.positions()), (len(mesh_y), len(mesh_x)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(X,Y, diff_grid, cmap=kw_diff['cMap'], levels=20,
                vmin=kw_diff['cMin'],vmax=kw_diff['cMax'],antialiased=True)
    ax.set_aspect('equal')
    ax.set_xlim(left, right)
    ax.set_ylim(-depth, 0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.set_xlabel(kw['xlabel'])#+' max_abs:{:.2f}'.format(max(abs(one_line_diff))))
    ax.set_ylabel(kw['ylabel'])
    ax.grid(linestyle='--', linewidth=0.5,alpha = 0.5)
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="4.5%", pad=.1)
    m = plt.cm.ScalarMappable(cmap=kw_diff['cMap'])
    m.set_array(diff_grid)
    m.set_clim(kw['cMin'],kw['cMax'])
    cb = plt.colorbar(m, boundaries=np.linspace(kw['cMin'],kw['cMax'], 64),cax=cbaxes)
    cb.ax.set_yticks(np.linspace(kw['cMin'],kw['cMax'],5))
    cb.ax.set_yticklabels(['{:.2f}'.format(x) for x in cb.ax.get_yticks()])
    cb.ax.set_ylabel(kw['label'])

    return fig

# Create folder for the difference contour
os.makedirs(join('difference_contour'), exist_ok=True)
# plot the difference contour reference to the first one
for i in range(len(all_mgrs)-1):
    fig = plot_difference_contour(all_mgrs[0], all_mgrs[i+1], output_folders[0], output_folders[i+1], **kw)
    fig.savefig(join('difference_contour',output_folders[i+1]+'_vs_'+output_folders[0]+'_contour.png'), dpi=300, bbox_inches='tight')
    plt.close()


# %% 
import matplotlib.tri as tri
# plot the resistivity profile
def plot_resistivity_contour(mgr, urf_file_name, **kw):
    mgr = load_1inversion_data(join(output_path,urf_file_name))
    paraDomain = mgr['paraDomain']
    model = np.log10(mgr['model'])
    xc = paraDomain.cellCenter()[:,0]
    yc = paraDomain.cellCenter()[:,1]
    triang = tri.Triangulation(xc, yc) # build grid based on centroids
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.tricontourf(triang, model, cmap=kw['cMap'], levels=50,
                vmin=np.log10(kw['cMin']),vmax=np.log10(kw['cMax']))
    ax.set_aspect('equal')
    ax.set_xlim(left, right)
    depth = 10
    ax.set_ylim(-depth, 0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.set_xlabel(kw['xlabel'])
    ax.set_ylabel(kw['ylabel'])
    ax.grid(linestyle='--', linewidth=0.5,alpha = 0.5)
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=.15)
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    m.set_array(model)
    m.set_clim(np.log10(kw['cMin']),np.log10(kw['cMax']))
    cb = plt.colorbar(m, boundaries=np.linspace(np.log10(kw['cMin']),np.log10(kw['cMax']), 50),cax=cbaxes)
    cb.ax.set_yticks(np.linspace(np.log10(kw['cMin']),np.log10(kw['cMax']),5))
    cb.ax.set_yticklabels(['{:.0f}'.format(10**x) for x in cb.ax.get_yticks()])
    cb.ax.set_ylabel(kw['label'])
    title_str = 'Inverted Resistivity Profile at {}'.format(
        datetime.strptime(urf_file_name[:8], "%y%m%d%H").strftime("%Y/%m/%d %H:00"))
    ax.set_title(title_str)
    ax.set_xlabel(kw['xlabel'])
    ax.set_ylabel(kw['ylabel'])

    Clip_cornor(ax, data, left, right, depth)
    return fig

# Create folder for the resistivity contour
os.makedirs(join('resistivity_contour'), exist_ok=True)
kw = dict(cMin=100, cMax=1000, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
# plot the resistivity profile
for i in range(len(all_mgrs)):
    fig = plot_resistivity_contour(all_mgrs[i], output_folders[i], **kw)
    fig.savefig(join('resistivity_contour',output_folders[i]+'_contour.png'), dpi=300, bbox_inches='tight')
    plt.close()
# %%
