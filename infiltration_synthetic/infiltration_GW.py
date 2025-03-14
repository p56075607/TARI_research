# %%
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt
import numpy as np
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from plot_util import  plot_inverted_profile, plot_inverted_contour,Clip_cornor, plot_convergence, crossplot, data_misfit, export_inversion_info
from datetime import datetime, timedelta
from os.path import  isdir, join
from os import listdir
import re
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap

rho1 = 200
rho2 = 400
kw = dict(cMin=rho1, cMax=rho2, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
# %%
stg_fname = "test1202-12.stg"
data = ert.load(stg_fname)
print(data)
# %%
ert.show(data)

mid, sep = ert.visualization.midconfERT(data)
df = pd.DataFrame({'a':data['a'], 'b':data['b'], 'm':data['m'], 'n':data['n'], 'sep': sep})
# calculate the number of sep in each range: 0~9999, 10000~19999, 20000~29999, 30000~39999, 40000~49999, 50000~59999
df['sep_range'] = pd.cut(df['sep'], bins=np.arange(0, 70000, 10000), right=False)
df['sep_range'].value_counts()


df_40000_49999 = df[(df['sep'] > 30000) & (df['sep'] < 40000)]
# plot df_40000_49999.iloc[0] ['a'] and ['b'] and ['m'] and ['n'] in 1D
plt.plot(df_40000_49999.iloc[100]['a'],0, 'ro')
plt.plot(df_40000_49999.iloc[100]['b'],0, 'yo')
plt.plot(df_40000_49999.iloc[100]['m'],0, 'go')
plt.plot(df_40000_49999.iloc[100]['n'],0, 'bo')
plt.show()

# draw a pie plot of the number of sep in each range, 0~9999 and 50000~59999 are the same
df['sep_range'].value_counts().plot(kind='pie', autopct='%1.1f%%')

# %%
left = min(pg.x(data))
right = max(pg.x(data))
depth = 10


inf_depths = np.linspace(-1,0,11)[::-1]#-0.1
DATA = []
for n,inf_depth in enumerate(inf_depths):
    world = mt.createWorld(start=[left, 0], end=[right, -depth],
                        layers=[inf_depth], 
                        worldMarker=True)
    mesh = mt.createMesh(world, quality=34, area=0.01)
    # ax, cb = pg.show(mesh, markers=True,**kw)
    if (n == 1) | (n == 2):
        rhomap = [[0, rho1], [2, rho2]]
    elif inf_depth == 0:
        rhomap = [[2, rho2]]
    else:
        rhomap = [[1, rho1], [2, rho2]]

    ax, cb = pg.show(mesh, 
        data=rhomap, 
        showMesh=True,**kw)
    ax.set_ylim(-10, 0)
    ax.set_xlim(0, 1)

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
    data_i.save(join('syn_output', 'syn_{}.ohm'.format(n)))

# %%
ohm_path = r'C:\Users\Git\TARI_research\infiltration_synthetic\syn_output'
ohmfiles = ([_ for _ in listdir(ohm_path) if _.endswith('.ohm')])
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

# 使用 sorted() 函數進行排序
sorted_ohmfiles = sorted(ohmfiles, key=extract_number)
output_path = 'GW_syn_output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for i,ohm_file_name in enumerate(sorted_ohmfiles):
    pg.boxprint('Processing {:d} of {:d}: {:s}'.format(i+1, len(sorted_ohmfiles), ohm_file_name))

    time = (datetime(2025, 1, 1, 0, 0) + timedelta(hours=i)).strftime("%y%m%d%H")
    print(f'Inversion {i+1} of {len(sorted_ohmfiles)}, time: {time}')

    data = ert.load(join(ohm_path, ohm_file_name))
    if i == 0:
        # left = min(pg.x(data))
        # right = max(pg.x(data))
        # length = right - left
        # depth = length/4
        
        # plc = mt.createParaMeshPLC(data,quality=34,paraDX=1/4,paraMaxCellSize=0.2
        #                                     ,paraDepth=depth)
        # mesh = mt.createMesh(plc)
        # print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarker() == 2) if x]))

        # # ax,cb = pg.show(mesh,showMesh=True)
        # # ax.set_xlim([left,right])
        # # ax.set_ylim([-depth,0])
        left = min(pg.x(data))
        right = max(pg.x(data))
        length = right - left
        depth = 10

        yDevide = np.linspace(start=0, stop=-depth, num=101 )
        xDevide = np.linspace(start=left, stop=right, num=101)
        inversionDomain = pg.createGrid(x=xDevide,
                                        y=yDevide[::-1],
                                        # y=pg.cat([0],yDevide[1:]),
                                        # y=-pg.cat(
                                        #     pg.utils.grange(0, 0.25, n=2),
                                        #     pg.utils.grange(0.25, 15, n=20))[::-1],
                                        marker=2
                                        )
        mesh = pg.meshtools.appendTriangleBoundary(inversionDomain, marker=1,
                                                xbound=200, ybound=200,
                                                #    area=10
                                                )
        print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarker() == 2) if x]))

        # ax,cb = pg.show(mesh,showMesh=True)
        # ax.set_xlim([20,22])
        # ax.set_ylim([-0.5,0])
    if os.path.exists(join(output_path, f'{time}')):
        continue
    lam = 100
    mgr = ert.ERTManager(data)
    model = mgr.invert(data,mesh=mesh,
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
    mgr.data.save(join(save_path,'ERTManager',f'{time}_inv.ohm'))
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
left = min(pg.x(data))
right = max(pg.x(data))
length = right - left
depth = 10

yDevide = np.linspace(start=0, stop=-depth, num=101 )
xDevide = np.linspace(start=left, stop=right, num=101)
inversionDomain = pg.createGrid(x=xDevide,
                                y=yDevide[::-1],
                                # y=pg.cat([0],yDevide[1:]),
                                # y=-pg.cat(
                                #     pg.utils.grange(0, 0.25, n=2),
                                #     pg.utils.grange(0.25, 15, n=20))[::-1],
                                marker=2
                                )
mesh = pg.meshtools.appendTriangleBoundary(inversionDomain, marker=1,
                                        xbound=200, ybound=200,
                                        #    area=10
                                        )
print(mesh,'paraDomain cell#:',len([i for i, x in enumerate(mesh.cellMarker() == 2) if x]))

ax,cb = pg.show(mesh,showMesh=True)
ax.set_xlim([20,22])
ax.set_ylim([-0.5,0])
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

target_x = (x_coord == 21.25)


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
    DIFF_1D_target.append(DIFF_target)

DIFF_1D_target = np.array(DIFF_1D_target).T

# 繪製圖形
time_axis = np.linspace(0, 10, 11)


colors = [(0, 0, 1), (1, 1, 1), (1, 1, 1)]  # 從白色到藍色的顏色組合
nodes = [0, 1-(-3/-13), 1]  
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))     
kw = dict(cMin=-13, cMax=0,logScale=False,
            label='Relative resistivity \ndifference (%)',
            xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical',cMap=custom_cmap)       
Tmesh, Ymesh = np.meshgrid(time_axis, y_coord_sorted)
fig, ax = plt.subplots(figsize=(27.5, 12))
contour = ax.contourf(Tmesh, Ymesh, DIFF_1D_target, levels=15, cmap=kw['cMap'], vmin=kw['cMin'], vmax=kw['cMax'])
ax.set_ylim(-1.5,0)
ax.set_xlim(0,10)
fontsize = 30
ax.set_xlabel('Time (hrs)', fontsize=fontsize)
ax.set_ylabel('Depth (m)', fontsize=fontsize)

# plot the data point
ax.scatter(Tmesh, Ymesh, c=DIFF_1D_target, s=80,edgecolor='black', cmap=kw['cMap'], vmin=kw['cMin'], vmax=kw['cMax'])

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
# ax.set_yticks(np.arange(-10, 0, 1))
ax.set_xticks(np.arange(0, 10, 1))

ax.tick_params(labelsize=fontsize)
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')

ax.axhline(-0.5, color='black', linewidth=1, linestyle='--')

# add the ture line slope = -10/45
ax.plot(time_axis, -0.1*time_axis, color='red', linewidth=2)
# points each times and depth
for i in range(len(time_axis)):
    ax.scatter(time_axis[i], -0.1*time_axis[i], color='red', marker='o', s=100)