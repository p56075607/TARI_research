# %%
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import join
plt.rcParams["font.family"] = "Microsoft Sans Serif"
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import itertools

# %%
electrode_x = [0,1,1.5,2,2.5,3,
               5,7,9,11,13,15,17,
               18,19,19.5,20,20.5,21,
               23,25,27,29,31,33,35,
               37,38,38.5,39,39.5,40]
electrode_y = np.zeros(32)
fig, ax = plt.subplots()
ax.plot(electrode_x, electrode_y,'kv',label='Electrode')
def comprehensive_array(n):
    def abmn(n):
        """
        Construct all possible four-point configurations for a given
        number of sensors after Noel and Xu (1991).
        """
        combs = list(itertools.combinations(range(n), 4))
        
        # Calculate the number of unique permutations
        num_perms = len(combs) * 2
        
        # Initialize an array to store all permutations
        perms = np.empty((num_perms, 4), 'int')
        
        print(f"Comprehensive data set: {len(perms)} configurations.")
        
        index = 0
        for comb in combs:
                            # A           B         M      N 
            perms[index, :] = [comb[0], comb[1], comb[2], comb[3]]  # ABMN
            index += 1
            # perms[index, :] = [comb[0], comb[2], comb[1], comb[3]]  # AMBN
            # index += 1
            perms[index, :] = [comb[0], comb[3], comb[1], comb[2]]  # AMNB
            index += 1
        
        return perms

    # Add configurations
    cfgs = abmn(n) # create all possible 4P cgfs for 16 electrodes

    # Add electrodes
    scheme = pg.DataContainerERT() 

    for i in range(n):
        scheme.createSensor([electrode_x[i], electrode_y[i]]) # 2D, no topography

    for i, cfg in enumerate(cfgs):
        scheme.createFourPointData(i, *map(int, cfg)) # (We have to look into this: Mapping of int necessary since he doesn't like np.int64?)
    skip_step = 5
    remove_index = np.array([x for x in np.arange(len(scheme['a']))])
    remove_index = remove_index % skip_step != 0
    scheme.remove(remove_index)

    return scheme

scheme = comprehensive_array(len(electrode_x))
# %%
left = 0
right = 40
depth = 10
rho1 = 200
rho2 = 2000
kw = dict(cMin=rho1, cMax=rho2, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
DATA = []
inf_depths = np.linspace(-1,0,6)[::-1]#-0.1
for n,inf_depth in enumerate(inf_depths):
    world = mt.createWorld(start=[left, 0], end=[right, -depth],
                        layers=[inf_depth], 
                        worldMarker=True)
    mesh = mt.createMesh(world, quality=34, area=0.01)
    # ax, cb = pg.show(mesh, markers=True,**kw)
    if (n == 1) :#| (n == 2):
        rhomap = [[0, rho1], [2, rho2]]
    elif inf_depth == 0:
        rhomap = [[2, rho2]]
    else:
        rhomap = [[1, rho1], [2, rho2]]

    ax, cb = pg.show(mesh, 
        data=rhomap, 
        showMesh=True,**kw)
    ax.set_ylim(-3, 0)
    ax.set_xlim(0, 1)
    data_i = ert.simulate(mesh=mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                            noiseAbs=1e-6, 
                            seed=1337) 
    def print_data_siminfo(data_sim):
        pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
        pg.info('Simulated data', data_sim)
        pg.info('The data contains:', data_sim.dataMap().keys())
        pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
        pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random seed for repeatable noise in synthetic experiments 
    print_data_siminfo(data_i)
    DATA.append(data_i)

# %%
mesh2 = mt.createMesh(mt.createParaMeshPLC(scheme,paraDepth=10,paraDX=1/10,paraMaxCellSize=0.25))
ax,_ = pg.show(mesh2,markers=False, showMesh=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
ax.set_xlim([0, 40])
ax.set_ylim([-10, 0])
# %%
tlmgr = ert.TimelapseERT(DATA)
tlmgr.invert(mesh=mesh2, lam=100,zWeight=1, maxIter=40,verbose=True)
# %%
kw_appres = dict(cMin=rho1, cMax=rho2, cMap='jet', logScale=True, orientation='vertical',
                    ylabel='Array type and \nElectrode separation (m)')
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label=r'$\Delta \rho$ (%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
fig, ax = plt.subplots(figsize=(3*10, 6*4), 
                       ncols=3, nrows=6,constrained_layout=False)
for n,inf_depth in enumerate(inf_depths):
    ert.show(DATA[n], ax=ax[n, 0],**kw_appres)
    ax[n, 0].set_title('Infiltration depth: %.1f m'%-inf_depth)
    pg.show(tlmgr.pd, tlmgr.models[n],ax=ax[n,1], **kw)
    ax[n, 1].set_title('Infiltration depth: %.1f m'%-inf_depth)
    ax[n,1].set_ylim(-depth, 0)
    ax[n,1].set_xlim([left,right])
    triangle_left = np.array([[left, -depth], [left+depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax[n, 1].add_patch(plt.Polygon(triangle_left,color='white'))
    ax[n, 1].add_patch(plt.Polygon(triangle_right,color='white'))   
    if n == 0:
        pg.show(tlmgr.pd, tlmgr.models[n],ax=ax[n,2], **kw)
        ax[n, 2].set_title('Infiltration depth: {:.1f} m, $\chi2$={:.2f}'.format(
            -inf_depth,tlmgr.chi2s[n]))
        ax[n,2].set_ylim(-depth, 0)
        ax[n,2].set_xlim([left,right])
        ax[n, 2].add_patch(plt.Polygon(triangle_left,color='white'))
        ax[n, 2].add_patch(plt.Polygon(triangle_right,color='white'))   
    else:
        contour = True
        if contour:
            
            # res_change = pg.interpolate(ERTDomain, 100*(tlmgr.models[n]-tlmgr.models[0])/tlmgr.models[0], ERTDomain.cellCenters())
            # plot_residual_contour(ax[n,2],ERTDomain,res_change,
            #                      'Infiltration depth: %.1f m'%-inf_depth,
            #                      xDevide,yDevide,**kw_compare)
            # ax[n, 2].set_title('Depth %.1f vs 0 m'%-inf_depth)
            # interface2 = mt.createLine(start=[left, inf_depth], end=[right, inf_depth])
            # ax[n, 2].plot(pg.x(interface2.nodes()),pg.y(interface2.nodes()),'--k')
            levels = 50
            # 使用 TwoSlopeNorm 設定色階
            # midnorm = matplotlib.colors.TwoSlopeNorm(vmin=kw_compare['cMin'], vcenter=0, vmax=kw_compare['cMax'])
            class StretchOutNormalize(plt.Normalize):
                def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
                    self.low = low
                    self.up = up
                    plt.Normalize.__init__(self, vmin, vmax, clip)

                def __call__(self, value, clip=None):
                    x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
                    return np.ma.masked_array(np.interp(value, x, y))

            clim = [kw_compare['cMin'], kw_compare['cMax']]
            midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-10, up=10)
            triang = matplotlib.tri.Triangulation(tlmgr.pd.cellCenter()[:,0], tlmgr.pd.cellCenter()[:,1])
            rho_diff = 100*(np.log10(tlmgr.models[n])-np.log10(tlmgr.models[0]))/np.log10(tlmgr.models[0])
            # rho_diff[(rho_diff >= low) & (rho_diff <= up)] = 1e-10
            cax = ax[n,2].tricontourf(triang, rho_diff,
                                cmap=kw_compare['cMap'], levels=levels,
                                norm=midnorm)
            # 添加 colorbar
            divider = make_axes_locatable(ax[n,2])
            cbaxes = divider.append_axes("right", size="4%", pad=0.15)
            mappable = plt.cm.ScalarMappable(norm=midnorm, cmap=kw_compare['cMap'])
            mappable.set_array(rho_diff)
            cb = plt.colorbar(mappable, cax=cbaxes, boundaries=np.linspace(kw_compare['cMin'], kw_compare['cMax'], levels))
            cb_ytick = np.linspace(kw_compare['cMin'], kw_compare['cMax'], 5)
            cb.ax.set_yticks(cb_ytick)
            cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
            cb.ax.set_ylabel(r'$\Delta \rho$ (%)')
            ax[n,2].set_ylim(-depth, 0)
            ax[n,2].set_xlim([left,right])
            ax[n,2].set_aspect('equal')
            ax[n, 2].add_patch(plt.Polygon(triangle_left,color='white'))
            ax[n, 2].add_patch(plt.Polygon(triangle_right,color='white'))   
            interface2 = mt.createLine(start=[left, inf_depth], end=[right, inf_depth])
            ax[n,2].plot(pg.x(interface2.nodes()),pg.y(interface2.nodes()),'--w')
        else:
            pg.show(tlmgr.pd, 
                100*(tlmgr.models[n]-tlmgr.models[0])/tlmgr.models[0],ax=ax[n,2],
                **kw_compare)
            
# fig.savefig(join('ERT_infiltration_noneuq_elespac.png'),dpi=300,bbox_inches='tight')