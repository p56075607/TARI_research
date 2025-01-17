# %%
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import scipy.linalg as lin
from scipy.sparse import coo_matrix, spdiags

kw = dict(logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')

def print_data_siminfo(data_sim):
    pg.info(np.linalg.norm(data_sim['err']), np.linalg.norm(data_sim['rhoa']))
    pg.info('Simulated data', data_sim)
    pg.info('The data contains:', data_sim.dataMap().keys())
    pg.info('Simulated rhoa (min/max)', min(data_sim['rhoa']), max(data_sim['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data_sim['err'])*100, max(data_sim['err'])*100)#seed : numpy.random see

# Create a mesh for the forward modeling
rho1 = 500
left = 0
right = 20
depth = 4

scheme = ert.createData(elecs=np.linspace(start=left, stop=right, num=41),
                           schemeName='dd')
world = mt.createWorld(start=[left, 0], end=[right, -depth],
                    worldMarker=True)
for p in scheme.sensors():
    world.createNode(p)

mesh = mt.createMesh(world, area=0.01)

rhomap = [[1, rho1]]
ax, cb = pg.show(mesh, 
    data=rhomap, 
    showMesh=True,**kw)
ax.plot(scheme.sensors(), np.zeros(len(scheme.sensors())), 'ko',markersize=3)
ax.set_ylim(-4, 0)


data = ert.simulate(mesh=mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                            noiseAbs=1e-6, 
                            seed=1337) 
print_data_siminfo(data)
# %%
mgr = ert.ERTManager(data)
mesh_inv = mt.createMesh(world, area=1)
mgr.invert(mesh=mesh_inv, lam=30, verbose=True)
# %%
from pygimli.frameworks.resolution import resolutionMatrix
RM = resolutionMatrix(mgr.inv)
pg.show(mesh_inv,RM.diagonal())
pg.show(mesh_inv, mgr.coverage())
# %%
mgr = ert.ERTManager(scheme)
model = pg.Vector(mesh.cellCount(), rho1)
mgr.fop.setMesh(mesh)
mgr.fop.createJacobian(model)

# %%
J = mgr.fop.jacobian()
# ax, cb = pg.show(J,orientation = 'vertical')
# ax.set_ylabel('Number of data',fontsize=12)
# ax.set_title('Number of model')
# cb.ax.set_xlabel('Elements value',fontsize=12)
print(J.shape)
J.save('J.bmat')
# %%

C = mgr.fop.createConstraints()
C.save('C.bmat')

# lam = 2.5*10**-6
# # J = np.array(J)

# JTJ = pg.matrix.Mult2Matrix(pg.matrix.TransposedMatrix(J),J)#J.T @ J
# I_JTJ = pg.matrix.IdentityMatrix((JTJ.cols()))
# I_JTJ*JTJ
# # JTJ = np.array(JTJ)
# CTC =pg.matrix.Mult2Matrix(pg.matrix.TransposedMatrix(C),C)  # C.T * C = (C_m)^-1
# # CTC = np.array(CTC)
# # RM = lin.inv(JTJ + CTC*lam).dot(JTJ)
# # print(RM.shape)

# %%
def loadbasesens(sensname):
    """
       Load base sensitivity matrix from BERT binary file.
    """
    fid = open(sensname, 'rb')
    basendata = int(np.fromfile(fid, 'int32', 1)[0])
    basenmodel = int(np.fromfile(fid, 'int32', 1)[0])
    SB = np.empty((basendata, basenmodel), 'float')

    for i in range(basendata):
        SB[i, :] = np.fromfile(fid, 'float', basenmodel)

    print(sensname + " loaded.")
    print("base Number of ABMNs: %s" % basendata)
    print("base Number of cells: %s" % basenmodel)

    return SB

J_loaded = loadbasesens('J.bmat')
# %%

def load_constraint_matrix(fname, myshape=None):
    """ Load constraint matrix from BERT in sparse format """
    i, j, data = np.loadtxt(fname, unpack=True, dtype=int)
    if myshape is None:
        myshape = (max(i)+1, max(j)+1)
    C = coo_matrix((data, (i, j)), shape=myshape, dtype=int)
    return C

C_loaded = load_constraint_matrix('C.bmat')
# %%
def compute_res(J, C, lam=1):
    """ Compute formal model resolution matrix and return its diagonal.

        Parameters:
        ===========
          Jb: Jacobian or sensitivity matrix
          Cb: Constraint (derivative) matrix
          alpha: tune amount of regularization
    """
    JTJ = J.T.dot(J)
    CTC = C.T.dot(C)  # C.T * C = (C_m)^-1

    RM = lin.inv(JTJ + CTC*lam).dot(JTJ)

    return RM

RM = compute_res(J_loaded, C_loaded, lam=1)