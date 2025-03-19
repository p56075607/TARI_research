# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import os
from os.path import join
import pygimli as pg
from pygimli.physics import ert  # the module
from math import pi
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

# %%
# load pg data
path = r"D:\R2MSDATA\TARI_E1_test\output_second_inversion\24030215_m_E1\ERTManager\inverison_data.ohm"
data = pg.load(path)
# %%
electrode_x = np.arange(1,21,1)
electrode_y = np.zeros(20)

def comprehensive_array(n):
    def abmn(n):
        """
        Construct all possible four-point configurations for a given
        number of sensors after Noel and Xu (1991).
        """
        combs = list(itertools.combinations(range(n), 4))
        
        # Calculate the number of unique permutations
        num_perms = int(n*(n-3)*(n-2)*(n-1)/8)
        
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
    # skip_step = 5
    # remove_index = np.array([x for x in np.arange(len(scheme['a']))])
    # remove_index = remove_index % skip_step != 0
    # scheme.remove(remove_index)

    return scheme

# data = comprehensive_array(len(electrode_x))
# %%
def midconfERT(data, ind=None, rnum=1, circular=False, switch=False):
    """Return the midpoint and configuration key for ERT data.

    Return the midpoint and configuration key for ERT data.

    Parameters
    ----------
    data : DataContainerERT
        data container with sensorPositions and a/b/m/n fields

    ind : []
        Documentme

    rnum : []
        Documentme

    circular : bool
        Return midpoint in degree (rad) instead if meter.

    Returns
    -------
    mid : np.array of float
        representative midpoint (middle of MN, AM depending on array)
    conf : np.array of float
        configuration/array key consisting of
        1) array type (Wenner-alpha/beta, Schlumberger, PP, PD, DD, MG)
            00000: pole-pole
            10000: pole-dipole or dipole-pole
            30000: Wenner-alpha
            40000: Schlumberger or Gradient
            50000: dipole-dipole or Wenner-beta
        2) potential dipole length (in electrode spacings)
            .XX..: dipole length
        3) separation factor (current dipole length or (di)pole separation)
            ...XX: pole/dipole separation (PP,PD,DD,GR) or separation
    """
    # 獲取第一個電極的x座標作為參考點
    x0 = data.sensorPosition(0).x()
    # 計算所有電極相對於第一個電極的x座標
    xe = pg.x(data.sensorPositions()) - x0
    # 獲取所有唯一的x座標值
    ux = pg.unique(xe)
    
    # 定義三個乘數因子，用於編碼不同的配置類型
    # mI: 個位數因子(1)，用於編碼極/偶極分離
    # mO: 百位數因子(100)，用於編碼偶極長度
    # mT: 萬位數因子(10000)，用於編碼陣列類型
    mI, mO, mT = 1, 100, 10000
    # 如果switch=True，則交換mI和mO的值
    if switch:
        mI, mO = mO, mI

    # 檢查是否為二維地形情況（非圓形）
    # 修改判斷條件，強制使用2D地形處理邏輯，除非明確指定circular=True
    if not circular:  # 2D with topography case
        # 計算相鄰電極之間的累積距離差
        dx = np.array(pg.utils.diff(pg.utils.cumDist(data.sensorPositions())))
        # 計算平均距離
        dxM = pg.mean(dx)
        
        # 檢查是否有地形變化（y或z座標不是常數）
        if min(pg.y(data)) != max(pg.y(data)) or \
           min(pg.z(data)) != max(pg.z(data)):
            # 地形情況
            # 檢查電極間距是否接近均勻
            if (max(abs(dx-dxM)) < dxM*0.9):
                # 如果最大間距變化小於平均間距的90%，假設等距電極排列
                dx = np.ones(len(dx)) * dxM
            else:
                # 地形可能有缺失電極，進行間距修正
                dx = np.floor(dx/np.round(dxM)) * dxM

        # 檢查間距是否很小（小於0.5單位）
        if max(dx) < 0.5:
            pg.debug("Detecting small distances, using mm accuracy")
            # 使用更高的四捨五入精度(3位小數)
            rnum = 3
            
        # 構建累積距離陣列，以第一個電極為0點
        xe = np.hstack((0., np.cumsum(np.round(dx, rnum)), np.nan))

        # 計算電極間的中位數距離並四捨五入
        de = np.median(np.diff(xe[:-1])).round(rnum)
        # 將絕對位置轉換為相對位置（以de為單位）
        ne = np.round(xe/de)
        
        # 檢查是否有負值座標（直接使用原始座標）
        if np.any(np.array(ux) < 0):
            # 對於有負值的情況，直接使用原始座標除以電極間距
            de = np.median(np.diff(sorted(ux)[:-1])).round(rnum)  # 排除可能的離群值
            ne = np.array(pg.x(data.sensorPositions())/de, dtype=int)  # 使用絕對座標
    else:  # 3D (without topo) case => take positions directly
        # 3D情況或圓形情況，直接使用電極位置
        de = np.median(np.diff(ux)).round(1)
        ne = np.array(xe/de, dtype=int)

    # 圓形電極布置的特殊處理
    if circular:
        # 計算圓心位置（所有電極位置的平均值）
        center = np.mean(data.sensorPositions(), axis=0)
        # '計算第一個電極到圓心的距離（即圓的半徑）
        r = data.sensors()[0].distance(center)
        # 計算第一個電極相對於圓心的向量
        s0 = data.sensors()[0]-center
        # 計算第二個電極相對於圓心的向量
        s1 = data.sensors()[1]-center
        # 計算第一個電極在極座標系中的角度
        p0 = np.arctan2(s0[1], s0[0])
        # 計算第二個電極在極座標系中的角度
        p1 = np.arctan2(s1[1], s1[0])
        
        # 根據電極的順序確定旋轉方向
        if p1 > p0:
            # 向左旋轉（逆時針）
            x = np.cos(np.linspace(0, 2*pi, data.sensorCount()+1)+p0)[:-1] * r
            y = np.sin(np.linspace(0, 2*pi, data.sensorCount()+1)+p0)[:-1] * r
        else:
            # 向右旋轉（順時針）
            x = np.cos(np.linspace(2*pi, 0, data.sensorCount()+1)+p0)[:-1] * r
            y = np.sin(np.linspace(2*pi, 0, data.sensorCount()+1)+p0)[:-1] * r

        # 將電極位置轉換為角度（用於圓形排列）
        a = np.array([np.arctan2(y[i], x[i]) for i in data['a']])
        b = np.array([np.arctan2(y[i], x[i]) for i in data['b']])
        m = np.array([np.arctan2(y[i], x[i]) for i in data['m']])
        n = np.array([np.arctan2(y[i], x[i]) for i in data['n']])

        # 標準化角度，使其在0到2π之間
        a = np.unwrap(a) % (np.pi*2)
        b = np.unwrap(b) % (np.pi*2)
        m = np.unwrap(m) % (np.pi*2)
        n = np.unwrap(n) % (np.pi*2)
    else:
        # 非圓形情況，使用相對電極位置
        a = np.array([ne[int(i)] for i in data['a']])
        b = np.array([ne[int(i)] for i in data['b']])
        m = np.array([ne[int(i)] for i in data['m']])
        n = np.array([ne[int(i)] for i in data['n']])

    # 如果提供了索引，則只處理指定的數據
    if ind is not None:
        a = a[ind]
        b = b[ind]
        m = m[ind]
        n = n[ind]

#     # 特殊情況處理：缺少A電極時使用B電極代替
#     anan = np.isnan(a)

    
#     print(anan)

# midconfERT(data)
# # %%
#     a[anan] = b[anan]
#     b[anan] = np.nan
    
    # 計算各電極對之間的絕對距離
    ab, am, an = np.abs(a-b), np.abs(a-m), np.abs(a-n)
    bm, bn, mn = np.abs(b-m), np.abs(b-n), np.abs(m-n)

    # 圓形排列中調整跨越180度的距離
    if circular:
        for v in [ab, mn, bm, an]:
            v[v > pi] = 2*pi - v[v > pi]

    # 檢查所有四個電極是否有有效值（提前定義iabmn）
    iabmn = np.isfinite(a) & np.isfinite(b) & np.isfinite(m) & np.isfinite(n)

    # 初始化：預設為極-極配置 (00000)
    sep = np.abs(a-m)  # 初始sep值僅為A和M之間的距離
    mid = (a+m) / 2    # 初始中點為A和M的中點

    # 1. 改進極-極配置判斷：只有在電流極與電位極各缺失一個時才標記為極-極
    # 檢查四極是否缺失了一個電流極和一個電位極
    isPP = (np.isnan(a) | np.isnan(b)) & (np.isnan(m) | np.isnan(n))
    # 但同時確保至少有一個電流極和一個電位極存在
    isPP = isPP & (np.isfinite(a) | np.isfinite(b)) & (np.isfinite(m) | np.isfinite(n))
    
    # 如果不符合極-極配置，先將sep設為一個不可能的大值，便於後續判斷
    sep[~isPP & iabmn] = 999999

    # 3-point (PD, DP) - 三極法配置處理 (10000)
    # 2. 擴展三極法配置判斷：處理更多可能的缺失電極情況
    
    # 2.1 極-偶極配置 (PD): 有效的n和m，但b缺失（原有邏輯）
    imn = np.isfinite(n) & np.isfinite(m) & (np.isnan(b) | np.isnan(a))
    # 確保至少有一個電流極存在
    imn = imn & (np.isfinite(a) | np.isfinite(b))
    mid[imn] = (m[imn]+n[imn]) / 2  # 中點為M和N的中點
    # 計算sep: 基礎距離 + 10000 + 偶極長度*100 + 修正因子*10000
    sep[imn] = np.minimum(am[imn], an[imn]) * mI + mT + mO * (mn[imn]-1) + \
        (np.sign(a[imn]-m[imn])/2+0.5) * mT
        
    # 2.2 偶極-極配置 (DP): 有效的a和b，但n缺失（原有邏輯）
    iab = np.isfinite(a) & np.isfinite(b) & (np.isnan(n) | np.isnan(m))
    # 確保至少有一個電位極存在
    iab = iab & (np.isfinite(m) | np.isfinite(n))
    mid[iab] = (a[iab]+b[iab]) / 2  # 中點為A和B的中點
    # 計算sep: 類似PD配置的邏輯
    sep[iab] = np.minimum(am[iab], bm[iab]) * mI + mT + mO * (ab[iab]-1) + \
        (np.sign(a[iab]-n[iab])/2+0.5) * mT

    # 4-point alpha配置: Wenner-alpha (30000) 或 Schlumberger (40000)
    # 檢查所有四個電極是否有有效值
    ialfa = np.copy(iabmn)
    
    # 3. 改進 alpha 配置判斷
    # 計算MN電極的中點
    mnmid = (m[iabmn] + n[iabmn]) / 2
    
    # 判斷A和B是否在MN中點的兩側
    opposite_sides = np.sign((a[iabmn]-mnmid)*(b[iabmn]-mnmid)) < 0
    
    # 判斷M和N是否都在A和B之間 (不考慮A、B的順序，只判斷是否包含)
    # 創建每組電極的最小值和最大值
    ab_min = np.minimum(a[iabmn], b[iabmn])
    ab_max = np.maximum(a[iabmn], b[iabmn])
    mn_min = np.minimum(m[iabmn], n[iabmn])
    mn_max = np.maximum(m[iabmn], n[iabmn])
    
    # M和N都在A和B之間
    mn_between_ab = (mn_min >= ab_min) & (mn_max <= ab_max)
    
    # 特例：當AB包含在MN之間(MABN)時也視為alpha配置
    ab_between_mn = (ab_min >= mn_min) & (ab_max <= mn_max)
    
    # 同時滿足條件: A、B在MN中點兩側，且MN在AB之間或AB在MN之間
    ialfa[iabmn] = opposite_sides & (mn_between_ab | ab_between_mn)

    # 設置alpha配置的中點和sep值
    mid[ialfa] = (m[ialfa] + n[ialfa]) / 2
    spac = np.minimum(bn[ialfa], bm[ialfa])
    # 用於區分Wenner和Schlumberger的參數
    abmn3 = np.round((3*mn[ialfa]-ab[ialfa])*mT)/mT
    # 計算sep: 基礎距離 + 偶極長度修正 + 30000 + Schlumberger修正
    sep[ialfa] = spac * mI + (mn[ialfa]-1) * mO * (abmn3 != 0) + \
        3*mT + (abmn3 < 0)*mT

    # 4-point beta配置: dipole-dipole (50000) 或 Wenner-beta
    ibeta = np.copy(iabmn)
    
    # 4. 改進 beta 配置判斷
    # 判斷A、B與M、N是否完全分開（不交叉）
    ab_fully_separated = (ab_max <= mn_min) | (mn_max <= ab_min)
    
    # beta配置條件：AB和MN完全分開，且不是alpha配置
    ibeta[iabmn] = ab_fully_separated & (~ialfa[iabmn])

    # 圓形排列的特殊處理（保持原始代碼）
    if circular:
        # 重新初始化beta標誌
        ibeta = np.copy(iabmn)

        def _averageAngle(vs):
            sumsin = 0
            sumcos = 0

            for v in vs:
                sumsin += np.sin(v)
                sumcos += np.cos(v)

            return np.arctan2(sumsin, sumcos)

        abC = _averageAngle([a[ibeta], b[ibeta]])
        mnC = _averageAngle([m[ibeta], n[ibeta]])

        mid[ibeta] = _averageAngle([abC, mnC])

        # special case when dipoles are completely opposite
        iOpp = abs(abs((mnC - abC)) - np.pi) < 1e-3
        mid[iOpp] = _averageAngle([b[iOpp], m[iOpp]])

        minAb = min(ab[ibeta])
        sep[ibeta] = 5 * mT + (np.round(ab[ibeta]/minAb)) * mO + \
            np.round(np.minimum(np.minimum(am[ibeta], an[ibeta]),
                                np.minimum(bm[ibeta], bn[ibeta])) / minAb) * mI
    else:
        # 非圓形情況下設置beta配置的中點和sep值
        mid[ibeta] = (a[ibeta] + b[ibeta] + m[ibeta] + n[ibeta]) / 4

        # 計算sep: 50000 + AB長度*100 + 最小電極距離
        sep[ibeta] = 5 * mT + (ab[ibeta]-1) * mO + np.minimum(
            np.minimum(am[ibeta], an[ibeta]),
            np.minimum(bm[ibeta], bn[ibeta])) * mI

    # 5. 新增gamma配置處理(60000)
    # 識別未被分類為alpha或beta，但有四個有效電極的測量
    igamma = iabmn & (~ialfa) & (~ibeta) & (sep >= 999999)
    
    # 處理gamma配置（AMBN等交錯排列）
    if np.any(igamma):
        # gamma配置的中點設為四個電極的平均
        mid[igamma] = (a[igamma] + b[igamma] + m[igamma] + n[igamma]) / 4
        
        # 計算sep: 60000(gamma類型) + 最大電極間距*100 + 最小電極間距
        sep[igamma] = 6 * mT + np.maximum(np.maximum(ab[igamma], mn[igamma]), 
                                         np.maximum(am[igamma], bn[igamma])) * mO + \
                    np.minimum(np.minimum(am[igamma], an[igamma]),
                               np.minimum(bm[igamma], bn[igamma])) * mI

    # 最終處理：將相對座標轉換回實際座標
    if not circular:
        mid *= de
        mid += x0

    return mid, sep

mid, sep = midconfERT(data)
# %%
df = pd.DataFrame({'a':data['a'], 'b':data['b'], 'm':data['m'], 'n':data['n'], 'sep': sep})
# calculate the number of sep in each range: 0~9999, 10000~19999, 20000~29999, 30000~39999, 40000~49999, 50000~59999
df['sep_range'] = pd.cut(df['sep'], bins=np.arange(0, 80000, 10000), right=False)
# Get the value counts from 'sep_range' column
counts = df['sep_range'].value_counts()

# Define a function to calculate the absolute count from percentage
def absolute_value(p):
    total = counts.sum()
    return int(round(p/100. * total))

# Plot pie chart with custom autopct to show counts
counts.plot(kind='pie', autopct=lambda p: '{:d}, {:1.1f}%'.format(absolute_value(p),p))
plt.ylabel('')  # Remove default ylabel for clarity
plt.show()

# %%
tpye_num = 30000
df_selected = df[(df['sep'] > 0+tpye_num) & (df['sep'] < 10000+tpye_num)]
# plot df_selected.iloc[0] ['a'] and ['b'] and ['m'] and ['n'] in 1D

fig,ax = plt.subplots(figsize=(8,48))
l = 1000
for ind in range(0+l,1000+l):
    ax.text(df_selected.iloc[ind]['a'], ind, 'a', color='white', ha='center', va='center')
    ax.text(df_selected.iloc[ind]['b'], ind, 'b', color='white', ha='center', va='center')
    ax.text(df_selected.iloc[ind]['m'], ind, 'm', color='white', ha='center', va='center')
    ax.text(df_selected.iloc[ind]['n'], ind, 'n', color='white', ha='center', va='center')
    ax.plot(df_selected.iloc[ind]['a'],ind, 'ro',markersize=10)
    ax.plot(df_selected.iloc[ind]['b'],ind, 'yo',markersize=10)
    ax.plot(df_selected.iloc[ind]['m'],ind, 'go',markersize=10)
    ax.plot(df_selected.iloc[ind]['n'],ind, 'bo',markersize=10)
    # ax.set_xlim(min(pg.x(data)),max(pg.x(data)))
    # ax.set_ylim(0,len(df_selected))

# %%
mesh_path = r"D:\R2MSDATA\TARI_E1_test\output_second_inversion\24030215_m_E1\ERTManager\mesh.bms"
mesh = pg.load(mesh_path)
lam = 100
mgr = ert.ERTManager(data)
model = mgr.invert(data,mesh=mesh,
                    lam=lam  ,zWeight=1,
                    maxIter = 6,
                    verbose=True)

rrms = mgr.inv.relrms()
chi2 = mgr.inv.chi2()
pg.boxprint('rrms={:.2f}%, chi^2={:.3f}'.format(rrms, chi2))

# %%
mgr.showResultAndFit()
# Plot the inverted profile
kw = dict(label='Resistivity $\Omega m$',
            logScale=True,cMap='jet',cMin=32,cMax=2512,
            xlabel="x (m)", ylabel="z (m)",
            orientation = 'vertical')
mgr.showResult(**kw)
# %%
print(data)
data['conf'] = sep
data.remove(data['conf']>=60000)
print(data)
df = pd.DataFrame({'a':data['a'], 'b':data['b'], 'm':data['m'], 'n':data['n'], 'sep': data['conf']})
# calculate the number of sep in each range: 0~9999, 10000~19999, 20000~29999, 30000~39999, 40000~49999, 50000~59999, 60000~69999
df['sep_range'] = pd.cut(df['sep'], bins=np.arange(0, 80000, 10000), right=False)
df['sep_range'].value_counts()
df['sep_range'].value_counts().plot(kind='pie', autopct='%1.1f%%')

# %%
mgr2 = ert.ERTManager(data)
model2 = mgr2.invert(data,mesh=mesh,
                    lam=lam  ,zWeight=1,
                    maxIter = 6,
                    verbose=True)
# %%
mgr2.showResultAndFit()
mgr2.showResult(**kw)
