# %%
# Import moduals
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from inverison_util import convertURF
from ridx_analyse import ridx_analyse

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Yahei"
# %config InlineBackend.figure_format='svg' # Setting figure format for this notebook
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import pygimli.meshtools as mt
from datetime import datetime
from os.path import join
from os import listdir
from datetime import timedelta
import matplotlib.dates as mdates
import pickle
import pandas as pd
from datetime import datetime
from collections import Counter

urf_path = r'D:\R2MSDATA_2024\TARI_E1_test\urf'
urffiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.urf')])
ohmfiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.ohm')])

def get_datetime_list_and_count(directory):
    # Initialize an empty list to store datetime objects
    datetime_list = []
    urffiles = sorted([_ for _ in listdir(directory) if _.endswith('.urf')])
    # Loop through all files in the directory
    for filename in urffiles:
        # Check if the filename matches the expected format
        if len(filename) > 8 and filename[:8].isdigit():
            # Extract the date-time part from the filename
            date_time_str = filename[:8]
            # Convert the string to a datetime object
            date_time_obj = datetime.strptime(date_time_str, '%y%m%d%H')
            # Append the datetime object to the list
            datetime_list.append(date_time_obj)

    # Count the occurrence of each date
    date_count = Counter([dt.date() for dt in datetime_list])

    return datetime_list, date_count

dates, date_count = get_datetime_list_and_count(urf_path)
print(len(date_count))
# %%
ridx_urf_path = r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor\E1_check\urf_E1_ridx'
unsorted_quality_info = ridx_analyse(ridx_urf_path, formula_choose='C')

ridx = unsorted_quality_info/100
rest = 50000
t3 = np.argsort(ridx)[rest:]
remove_index = np.full((len(unsorted_quality_info)), False)
for i in range(len(t3)):
    remove_index[t3[i]] = True

# %%
def read_ohm_r_column(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the number of electrodes and their positions
    num_electrodes = int(lines[0].strip().split()[0])
    electrodes = []
    idx = 2  # Start reading electrode positions after the header line

    # Skip electrode positions
    while len(electrodes) < num_electrodes and idx < len(lines):
        line = lines[idx].strip()
        if line and not line.startswith('#'):
            electrodes.append([float(x) for x in line.split()])
        idx += 1

    # Extract the number of data points
    while idx < len(lines) and not lines[idx].strip().split()[0].isdigit():
        idx += 1

    if idx >= len(lines):
        raise ValueError("Data points line not found")

    num_data = int(lines[idx].strip().split()[0])
    idx += 2  # Skip the data count line and the header line

    r_values = []
    while len(r_values) < num_data and idx < len(lines):
        line = lines[idx].strip()
        if line and not line.startswith('#'):
            data = line.split()
            r_values.append(float(data[4]))  # Assuming 'r' is the 5th column
        idx += 1

    if len(r_values) != num_data:
        raise ValueError("Mismatch between number of data points and extracted data")

    return r_values

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
    mn_between_ab = (mn_min > ab_min) & (mn_max < ab_max)
    
    # 特例：當AB包含在MN之間(MABN)時也視為alpha配置
    ab_between_mn = (ab_min > mn_min) & (ab_max < mn_max)
    
    # 判斷A、B與M、N是否完全分開（beta配置的典型特徵）
    ab_fully_separated = (ab_max < mn_min) | (mn_max < ab_min)
    
    # 同時滿足條件: A、B在MN中點兩側，且MN在AB之間或AB在MN之間
    # 排除完全分開的情況（這應該屬於beta配置）
    ialfa[iabmn] = opposite_sides & (mn_between_ab | ab_between_mn) & (~ab_fully_separated)

    # 設置alpha配置的中點和sep值
    mid[ialfa] = (m[ialfa] + n[ialfa]) / 2
    spac = np.minimum(bn[ialfa], bm[ialfa])
    
    # 用於區分Wenner和Schlumberger的參數
    # 計算AB和MN之間的關係
    abmn3 = np.round((3*mn[ialfa]-ab[ialfa])*mT)/mT
    
    # 計算AB和MN的中點
    ab_mid = (a[ialfa] + b[ialfa]) / 2
    mn_mid = (m[ialfa] + n[ialfa]) / 2
    
    # 判斷MN中點是否與AB中點對齊（考慮數值精度問題，使用近似相等）
    midpoints_aligned = np.abs(ab_mid - mn_mid) < 1e-6
    
    # 特例：計算MABN配置中的AB和MN關係
    # 判斷是否為MABN型態(AB在MN內)
    is_mabn = (ab_min[ialfa] >= mn_min[ialfa]) & (ab_max[ialfa] <= mn_max[ialfa])
    # 對MABN配置計算3×AB和MN的關係
    abmn3_mabn = np.round((3*ab[ialfa]-mn[ialfa])*mT)/mT
    
    # 判斷Wenner-alpha條件：
    # 常規情況: 3×MN = AB (abmn3 = 0) 且 MN中點與AB中點對齊
    # 特殊情況: MABN且3×AB = MN (abmn3_mabn = 0) 且 AB中點與MN中點對齊
    is_wenner = ((abmn3 == 0) & midpoints_aligned) | \
               (is_mabn & (abmn3_mabn == 0) & midpoints_aligned)
    
    # 計算sep值:
    # 如果是Wenner-alpha，使用30000
    # 如果是Schlumberger，使用40000
    # 其中加上基礎距離和偶極長度修正
    sep[ialfa] = spac * mI + (mn[ialfa]-1) * mO * ((~is_wenner) | is_mabn) + \
        3*mT + (~is_wenner)*mT

    # 4-point beta配置: dipole-dipole (50000) 或 Wenner-beta
    ibeta = np.copy(iabmn)
    
    # beta配置條件：AB和MN完全分開，且不是alpha配置
    # 這裡直接使用之前計算的ab_fully_separated
    ibeta[iabmn] = ab_fully_separated

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

median_RHOA = []
alpha_one_data = []  # 改成收集所有 sep=30001 的資料
alpha_one_dates = []  # 對應的日期時間
alpha_RHOA = []
beta_RHOA = []
gamma_RHOA = []

for i,urf_file_name in enumerate(urffiles):
    if urf_file_name[:-4]+'.ohm' in ohmfiles: # 檢查是否有 ohm 檔案，若有就視為做過不跑反演
        print(urf_file_name[:-4]+'.urf is already processed. Skip it!')
        ohm_file_name = join(urf_path,urf_file_name[:-4]+'.ohm')
    else:
        print('Processing: '+urf_file_name)
        ohm_file_name = convertURF(join(urf_path,urf_file_name),has_trn = False)

    if i == 0:
        data = ert.load(ohm_file_name)
        data.remove(remove_index)
        print(data) 
        data['k'] = ert.createGeometricFactors(data, numerical=True) # 以數值方法計算幾何因子，較耗費電腦資源
        rhoa = data['k'] * data['r']
        
        # 計算 sep 值用於分類
        mid, sep = midconfERT(data)
        
        # 根據 sep 值分類不同的電極陣列類型
        alpha_mask = ((sep >= 30000) & (sep < 40000)) | ((sep >= 40000) & (sep < 50000))  # Wenner + Schlumberger & Gradient
        alpha_one_mask = (sep == 30001)  # 新增這行：專門篩選 sep=30001 的資料
        beta_mask = (sep >= 50000) & (sep < 60000)  # Dipole-dipole
        gamma_mask = (sep >= 60000) & (sep < 70000)  # gamma type
        
        # ===== 在這裡加入座標分析程式碼 =====
        print("=== sep=30001 的 ABMN 座標 ===")
        print(f"總共找到 {np.sum(alpha_one_mask)} 組 sep=30001 的資料")

        # 建立包含 ABMN 座標的 DataFrame
        df_alpha_one_coords = pd.DataFrame({
            'a': data['a'][alpha_one_mask],
            'b': data['b'][alpha_one_mask], 
            'm': data['m'][alpha_one_mask],
            'n': data['n'][alpha_one_mask],
            'sep': sep[alpha_one_mask]
        })

        # 顯示所有 14 組座標
        print("\n完整的 ABMN 座標表:")
        print(df_alpha_one_coords.to_string(index=True))

        # 分析電極排列模式
        print("\n=== 電極排列模式分析 ===")
        for j, (idx, row) in enumerate(df_alpha_one_coords.iterrows()):
            a, b, m, n = int(row['a']), int(row['b']), int(row['m']), int(row['n'])
            print(f"\n第 {j+1} 組 (索引 {idx}): A={a}, B={b}, M={m}, N={n}")
            
            # 排序電極位置以查看排列模式
            electrodes = [('A', a), ('B', b), ('M', m), ('N', n)]
            electrodes.sort(key=lambda x: x[1])  # 按位置排序
            pattern = ' - '.join([f"{name}({pos})" for name, pos in electrodes])
            print(f"  排列模式: {pattern}")
            
            # 計算電極間距
            mn_dist = abs(m - n)
            ab_dist = abs(a - b)
            am_dist = abs(a - m)
            an_dist = abs(a - n)
            bm_dist = abs(b - m)
            bn_dist = abs(b - n)
            
            print(f"  MN間距: {mn_dist}, AB間距: {ab_dist}")
            print(f"  最小分離距離: {min(am_dist, an_dist, bm_dist, bn_dist)}")
            
            # 檢查是否符合 Wenner-alpha 配置
            # Wenner-alpha: A-M-N-B 等間距排列
            all_positions = sorted([a, b, m, n])
            spacings = [all_positions[k+1] - all_positions[k] for k in range(3)]
            is_equal_spacing = all(abs(s - spacings[0]) < 0.1 for s in spacings)
            
            if is_equal_spacing:
                print(f"  ✓ 符合 Wenner-alpha 等間距配置 (間距: {spacings[0]})")
            else:
                print(f"  ✗ 不符合等間距配置 (間距: {spacings})")

        # 統計分析
        print("\n=== 統計分析 ===")
        print(f"A 電極位置範圍: {df_alpha_one_coords['a'].min()} - {df_alpha_one_coords['a'].max()}")
        print(f"B 電極位置範圍: {df_alpha_one_coords['b'].min()} - {df_alpha_one_coords['b'].max()}")
        print(f"M 電極位置範圍: {df_alpha_one_coords['m'].min()} - {df_alpha_one_coords['m'].max()}")
        print(f"N 電極位置範圍: {df_alpha_one_coords['n'].min()} - {df_alpha_one_coords['n'].max()}")

        # 計算每組配置的基本間距
        print("\n=== 基本間距分析 ===")
        for j, (idx, row) in enumerate(df_alpha_one_coords.iterrows()):
            a, b, m, n = int(row['a']), int(row['b']), int(row['m']), int(row['n'])
            positions = sorted([a, b, m, n])
            basic_spacing = min([positions[k+1] - positions[k] for k in range(3)])
            print(f"第 {j+1} 組基本間距: {basic_spacing}")

        # 儲存座標資料
        df_alpha_one_coords.to_csv('alpha_one_coordinates.csv', index=True)
        print(f"\n座標資料已儲存至 alpha_one_coordinates.csv")
        
    else:
        r_values = read_ohm_r_column(ohm_file_name)
        removed_r = np.delete(r_values, t3)
        rhoa = data['k'] * removed_r
    
    # 計算各類型的中位數
    median_RHOA.append(np.median(rhoa))
    
    # 收集所有 sep=30001 的資料，每個時段一個 row
    if np.any(alpha_one_mask):
        alpha_one_rhoa_values = rhoa[alpha_one_mask]
        alpha_one_data.append(list(alpha_one_rhoa_values))  # 儲存為列表
    else:
        alpha_one_data.append([])  # 沒有資料時存空列表

# 建立 median_RHOA 的 DataFrame
df_median = pd.DataFrame({
    'datetime': dates,
    'median_RHOA': median_RHOA
})
df_median.set_index('datetime', inplace=True)

# 建立 alpha_one_RHOA 的 DataFrame（每個時段一個 row，多個 column）
# 找出最大的 column 數量
max_cols = max(len(data) for data in alpha_one_data) if alpha_one_data else 0

# 建立 column 名稱
column_names = [f'alpha_one_RHOA_{i+1}' for i in range(max_cols)]

# 準備資料，補齊短缺的 column 為 NaN
alpha_one_matrix = []
for data_i in alpha_one_data:
    row = data_i + [np.nan] * (max_cols - len(data_i))  # 補齊 NaN
    alpha_one_matrix.append(row)

# 建立 DataFrame
df_alpha_one = pd.DataFrame(alpha_one_matrix, columns=column_names)
df_alpha_one['datetime'] = dates
df_alpha_one.set_index('datetime', inplace=True)

# 格式化日期為 2024/02/29 17:00 格式
df_median.index = df_median.index.strftime('%Y/%m/%d %H:%M')
df_alpha_one.index = df_alpha_one.index.strftime('%Y/%m/%d %H:%M')

# 儲存為 CSV 檔案
df_median.to_csv('median_rhoa_data.csv')
df_alpha_one.to_csv('alpha_one_rhoa_data.csv')

print("median_RHOA 資料已儲存至 median_rhoa_data.csv")
print("alpha_one_RHOA 資料已儲存至 alpha_one_rhoa_data.csv")
print(f"\nmedian_RHOA 資料維度: {df_median.shape}")
print(f"alpha_one_RHOA 資料維度: {df_alpha_one.shape}")
print(f"最大 sep=30001 資料數量: {max_cols}")



