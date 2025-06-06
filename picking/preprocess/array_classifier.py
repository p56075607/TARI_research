"""
Array classifier for ERT data based on midconfERT logic
Identifies different array types following the same logic as PyGIMLi's midconfERT function
"""

import numpy as np
import itertools

def classify_array_type_midconf(a, b, m, n):
    """
    Classify array type based on midconfERT logic
    
    Parameters:
    a, b, m, n: electrode indices
    
    Returns:
    'alpha', 'beta', or 'gamma'
    """
    
    # 計算各電極對之間的絕對距離
    ab, am, an = np.abs(a-b), np.abs(a-m), np.abs(a-n)
    bm, bn, mn = np.abs(b-m), np.abs(b-n), np.abs(m-n)
    
    # 檢查所有四個電極是否有有效值（假設ERT資料中電極索引都是有效的數值）
    # 在實際ERT資料中，a,b,m,n通常都是有效的電極索引，不會是NaN
    iabmn = True  # 簡化假設所有電極索引都有效
    
    if not iabmn:
        return 'gamma'
    
    # 3. Alpha配置判斷 (Wenner-alpha: 30000, Schlumberger: 40000)
    # 計算MN電極的中點
    mnmid = (m + n) / 2
    
    # 判斷A和B是否在MN中點的兩側
    opposite_sides = np.sign((a-mnmid)*(b-mnmid)) < 0
    
    # 判斷M和N是否都在A和B之間
    ab_min = np.minimum(a, b)
    ab_max = np.maximum(a, b)
    mn_min = np.minimum(m, n)
    mn_max = np.maximum(m, n)
    
    # M和N都在A和B之間
    mn_between_ab = (mn_min > ab_min) & (mn_max < ab_max)
    
    # 特例：當AB包含在MN之間(MABN)時也視為alpha配置
    ab_between_mn = (ab_min > mn_min) & (ab_max < mn_max)
    
    # 判斷A、B與M、N是否完全分開（beta配置的典型特徵）
    ab_fully_separated = (ab_max < mn_min) | (mn_max < ab_min)
    
    # Alpha配置條件: A、B在MN中點兩側，且MN在AB之間或AB在MN之間，排除完全分開的情況
    ialfa = opposite_sides & (mn_between_ab | ab_between_mn) & (~ab_fully_separated)
    
    if ialfa:
        return 'alpha'
    
    # 4. Beta配置判斷 (dipole-dipole: 50000)
    # Beta配置條件：AB和MN完全分開
    if ab_fully_separated:
        return 'beta'
    
    # 5. Gamma配置：未被分類為alpha或beta的四電極測量
    return 'gamma'

def classify_array_type(a, b, m, n, electrodes_pos=None):
    """
    Wrapper function to maintain compatibility
    """
    return classify_array_type_midconf(a, b, m, n)

def is_wenner_array(a, b, m, n):
    """Check if configuration is Wenner array using midconfERT logic"""
    # 使用midconfERT邏輯判斷
    array_type = classify_array_type_midconf(a, b, m, n)
    if array_type != 'alpha':
        return False
        
    # 進一步判斷是否為Wenner
    mnmid = (m + n) / 2
    opposite_sides = np.sign((a-mnmid)*(b-mnmid)) < 0
    
    if not opposite_sides:
        return False
        
    # 計算AB和MN的中點對齊情況
    ab_mid = (a + b) / 2
    mn_mid = (m + n) / 2
    midpoints_aligned = np.abs(ab_mid - mn_mid) < 1e-6
    
    # 計算3×MN和AB的關係
    ab, mn = np.abs(a-b), np.abs(m-n)
    abmn3 = np.round((3*mn-ab)*10000)/10000
    
    return (abmn3 == 0) & midpoints_aligned

def is_schlumberger_array(a, b, m, n):
    """Check if configuration is Schlumberger array using midconfERT logic"""
    array_type = classify_array_type_midconf(a, b, m, n)
    if array_type != 'alpha':
        return False
        
    # 如果是alpha但不是Wenner，則為Schlumberger
    return not is_wenner_array(a, b, m, n)

def is_gradient_array(a, b, m, n):
    """Check if configuration is Gradient array - same as Schlumberger in midconfERT"""
    return is_schlumberger_array(a, b, m, n)

def is_dipole_dipole_array(a, b, m, n):
    """Check if configuration is Dipole-dipole array using midconfERT logic"""
    return classify_array_type_midconf(a, b, m, n) == 'beta'

def comprehensive_array_with_classification(n, electrode_x, electrode_y):
    """
    Create comprehensive array with array type classification using midconfERT logic
    
    Parameters:
    n: number of electrodes
    electrode_x, electrode_y: electrode positions
    
    Returns:
    scheme: PyGIMLi DataContainerERT with array type classification
    """
    import pygimli as pg
    
    def abmn(n):
        """Generate all possible four-point configurations"""
        combs = list(itertools.combinations(range(n), 4))
        num_perms = int(n*(n-3)*(n-2)*(n-1)/8)
        perms = np.empty((num_perms, 4), 'int')
        
        print(f"Comprehensive data set: {len(perms)} configurations.")
        
        index = 0
        for comb in combs:
            perms[index, :] = [comb[0], comb[1], comb[2], comb[3]]  # ABMN
            index += 1
            perms[index, :] = [comb[0], comb[3], comb[1], comb[2]]  # AMNB
            index += 1
        
        return perms

    # Generate configurations
    cfgs = abmn(n)
    
    # Create scheme
    scheme = pg.DataContainerERT()
    
    # Add electrodes
    for i in range(n):
        scheme.createSensor([electrode_x[i], electrode_y[i]])
    
    # Add configurations with classification
    array_types = []
    for i, cfg in enumerate(cfgs):
        scheme.createFourPointData(i, *map(int, cfg))
        array_type = classify_array_type_midconf(*cfg)
        array_types.append(array_type)
    
    # Add array type information to scheme
    scheme['array_type'] = np.array(array_types)
    
    return scheme

def get_array_indices(scheme):
    """
    Get indices for different array types
    
    Returns:
    dict with 'alpha', 'beta', 'gamma' keys containing indices
    """
    array_types = scheme['array_type']
    
    indices = {
        'alpha': np.where(array_types == 'alpha')[0],
        'beta': np.where(array_types == 'beta')[0],
        'gamma': np.where(array_types == 'gamma')[0]
    }
    
    return indices

def classify_with_midconf_sep(a, b, m, n):
    """
    Generate midconfERT-style separation codes for detailed classification
    
    Returns:
    array_type: 'alpha', 'beta', 'gamma'
    sep_code: separation code following midconfERT logic
    """
    # 計算各電極對之間的絕對距離
    ab, am, an = np.abs(a-b), np.abs(a-m), np.abs(a-n)
    bm, bn, mn = np.abs(b-m), np.abs(b-n), np.abs(m-n)
    
    # 定義編碼因子
    mI, mO, mT = 1, 100, 10000
    
    # Alpha配置判斷
    mnmid = (m + n) / 2
    opposite_sides = np.sign((a-mnmid)*(b-mnmid)) < 0
    
    ab_min = np.minimum(a, b)
    ab_max = np.maximum(a, b)
    mn_min = np.minimum(m, n)
    mn_max = np.maximum(m, n)
    
    mn_between_ab = (mn_min > ab_min) & (mn_max < ab_max)
    ab_between_mn = (ab_min > mn_min) & (ab_max < mn_max)
    ab_fully_separated = (ab_max < mn_min) | (mn_max < ab_min)
    
    ialfa = opposite_sides & (mn_between_ab | ab_between_mn) & (~ab_fully_separated)
    
    if ialfa:
        # Alpha配置
        spac = np.minimum(bn, bm)
        
        # 判斷Wenner vs Schlumberger
        ab_mid = (a + b) / 2
        mn_mid = (m + n) / 2
        midpoints_aligned = np.abs(ab_mid - mn_mid) < 1e-6
        abmn3 = np.round((3*mn-ab)*mT)/mT
        
        is_mabn = (ab_min >= mn_min) & (ab_max <= mn_max)
        abmn3_mabn = np.round((3*ab-mn)*mT)/mT
        
        is_wenner = ((abmn3 == 0) & midpoints_aligned) | \
                   (is_mabn & (abmn3_mabn == 0) & midpoints_aligned)
        
        if is_wenner:
            sep_code = 3*mT + spac * mI + (mn-1) * mO * is_mabn  # Wenner: 30000
        else:
            sep_code = 4*mT + spac * mI + (mn-1) * mO  # Schlumberger: 40000
            
        return 'alpha', int(sep_code)
    
    elif ab_fully_separated:
        # Beta配置 (dipole-dipole)
        sep_code = 5 * mT + (ab-1) * mO + np.minimum(
            np.minimum(am, an), np.minimum(bm, bn)) * mI
        return 'beta', int(sep_code)
    
    else:
        # Gamma配置
        sep_code = 6 * mT + np.maximum(np.maximum(ab, mn), 
                                     np.maximum(am, bn)) * mO + \
                  np.minimum(np.minimum(am, an), np.minimum(bm, bn)) * mI
        return 'gamma', int(sep_code) 