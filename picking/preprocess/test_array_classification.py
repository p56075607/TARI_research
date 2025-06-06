"""
Test script to validate array classification against midconfERT logic
"""

import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft YaHei'
import pygimli as pg
from pygimli.physics import ert

# Import our classifier
from array_classifier import classify_array_type_midconf, classify_with_midconf_sep

def load_test_data():
    """載入測試資料"""
    try:
        from convertURF import convertURF
        path = r"C:\Users\B30122\Downloads\22022019_m_E1\22022019_m_E1.urf"
        ohm_path = convertURF(path)
        data = pg.load(ohm_path)
        data['k'] = ert.createGeometricFactors(data, numerical=True)
        data['rhoa'] = data['r'] * data['k']
        return data
    except:
        print("無法載入測試資料，使用合成資料")
        return create_synthetic_data()

def create_synthetic_data():
    """創建合成測試資料"""
    # 創建一些典型的陣列配置進行測試
    configs = []
    
    # Wenner配置範例: A-M-N-B等距排列
    configs.append([0, 3, 1, 2])  # A=0, B=3, M=1, N=2 (Wenner)
    configs.append([1, 4, 2, 3])  # A=1, B=4, M=2, N=3 (Wenner)
    
    # Schlumberger配置範例: MN小間距，AB大間距
    configs.append([0, 5, 2, 3])  # A=0, B=5, M=2, N=3 (Schlumberger)
    configs.append([1, 6, 3, 4])  # A=1, B=6, M=3, N=4 (Schlumberger)
    
    # Dipole-dipole配置範例: AB和MN分離
    configs.append([0, 1, 3, 4])  # A=0, B=1, M=3, N=4 (DD)
    configs.append([1, 2, 4, 5])  # A=1, B=2, M=4, N=5 (DD)
    
    # Gamma配置範例: 交錯排列
    configs.append([0, 2, 1, 3])  # A=0, B=2, M=1, N=3 (AMBN - gamma)
    configs.append([0, 3, 2, 1])  # A=0, B=3, M=2, N=1 (ANMB - gamma)
    
    return configs

def test_classification():
    """測試分類結果"""
    print("測試陣列分類器...")
    
    # 使用合成資料測試
    configs = create_synthetic_data()
    
    results = []
    for i, (a, b, m, n) in enumerate(configs):
        array_type = classify_array_type_midconf(a, b, m, n)
        array_type_detailed, sep_code = classify_with_midconf_sep(a, b, m, n)
        
        results.append({
            'config_id': i,
            'a': a, 'b': b, 'm': m, 'n': n,
            'array_type': array_type,
            'array_type_detailed': array_type_detailed,
            'sep_code': sep_code,
            'description': describe_config(a, b, m, n)
        })
        
        print(f"配置 {i}: A={a}, B={b}, M={m}, N={n}")
        print(f"  類型: {array_type}")
        print(f"  詳細類型: {array_type_detailed}")
        print(f"  分離代碼: {sep_code}")
        print(f"  描述: {describe_config(a, b, m, n)}")
        print()
    
    return pd.DataFrame(results)

def describe_config(a, b, m, n):
    """描述配置類型"""
    # 按位置排序
    positions = sorted([(a, 'A'), (b, 'B'), (m, 'M'), (n, 'N')])
    sequence = ''.join([label for pos, label in positions])
    
    # 計算距離
    ab = abs(a - b)
    mn = abs(m - n)
    
    if sequence == 'AMNB':
        return f"Wenner型 (等距: {ab==3 and mn==1})"
    elif sequence == 'AMBN':
        if ab > mn * 2:
            return "Schlumberger型"
        else:
            return "Gamma交錯型"
    elif sequence == 'ABMN' or sequence == 'MNAB':
        return "Dipole-dipole型"
    else:
        return f"其他型 ({sequence})"

def compare_with_midconf():
    """與原始midconfERT進行比較"""
    try:
        # 嘗試載入真實資料進行比較
        data = load_test_data()
        
        if hasattr(data, 'sensorCount'):
            print(f"成功載入資料，共 {data.sensorCount()} 個電極，{len(data)} 個測量")
            
            # 提取前100個測量進行測試
            test_size = min(100, len(data))
            a_vals = data['a'][:test_size]
            b_vals = data['b'][:test_size]
            m_vals = data['m'][:test_size]
            n_vals = data['n'][:test_size]
            
            # 使用我們的分類器
            our_types = []
            our_codes = []
            
            for i in range(test_size):
                array_type = classify_array_type_midconf(a_vals[i], b_vals[i], 
                                                       m_vals[i], n_vals[i])
                _, sep_code = classify_with_midconf_sep(a_vals[i], b_vals[i], 
                                                      m_vals[i], n_vals[i])
                our_types.append(array_type)
                our_codes.append(sep_code)
            
            # 統計結果
            type_counts = pd.Series(our_types).value_counts()
            print("\n分類統計:")
            print(type_counts)
            
            # 按分離代碼範圍統計
            code_ranges = []
            for code in our_codes:
                if 30000 <= code < 40000:
                    code_ranges.append('Alpha-Wenner (30000-39999)')
                elif 40000 <= code < 50000:
                    code_ranges.append('Alpha-Schlumberger (40000-49999)')
                elif 50000 <= code < 60000:
                    code_ranges.append('Beta-DD (50000-59999)')
                elif 60000 <= code < 70000:
                    code_ranges.append('Gamma (60000-69999)')
                else:
                    code_ranges.append(f'Other ({code})')
            
            range_counts = pd.Series(code_ranges).value_counts()
            print("\n分離代碼統計:")
            print(range_counts)
            
            return True
        else:
            print("無法載入真實資料，僅使用合成資料測試")
            return False
            
    except Exception as e:
        print(f"比較過程中發生錯誤: {e}")
        return False

def plot_classification_results():
    """繪製分類結果"""
    results_df = test_classification()
    
    # 按類型分組統計
    type_counts = results_df['array_type'].value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    type_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('陣列類型分布 (合成資料)')
    
    plt.subplot(1, 2, 2)
    # 顯示每個配置的電極排列
    for idx, row in results_df.iterrows():
        positions = [row['a'], row['b'], row['m'], row['n']]
        labels = ['A', 'B', 'M', 'N']
        colors = ['red', 'yellow', 'green', 'blue']
        
        plt.scatter(positions, [idx]*4, c=colors, s=100, alpha=0.7)
        for pos, label in zip(positions, labels):
            plt.text(pos, idx, label, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('電極位置')
    plt.ylabel('配置編號')
    plt.title('電極配置示意圖')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def main():
    """主測試函數"""
    print("=" * 60)
    print("陣列分類器測試程式")
    print("=" * 60)
    
    # 1. 基本分類測試
    print("\n1. 基本分類測試:")
    results_df = test_classification()
    
    # 2. 與midconfERT比較
    print("\n2. 與真實資料比較:")
    compare_success = compare_with_midconf()
    
    # 3. 繪製結果
    print("\n3. 繪製分類結果:")
    plot_classification_results()
    
    # 4. 輸出詳細結果
    print("\n4. 詳細分類結果:")
    print(results_df.to_string(index=False))
    
    print("\n=" * 60)
    print("測試完成!")
    print("=" * 60)

if __name__ == "__main__":
    main() 