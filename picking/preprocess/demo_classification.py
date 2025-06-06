"""
簡單演示新的陣列分類器功能
"""

from array_classifier import classify_array_type_midconf, classify_with_midconf_sep
import numpy as np
import pandas as pd

def demo_basic_classification():
    """演示基本分類功能"""
    print("=" * 60)
    print("ERT 陣列分類器演示 (基於 midconfERT 邏輯)")
    print("=" * 60)
    
    # 定義一些典型的測試配置
    test_configs = [
        # [a, b, m, n, 預期類型, 描述]
        [0, 3, 1, 2, 'alpha', 'Wenner 配置 (A-M-N-B)'],
        [1, 4, 2, 3, 'alpha', 'Wenner 配置 (A-M-N-B)'],
        [0, 5, 2, 3, 'alpha', 'Schlumberger 配置 (A-M-N-B, AB>MN)'],
        [0, 1, 3, 4, 'beta', 'Dipole-dipole 配置 (A-B | M-N)'],
        [1, 2, 5, 6, 'beta', 'Dipole-dipole 配置 (A-B | M-N)'],
        [0, 2, 1, 3, 'gamma', 'Gamma 配置 (A-M-B-N)'],
        [0, 3, 2, 1, 'gamma', 'Gamma 配置 (A-N-M-B)'],
        [2, 1, 0, 3, 'gamma', 'Gamma 配置 (M-B-A-N)'],
    ]
    
    results = []
    
    print("\n分類結果:")
    print("-" * 80)
    print(f"{'配置':<15} {'分類':<8} {'代碼':<8} {'預期':<8} {'✓':<3} {'描述'}")
    print("-" * 80)
    
    for i, (a, b, m, n, expected, description) in enumerate(test_configs):
        # 基本分類
        array_type = classify_array_type_midconf(a, b, m, n)
        
        # 詳細分類（含代碼）
        array_type_detailed, sep_code = classify_with_midconf_sep(a, b, m, n)
        
        # 檢查是否符合預期
        correct = "✓" if array_type == expected else "✗"
        
        # 格式化配置字串
        config_str = f"({a},{b},{m},{n})"
        electrode_order = get_electrode_order(a, b, m, n)
        
        print(f"{config_str:<15} {array_type:<8} {sep_code:<8} {expected:<8} {correct:<3} {description}")
        print(f"{'電極順序:':<15} {electrode_order}")
        print()
        
        results.append({
            'config': config_str,
            'a': a, 'b': b, 'm': m, 'n': n,
            'type': array_type,
            'sep_code': sep_code,
            'expected': expected,
            'correct': correct == "✓",
            'description': description,
            'electrode_order': electrode_order
        })
    
    # 統計結果
    df = pd.DataFrame(results)
    accuracy = df['correct'].mean() * 100
    
    print("-" * 80)
    print(f"總體準確率: {accuracy:.1f}% ({df['correct'].sum()}/{len(df)})")
    
    # 按類型統計
    type_stats = df.groupby('type').size()
    print(f"\n分類統計:")
    for array_type, count in type_stats.items():
        print(f"  {array_type}: {count} 個配置")
    
    return df

def get_electrode_order(a, b, m, n):
    """取得電極的位置順序"""
    electrodes = [(a, 'A'), (b, 'B'), (m, 'M'), (n, 'N')]
    electrodes.sort(key=lambda x: x[0])  # 按位置排序
    return '-'.join([label for pos, label in electrodes])

def demo_separation_codes():
    """演示分離代碼的含義"""
    print("\n" + "=" * 60)
    print("分離代碼範圍說明")
    print("=" * 60)
    
    code_ranges = {
        (30000, 39999): "Wenner-alpha 陣列",
        (40000, 49999): "Schlumberger/Gradient 陣列", 
        (50000, 59999): "Dipole-dipole 陣列",
        (60000, 69999): "Gamma 類型陣列"
    }
    
    for (start, end), description in code_ranges.items():
        print(f"{start:>6} - {end:<6}: {description}")
    
    print(f"\n代碼組成 (以 midconfERT 邏輯):")
    print(f"  TTOOSS: TT=陣列類型(萬位), OO=偶極長度(百位), SS=分離距離(個位)")

def demo_real_examples():
    """演示真實的配置範例"""
    print("\n" + "=" * 60) 
    print("真實配置範例")
    print("=" * 60)
    
    # 模擬一些真實的ERT配置
    real_configs = []
    
    # 生成Wenner配置 (a=3的倍數)
    for a in range(3):
        real_configs.append([a, a+3, a+1, a+2])
    
    # 生成Schlumberger配置
    for a in [0, 1]:
        real_configs.append([a, a+5, a+2, a+3])
    
    # 生成Dipole-dipole配置
    for separation in [1, 2]:
        real_configs.append([0, 1, 2+separation, 3+separation])
    
    print("配置範例分類結果:")
    print("-" * 50)
    
    for i, (a, b, m, n) in enumerate(real_configs):
        array_type, sep_code = classify_with_midconf_sep(a, b, m, n)
        electrode_order = get_electrode_order(a, b, m, n)
        
        print(f"配置 {i+1}: ({a},{b},{m},{n}) -> {array_type} ({sep_code}) [{electrode_order}]")

def main():
    """主演示函數"""
    # 基本分類演示
    df = demo_basic_classification()
    
    # 分離代碼說明
    demo_separation_codes()
    
    # 真實範例
    demo_real_examples()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("如需詳細測試，請執行: python test_array_classification.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 