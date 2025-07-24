import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def exp_decay(t, delta_theta, k, theta_0):
    """
    修正後的指數衰減函數: θ(t) = Δθ * exp(-k*t) + θ₀
    
    Parameters:
    t: 時間 (天)
    delta_theta: 初始含水量變化幅度
    k: 衰減常數
    theta_0: 基準含水量
    """
    return delta_theta * np.exp(-k * t) + theta_0

def fit_theta_decay(time_data, theta_values):
    """含水量直接擬合計算τ（使用修正函數）"""
    try:
        # 初始參數估計
        delta_theta_guess = theta_values.iloc[0] - theta_values.iloc[-1]
        k_guess = 0.1
        theta_0_guess = theta_values.iloc[-1]
        
        # 擬合修正後的指數衰減函數
        popt, pcov = curve_fit(
            exp_decay, 
            time_data, 
            theta_values,
            p0=[delta_theta_guess, k_guess, theta_0_guess],
            bounds=([-np.inf, 0.001, 0], [np.inf, 2.0, 50])
        )
        
        delta_theta_fit, k_fit, theta_0_fit = popt
        tau = 1 / k_fit
        
        # 檢查擬合品質
        r_squared = calculate_r_squared(theta_values, exp_decay(time_data, *popt))
        
        return tau, r_squared, popt
            
    except Exception as e:
        print(f"含水量擬合失敗: {e}")
        return -1, -1, None

def calculate_r_squared(y_observed, y_predicted):
    """計算決定係數R²"""
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)

def plot_combined_analysis(cumulative_data, tau_theta, theta_fit_params):
    """繪製組合後的乾燥期間分析圖"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 生成平滑的時間軸
    t_smooth = np.linspace(cumulative_data['delay_days'].min(), cumulative_data['delay_days'].max(), 200)
    
    # 按窗口繪製不同顏色的散點
    colors = plt.cm.tab10(np.linspace(0, 1, cumulative_data['window_id'].nunique()))
    
    for i, window_id in enumerate(cumulative_data['window_id'].unique()):
        window_data = cumulative_data[cumulative_data['window_id'] == window_id]
        ax.scatter(window_data['delay_days'], window_data['mean_1m'], 
                  color=colors[i], s=50, alpha=0.7, label=f'窗口 {window_id+1}')
    
    # 含水量擬合曲線
    if theta_fit_params is not None:
        theta_fit_smooth = exp_decay(t_smooth, *theta_fit_params)
        
        # 準備含水量方程式字符串
        delta_theta, k, theta_0 = theta_fit_params
        theta_equation = f'θ(t) = {delta_theta:.2f}·exp(-{k:.3f}·t) + {theta_0:.2f}'
        
        ax.plot(t_smooth, theta_fit_smooth, 'r-', linewidth=3, 
                label=f'組合擬合曲線\n(τ={tau_theta:.2f} 天)')
    else:
        theta_equation = '含水量擬合失敗'
    
    ax.set_xlabel('組合延時 (天)', fontsize=12)
    ax.set_ylabel('體積含水量 θ (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.0, 1), loc='upper right', fontsize=10)
    
    # 在圖中添加方程式文本框
    equation_text = f'擬合方程式：\n\n{theta_equation}'
    
    # 創建文本框
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.1, equation_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.title('組合乾燥期間分析：含水量衰減', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def main():
    """主程式"""
    # 讀取資料
    print("正在讀取資料...")
    theta_rho = pd.read_csv('theta_rho.csv', parse_dates=['date_time'])
    
    # 讀取 E1_window_theta.csv
    e1_window_theta = pd.read_csv('E1_window_theta.csv')
    
    # 過濾掉空值並轉換時間格式
    e1_window_theta = e1_window_theta.dropna()
    e1_window_theta['x'] = pd.to_datetime(e1_window_theta['x'], format='%Y/%m/%d %H')
    
    # 解析E1_window_theta時間區間，兩兩一組
    wet_periods = e1_window_theta['x'].values[::2]
    dry_periods = e1_window_theta['x'].values[1::2]
    
    print(f"找到 {len(wet_periods)} 個乾燥期間")
    
    # 儲存各窗口的結果
    window_results = []
    
    # 處理每個乾燥期間
    for i in range(len(wet_periods)):
        start_time = wet_periods[i]
        end_time = dry_periods[i]
        
        print(f"處理乾燥期間 {i+1}: {start_time} 到 {end_time}")
        
        # 提取乾燥期間資料
        mask = (theta_rho['date_time'] >= start_time) & (theta_rho['date_time'] <= end_time)
        window_data = theta_rho.loc[mask].copy()
        
        if len(window_data) < 4:
            print(f"  資料點不足 ({len(window_data)} 點)")
            continue
            
        # 計算時間差（天）
        window_data['delay_days'] = (window_data['date_time'] - start_time).dt.total_seconds() / (3600*24)
        
        # 添加窗口相關資訊
        window_data['start_theta'] = window_data['mean_1m'].iloc[0]
        window_data['window_id'] = i
        
        window_results.append(window_data[['window_id', 'delay_days', 'mean_1m', 'start_theta']])
    
    if len(window_results) == 0:
        print("沒有有效的窗口資料")
        return None
    
    # 合併各窗口的資料
    df_all = pd.concat(window_results, ignore_index=True)
    
    # 根據窗口起始含水量排序（從高到低，模擬乾燥過程）
    window_order = df_all.groupby('window_id')['start_theta'].first().sort_values(ascending=False).index
    
    print(f"窗口排序順序: {list(window_order)}")
    
    cumulative_data = pd.DataFrame()
    # 調整每個窗口的延時，使得時間軸連續
    for window_id in window_order:
        window_data = df_all[df_all['window_id'] == window_id].copy()
        start_theta = window_data['start_theta'].iloc[0]
        
        if not cumulative_data.empty:
            # 找到累積資料中與當前窗口起始含水量最接近的值
            closest_idx = (cumulative_data['mean_1m'] - start_theta).abs().idxmin()
            time_offset = cumulative_data.loc[closest_idx, 'delay_days']
            adjusted_time = window_data['delay_days'] + time_offset
        else:
            adjusted_time = window_data['delay_days']
        
        window_data['delay_days'] = adjusted_time
        cumulative_data = pd.concat([cumulative_data, window_data], ignore_index=True)
        
        print(f"窗口 {window_id+1}: 起始含水量 {start_theta:.2f}%, 資料點 {len(window_data)}")
    
    print(f"\n組合後總資料點數: {len(cumulative_data)}")
    print(f"時間範圍: {cumulative_data['delay_days'].min():.2f} - {cumulative_data['delay_days'].max():.2f} 天")
    print(f"含水量範圍: {cumulative_data['mean_1m'].min():.2f} - {cumulative_data['mean_1m'].max():.2f} %")
    
    # 對組合後的資料進行擬合
    tau_theta, r2_theta, theta_params = fit_theta_decay(cumulative_data['delay_days'], cumulative_data['mean_1m'])
    
    print(f"\n組合擬合結果:")
    print(f"τ = {tau_theta:.2f} 天")
    print(f"R² = {r2_theta:.3f}")
    
    if theta_params is not None:
        delta_theta, k, theta_0 = theta_params
        print(f"擬合參數: Δθ = {delta_theta:.3f}, k = {k:.6f}, θ₀ = {theta_0:.3f}")
    
    # 繪製組合分析圖
    if tau_theta > 0:
        plot_combined_analysis(cumulative_data, tau_theta, theta_params)
    
    # 準備結果摘要
    results_summary = {
        'total_windows': len(wet_periods),
        'valid_windows': len(window_results),
        'total_data_points': len(cumulative_data),
        'time_range_days': cumulative_data['delay_days'].max() - cumulative_data['delay_days'].min(),
        'theta_range': cumulative_data['mean_1m'].max() - cumulative_data['mean_1m'].min(),
        'tau_theta': tau_theta,
        'r2_theta': r2_theta,
        'fit_params': theta_params
    }
    
    # 統計分析
    print("\n" + "="*60)
    print("統計分析結果")
    print("="*60)
    
    print(f"總窗口數: {results_summary['total_windows']}")
    print(f"有效窗口數: {results_summary['valid_windows']}")
    print(f"總資料點數: {results_summary['total_data_points']}")
    print(f"時間跨度: {results_summary['time_range_days']:.2f} 天")
    print(f"含水量變化範圍: {results_summary['theta_range']:.2f} %")
    
    if tau_theta > 0:
        print(f"\n組合擬合法 τ 結果:")
        print(f"  τ = {tau_theta:.2f} 天")
        print(f"  R² = {r2_theta:.3f}")
        
        if theta_params is not None:
            delta_theta, k, theta_0 = theta_params
            print(f"  擬合參數:")
            print(f"    Δθ (初始變化幅度) = {delta_theta:.3f} %")
            print(f"    k (衰減常數) = {k:.6f} day⁻¹")
            print(f"    θ₀ (基準含水量) = {theta_0:.3f} %")
    else:
        print("\n組合擬合失敗")
    
    # 保存組合後的資料
    cumulative_data.to_csv('combined_theta_decay_data.csv', index=False)
    print(f"\n組合資料已保存到 'combined_theta_decay_data.csv'")
    
    return results_summary, cumulative_data

if __name__ == "__main__":
    # 執行主程式
    results, combined_data = main()
    
    # 保存結果摘要
    import json
    with open('combined_tau_analysis_results.json', 'w', encoding='utf-8') as f:
        # 轉換 numpy 類型為 Python 原生類型以便 JSON 序列化
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value
        
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果摘要已保存到 'combined_tau_analysis_results.json'")
