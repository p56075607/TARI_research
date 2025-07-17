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

def fit_theta_decay(time_drydown, theta_values):
    """含水量直接擬合計算τ（使用修正函數）"""
    try:
        # 初始參數估計
        delta_theta_guess = theta_values.iloc[0] - theta_values.iloc[-1]
        k_guess = 0.1
        theta_0_guess = theta_values.iloc[-1]
        
        # 擬合修正後的指數衰減函數
        popt, pcov = curve_fit(
            exp_decay, 
            time_drydown, 
            theta_values,
            p0=[delta_theta_guess, k_guess, theta_0_guess],
            bounds=([-np.inf, 0.001, 0], [np.inf, 2.0, 50])
        )
        
        delta_theta_fit, k_fit, theta_0_fit = popt
        tau = 1 / k_fit
        
        # 檢查擬合品質
        r_squared = calculate_r_squared(theta_values, exp_decay(time_drydown, *popt))
        
        if r_squared > 0.3:  # 降低R²閾值以接受更多擬合結果
            return tau, r_squared, popt
        else:
            return -1, -1, None
            
    except Exception as e:
        print(f"含水量擬合失敗: {e}")
        return -1, -1, None

def calculate_tau_from_resistivity(time_drydown, rho_values, m=2.0):
    """使用電阻率理論計算τ"""
    try:
        # 計算ln(ρ)
        ln_rho = np.log(rho_values)
        
        # 線性擬合 ln(ρ) vs time
        coeffs = np.polyfit(time_drydown, ln_rho, 1)
        slope_ln_rho = coeffs[0]
        
        # 計算R²
        ln_rho_pred = coeffs[0] * time_drydown + coeffs[1]
        r_squared = calculate_r_squared(ln_rho, ln_rho_pred)
        
        # 根據理論: d ln(ρ)/dt = m * k
        k = slope_ln_rho / m
        
        if k > 0 and r_squared > 0.3:
            tau = 1 / k
            return tau, r_squared, coeffs
        else:
            return -1, -1, None
            
    except Exception as e:
        print(f"電阻率計算失敗: {e}")
        return -1, -1, None

def calculate_r_squared(y_observed, y_predicted):
    """計算決定係數R²"""
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)

def plot_drydown_analysis(time_drydown, drydown_data, tau_theta, tau_rho, 
                         theta_fit_params, rho_fit_params, period_num):
    """繪製乾燥期間分析圖"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 生成平滑的時間軸
    t_smooth = np.linspace(time_drydown.min(), time_drydown.max(), 100)
    
    # 含水量擬合
    if theta_fit_params is not None:
        theta_fit_smooth = exp_decay(t_smooth, *theta_fit_params)
        ax1.scatter(time_drydown, drydown_data['mean_1m'], color='blue', s=50, alpha=0.7, 
                   label='觀測含水量')
        
        # 準備含水量方程式字符串
        delta_theta, k, theta_0 = theta_fit_params
        theta_equation = f'θ(t) = {delta_theta:.2f}·exp(-{k:.3f}·t) + {theta_0:.2f}'
        
        ax1.plot(t_smooth, theta_fit_smooth, 'b-', linewidth=2, 
                label=f'擬合曲線 (τ={tau_theta:.2f} 天)')
    else:
        ax1.scatter(time_drydown, drydown_data['mean_1m'], color='blue', s=50, alpha=0.7, 
                   label='觀測含水量')
        theta_equation = '含水量擬合失敗'
    
    ax1.set_xlabel('時間 (天)', fontsize=12)
    ax1.set_ylabel('體積含水量 θ (%)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # 右側y軸 - ln(ρ)
    ax2 = ax1.twinx()
    
    ln_rho = np.log(drydown_data['rhoa'])
    
    # 線性擬合ln(ρ)
    if rho_fit_params is not None:
        ln_rho_fit_smooth = rho_fit_params[0] * t_smooth + rho_fit_params[1]
        ax2.plot(t_smooth, ln_rho_fit_smooth, 'r-', linewidth=2, 
                label=f'線性擬合 (τ={tau_rho:.2f} 天)')
        
        # 準備電阻率方程式字符串
        slope, intercept = rho_fit_params
        rho_equation = f'ln(ρ) = {slope:.4f}·t + {intercept:.2f}'
    else:
        rho_equation = '電阻率擬合失敗'
    
    ax2.scatter(time_drydown, ln_rho, color='red', s=50, alpha=0.7, label='觀測 ln(ρ)')
    ax2.set_ylabel('ln(視電阻率 ρ)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 合併圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # 在圖中添加方程式文本框
    equation_text = f'擬合方程式：\n\n含水量方法：\n{theta_equation}\n\n電阻率方法：\n{rho_equation}'
    
    # 創建文本框
    textstr = equation_text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.title(f'乾燥期間 {period_num} 分析：含水量衰減與電阻率變化', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def main():
    """主程式"""
    # 讀取資料
    print("正在讀取資料...")
    theta_rho = pd.read_csv('theta_rho.csv', parse_dates=['date_time'])
    e1_window = pd.read_csv('E1_window.csv', parse_dates=['x'])
    
    # 解析E1_window時間區間，兩兩一組
    wet_periods = e1_window['x'].values[::2]
    dry_periods = e1_window['x'].values[1::2]
    
    print(f"找到 {len(wet_periods)} 個乾燥期間")
    
    # 儲存結果
    results = []
    tau_theta_list = []
    tau_rho_list = []
    
    # 分析每個乾燥期間
    for i in range(len(wet_periods)):
        start_time = wet_periods[i]
        end_time = dry_periods[i]
        
        print(f"\n分析乾燥期間 {i+1}: {start_time} 到 {end_time}")
        
        # 提取乾燥期間資料
        mask = (theta_rho['date_time'] >= start_time) & (theta_rho['date_time'] <= end_time)
        drydown_data = theta_rho.loc[mask].copy()
        
        if len(drydown_data) < 4:
            print(f"  資料點不足 ({len(drydown_data)} 點)")
            continue
            
        # 計算時間差（天）
        time_drydown = (drydown_data['date_time'] - drydown_data['date_time'].iloc[0]).dt.total_seconds() / (3600*24)
        
        # 方法1：含水量直接擬合
        tau_theta, r2_theta, theta_params = fit_theta_decay(time_drydown, drydown_data['mean_1m'])
        
        # 方法2：電阻率理論計算
        tau_rho, r2_rho, rho_params = calculate_tau_from_resistivity(time_drydown, drydown_data['rhoa'])
        
        # 記錄結果
        results.append({
            'period': i+1,
            'start_time': start_time,
            'end_time': end_time,
            'duration_days': time_drydown.max(),
            'data_points': len(drydown_data),
            'tau_theta': tau_theta,
            'r2_theta': r2_theta,
            'tau_rho': tau_rho,
            'r2_rho': r2_rho,
            'theta_change': drydown_data['mean_1m'].iloc[0] - drydown_data['mean_1m'].iloc[-1],
            'rho_change': drydown_data['rhoa'].iloc[-1] - drydown_data['rhoa'].iloc[0]
        })
        
        # 收集有效τ值
        if tau_theta > 0:
            tau_theta_list.append(tau_theta)
        if tau_rho > 0:
            tau_rho_list.append(tau_rho)
        
        print(f"  含水量方法: τ = {tau_theta:.2f} 天, R² = {r2_theta:.3f}")
        print(f"  電阻率方法: τ = {tau_rho:.2f} 天, R² = {r2_rho:.3f}")
        
        # 繪製有效期間的詳細分析圖
        if (tau_theta > 0 or tau_rho > 0):
            plot_drydown_analysis(time_drydown, drydown_data, tau_theta, tau_rho, 
                                theta_params, rho_params, i+1)
    
    # 統計分析
    print("\n" + "="*60)
    print("統計分析結果")
    print("="*60)
    
    # 建立結果DataFrame
    df_results = pd.DataFrame(results)
    
    # 篩選有效結果
    valid_theta = df_results[df_results['tau_theta'] > 0]
    valid_rho = df_results[df_results['tau_rho'] > 0]
    
    print(f"\n有效分析期間:")
    print(f"  含水量方法: {len(valid_theta)} 個期間")
    print(f"  電阻率方法: {len(valid_rho)} 個期間")
    
    if len(tau_theta_list) > 0:
        print(f"\n含水量直接擬合法 τ 統計:")
        print(f"  平均值: {np.mean(tau_theta_list):.2f} ± {np.std(tau_theta_list):.2f} 天")
        print(f"  中位數: {np.median(tau_theta_list):.2f} 天")
        print(f"  範圍: {np.min(tau_theta_list):.2f} - {np.max(tau_theta_list):.2f} 天")
        print(f"  平均R²: {valid_theta['r2_theta'].mean():.3f}")
    
    if len(tau_rho_list) > 0:
        print(f"\n電阻率理論計算法 τ 統計:")
        print(f"  平均值: {np.mean(tau_rho_list):.2f} ± {np.std(tau_rho_list):.2f} 天")
        print(f"  中位數: {np.median(tau_rho_list):.2f} 天")
        print(f"  範圍: {np.min(tau_rho_list):.2f} - {np.max(tau_rho_list):.2f} 天")
        print(f"  平均R²: {valid_rho['r2_rho'].mean():.3f}")
    
    # 比較兩種方法
    if len(tau_theta_list) > 0 and len(tau_rho_list) > 0:
        print(f"\n兩種方法比較:")
        print(f"  含水量法平均τ: {np.mean(tau_theta_list):.2f} 天")
        print(f"  電阻率法平均τ: {np.mean(tau_rho_list):.2f} 天")
        print(f"  平均差異: {abs(np.mean(tau_theta_list) - np.mean(tau_rho_list)):.2f} 天")
        
        # 相關性分析（如果有配對的結果）
        paired_results = df_results[(df_results['tau_theta'] > 0) & (df_results['tau_rho'] > 0)]
        if len(paired_results) > 1:
            correlation = paired_results['tau_theta'].corr(paired_results['tau_rho'])
            print(f"  兩種方法相關係數: {correlation:.3f}")
    
    # 顯示詳細結果表
    print(f"\n詳細結果表:")
    print(df_results[['period', 'duration_days', 'data_points', 'tau_theta', 'r2_theta', 
                     'tau_rho', 'r2_rho', 'theta_change', 'rho_change']].to_string(index=False))
    
    # 繪製τ值分佈比較圖
    if len(tau_theta_list) > 0 or len(tau_rho_list) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # τ值分佈直方圖
        ax1.hist(tau_theta_list, bins=10, alpha=0.7, label='含水量方法', color='blue')
        ax1.hist(tau_rho_list, bins=10, alpha=0.7, label='電阻率方法', color='red')
        ax1.set_xlabel('τ (天)')
        ax1.set_ylabel('頻次')
        ax1.set_title('排水速度因子τ分佈比較')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱線圖比較
        data_for_box = []
        labels_for_box = []
        if len(tau_theta_list) > 0:
            data_for_box.append(tau_theta_list)
            labels_for_box.append('含水量方法')
        if len(tau_rho_list) > 0:
            data_for_box.append(tau_rho_list)
            labels_for_box.append('電阻率方法')
        
        ax2.boxplot(data_for_box, labels=labels_for_box)
        ax2.set_ylabel('τ (天)')
        ax2.set_title('排水速度因子τ箱線圖比較')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return df_results

if __name__ == "__main__":
    # 執行主程式
    results_df = main()
    
    # 保存結果到CSV
    results_df.to_csv('tau_analysis_results.csv', index=False)
    print(f"\n結果已保存到 'tau_analysis_results.csv'")
