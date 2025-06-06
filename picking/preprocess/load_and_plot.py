"""
Load and plot processed ERT data with array type classification
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import SITES_CONFIG, HYDRO_DATA_PATH

plt.rcParams["font.family"] = "Microsoft Sans Serif"

class ERTDataLoader:
    def __init__(self, site_name):
        self.site_name = site_name
        self.config = SITES_CONFIG[site_name]
        self.output_file = self.config['output_file']
        self.hydro_sheet = self.config['hydro_sheet']
        
        # 載入資料
        self.load_data()
    
    def load_data(self):
        """載入已處理的資料"""
        try:
            with open(self.output_file, 'rb') as f:
                self.dates = pickle.load(f)
                self.median_RHOA = pickle.load(f)
                self.Q1_RHOA = pickle.load(f)
                self.Q3_RHOA = pickle.load(f)
                self.median_RHOA_alpha = pickle.load(f)
                self.Q1_RHOA_alpha = pickle.load(f)
                self.Q3_RHOA_alpha = pickle.load(f)
                self.median_RHOA_beta = pickle.load(f)
                self.Q1_RHOA_beta = pickle.load(f)
                self.Q3_RHOA_beta = pickle.load(f)
                self.median_RHOA_gamma = pickle.load(f)
                self.Q1_RHOA_gamma = pickle.load(f)
                self.Q3_RHOA_gamma = pickle.load(f)
            print(f"成功載入 {self.site_name} 的資料")
        except FileNotFoundError:
            print(f"找不到 {self.output_file}，請先執行資料處理")
            raise
        except Exception as e:
            print(f"載入資料時發生錯誤: {e}")
            raise
    
    def read_hydro_data(self):
        """讀取水文資料"""
        try:
            df = pd.read_excel(HYDRO_DATA_PATH, sheet_name=self.hydro_sheet)
            df['TIME'] = pd.to_datetime(df['TIME'])
            df.set_index('TIME', inplace=True)
            df['Rain(mm)'] = pd.to_numeric(df['Rain(mm)'], errors='coerce')
            df.dropna(subset=['Rain(mm)'], inplace=True)
            daily_rainfall = df['Rain(mm)']
            return daily_rainfall
        except Exception as e:
            print(f"讀取水文資料時發生錯誤: {e}")
            return None
    
    def plot_comparison(self, show_quartiles=True):
        """繪製不同陣列類型的比較圖"""
        daily_rainfall = self.read_hydro_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 整體比較
        ax1 = axes[0, 0]
        ax1.plot(self.dates, self.median_RHOA, 'ko', markersize=2, label='整體')
        ax1.plot(self.dates, self.median_RHOA_alpha, 'ro', markersize=1.5, alpha=0.7, label='Alpha (W/S/G)')
        ax1.plot(self.dates, self.median_RHOA_beta, 'bo', markersize=1.5, alpha=0.7, label='Beta (DD)')
        ax1.plot(self.dates, self.median_RHOA_gamma, 'go', markersize=1.5, alpha=0.7, label='Gamma (其他)')
        
        if show_quartiles:
            ax1.fill_between(self.dates, self.Q1_RHOA, self.Q3_RHOA, alpha=0.2, color='black')
        
        ax1.set_title(f'{self.site_name} - 所有陣列類型比較')
        ax1.set_ylabel('視電阻率中位數 ($\Omega m$)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Alpha 陣列
        ax2 = axes[0, 1]
        ax2.plot(self.dates, self.median_RHOA_alpha, 'ro', markersize=2)
        if show_quartiles:
            ax2.fill_between(self.dates, self.Q1_RHOA_alpha, self.Q3_RHOA_alpha, alpha=0.3, color='red')
        ax2.set_title('Alpha 陣列 (Wenner/Schlumberger/Gradient)')
        ax2.set_ylabel('視電阻率中位數 ($\Omega m$)')
        ax2.grid(True, alpha=0.3)
        
        # Beta 陣列
        ax3 = axes[1, 0]
        ax3.plot(self.dates, self.median_RHOA_beta, 'bo', markersize=2)
        if show_quartiles:
            ax3.fill_between(self.dates, self.Q1_RHOA_beta, self.Q3_RHOA_beta, alpha=0.3, color='blue')
        ax3.set_title('Beta 陣列 (Dipole-dipole)')
        ax3.set_ylabel('視電阻率中位數 ($\Omega m$)')
        ax3.set_xlabel('日期')
        ax3.grid(True, alpha=0.3)
        
        # Gamma 陣列
        ax4 = axes[1, 1]
        ax4.plot(self.dates, self.median_RHOA_gamma, 'go', markersize=2)
        if show_quartiles:
            ax4.fill_between(self.dates, self.Q1_RHOA_gamma, self.Q3_RHOA_gamma, alpha=0.3, color='green')
        ax4.set_title('Gamma 陣列 (其他配置)')
        ax4.set_ylabel('視電阻率中位數 ($\Omega m$)')
        ax4.set_xlabel('日期')
        ax4.grid(True, alpha=0.3)
        
        # 設定所有子圖的x軸格式
        for ax in axes.flat:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.set_xlim(self.dates[0], self.dates[-1])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 加入降雨量到第一個子圖
        if daily_rainfall is not None:
            ax2_rain = ax1.twinx()
            ax2_rain.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c')
            ax2_rain.set_ylabel('降雨量 (mm)', color='c')
            ax2_rain.tick_params(axis='y', labelcolor='c')
        
        plt.show()
    
    def plot_time_series(self):
        """繪製時間序列圖"""
        daily_rainfall = self.read_hydro_data()
        
        fig, ax1 = plt.subplots(figsize=(25, 8))
        
        # 主要視電阻率資料
        ax1.plot(self.dates, self.median_RHOA, 'ko', markersize=2, label='整體中位數')
        ax1.fill_between(self.dates, self.Q1_RHOA, self.Q3_RHOA, alpha=0.2, color='black', label='四分位距')
        
        ax1.set_xlabel('日期')
        ax1.set_ylabel('視電阻率中位數 ($\Omega m$)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 設定x軸格式
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.set_xlim(self.dates[0], self.dates[-1])
        
        # 加入降雨量
        if daily_rainfall is not None:
            ax2 = ax1.twinx()
            ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='降雨量')
            ax2.set_ylabel('降雨量 (mm)', color='c')
            ax2.tick_params(axis='y', labelcolor='c')
            ax2.legend(loc='upper right')
        
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.title(f'{self.site_name} 站點視電阻率時間序列')
        plt.tight_layout()
        plt.show()
    
    def statistics_summary(self):
        """統計摘要"""
        print(f"\n{self.site_name} 站點統計摘要:")
        print("="*50)
        
        # 移除NaN值進行統計
        alpha_values = [x for x in self.median_RHOA_alpha if not np.isnan(x)]
        beta_values = [x for x in self.median_RHOA_beta if not np.isnan(x)]
        gamma_values = [x for x in self.median_RHOA_gamma if not np.isnan(x)]
        
        print(f"資料期間: {self.dates[0].strftime('%Y-%m-%d')} 至 {self.dates[-1].strftime('%Y-%m-%d')}")
        print(f"總測量次數: {len(self.dates)}")
        print()
        
        print("視電阻率統計 (Ω·m):")
        print(f"整體中位數 - 平均: {np.mean(self.median_RHOA):.2f}, 標準差: {np.std(self.median_RHOA):.2f}")
        
        if alpha_values:
            print(f"Alpha 陣列 - 平均: {np.mean(alpha_values):.2f}, 標準差: {np.std(alpha_values):.2f}, 資料點: {len(alpha_values)}")
        
        if beta_values:
            print(f"Beta 陣列 - 平均: {np.mean(beta_values):.2f}, 標準差: {np.std(beta_values):.2f}, 資料點: {len(beta_values)}")
        
        if gamma_values:
            print(f"Gamma 陣列 - 平均: {np.mean(gamma_values):.2f}, 標準差: {np.std(gamma_values):.2f}, 資料點: {len(gamma_values)}")

def compare_all_sites():
    """比較所有站點的資料"""
    sites = ['E1', 'E2', 'E3']
    fig, axes = plt.subplots(len(sites), 1, figsize=(20, 15))
    
    for i, site in enumerate(sites):
        try:
            loader = ERTDataLoader(site)
            ax = axes[i]
            
            ax.plot(loader.dates, loader.median_RHOA, 'ko', markersize=1.5, label='整體')
            ax.plot(loader.dates, loader.median_RHOA_alpha, 'ro', markersize=1, alpha=0.7, label='Alpha')
            ax.plot(loader.dates, loader.median_RHOA_beta, 'bo', markersize=1, alpha=0.7, label='Beta')
            ax.plot(loader.dates, loader.median_RHOA_gamma, 'go', markersize=1, alpha=0.7, label='Gamma')
            
            ax.set_title(f'{site} 站點')
            ax.set_ylabel('視電阻率 ($\Omega m$)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if i == len(sites) - 1:
                ax.set_xlabel('日期')
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        except Exception as e:
            print(f"無法載入 {site} 的資料: {e}")
    
    plt.tight_layout()
    plt.show()

def main():
    """主程式 - 展示所有分析功能"""
    sites = ['E1', 'E2', 'E3']
    
    print("ERT 資料分析工具")
    print("="*50)
    
    for site in sites:
        try:
            print(f"\n處理 {site} 站點...")
            loader = ERTDataLoader(site)
            
            # 統計摘要
            loader.statistics_summary()
            
            # 繪製比較圖
            loader.plot_comparison()
            
            # 繪製時間序列
            loader.plot_time_series()
            
        except Exception as e:
            print(f"無法處理 {site} 站點: {e}")
            continue
    
    # 比較所有站點
    print("\n繪製所有站點比較圖...")
    compare_all_sites()

if __name__ == "__main__":
    main() 