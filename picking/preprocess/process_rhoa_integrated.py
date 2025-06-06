"""
Integrated ERT data processing script
Processes E1, E2, E3 data with array type classification
Calculates median apparent resistivity for alpha, beta, gamma array types
"""

import sys
import os
from config import SITES_CONFIG, TOOLBOX_PATH, HYDRO_DATA_PATH, FORMULA_CHOOSE
sys.path.append(TOOLBOX_PATH)

from inverison_util import convertURF
from ridx_analyse import ridx_analyse
from urf2ohm import urf2ohm
from array_classifier import classify_array_type_midconf, get_array_indices

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"
import numpy as np
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt
from datetime import datetime, timedelta
from os.path import join
from os import listdir
import matplotlib.dates as mdates
import pickle
import pandas as pd
from collections import Counter
import winsound

class ERTProcessor:
    def __init__(self, site_name):
        self.site_name = site_name
        self.config = SITES_CONFIG[site_name]
        self.urf_path = self.config['urf_path']
        self.ridx_urf_path = self.config['ridx_urf_path']
        self.rest = self.config['rest']
        self.output_file = self.config['output_file']
        self.hydro_sheet = self.config['hydro_sheet']
        
        # Initialize data containers
        self.dates = []
        self.median_RHOA = []
        self.median_RHOA_alpha = []
        self.median_RHOA_beta = []
        self.median_RHOA_gamma = []
        self.Q1_RHOA = []
        self.Q3_RHOA = []
        self.Q1_RHOA_alpha = []
        self.Q3_RHOA_alpha = []
        self.Q1_RHOA_beta = []
        self.Q3_RHOA_beta = []
        self.Q1_RHOA_gamma = []
        self.Q3_RHOA_gamma = []

    def get_datetime_list_and_count(self, directory):
        """從檔案名稱擷取日期時間資訊"""
        datetime_list = []
        urffiles = sorted([_ for _ in listdir(directory) if _.endswith('.urf')])
        
        for filename in urffiles:
            if len(filename) > 8 and filename[:8].isdigit():
                date_time_str = filename[:8]
                date_time_obj = datetime.strptime(date_time_str, '%y%m%d%H')
                datetime_list.append(date_time_obj)

        date_count = Counter([dt.date() for dt in datetime_list])
        return datetime_list, date_count

    def setup_array_classification(self, data):
        """設定陣列類型分類（使用midconfERT邏輯）"""
        # 取得電極座標和資料點
        a_indices = data['a']
        b_indices = data['b']
        m_indices = data['m']
        n_indices = data['n']
        
        # 分類每個測量點的陣列類型（使用midconfERT邏輯）
        array_types = []
        for i in range(len(data['r'])):
            array_type = classify_array_type_midconf(a_indices[i], b_indices[i], 
                                           m_indices[i], n_indices[i])
            array_types.append(array_type)
        
        # 加入陣列類型資訊
        data['array_type'] = np.array(array_types)
        
        # 取得不同陣列類型的索引
        alpha_indices = np.where(np.array(array_types) == 'alpha')[0]
        beta_indices = np.where(np.array(array_types) == 'beta')[0]
        gamma_indices = np.where(np.array(array_types) == 'gamma')[0]
        
        return alpha_indices, beta_indices, gamma_indices

    def read_ohm_r_column(self, file_path):
        """讀取ohm檔案的電阻值欄位"""
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 擷取電極數量和位置
        num_electrodes = int(lines[0].strip().split()[0])
        electrodes = []
        idx = 2

        # 跳過電極位置
        while len(electrodes) < num_electrodes and idx < len(lines):
            line = lines[idx].strip()
            if line and not line.startswith('#'):
                electrodes.append([float(x) for x in line.split()])
            idx += 1

        # 擷取資料點數量
        while idx < len(lines) and not lines[idx].strip().split()[0].isdigit():
            idx += 1

        if idx >= len(lines):
            raise ValueError("找不到資料點行")

        num_data = int(lines[idx].strip().split()[0])
        idx += 2

        r_values = []
        while len(r_values) < num_data and idx < len(lines):
            line = lines[idx].strip()
            if line and not line.startswith('#'):
                data = line.split()
                r_values.append(float(data[4]))  # 假設r是第5欄
            idx += 1

        if len(r_values) != num_data:
            raise ValueError("資料點數量與擷取的資料不符")

        return r_values

    def process_quality_filtering(self):
        """處理資料品質過濾"""
        print(f"處理 {self.site_name} 的資料品質分析...")
        
        unsorted_quality_info = ridx_analyse(self.ridx_urf_path, formula_choose=FORMULA_CHOOSE)
        ridx = unsorted_quality_info/100
        t3 = np.argsort(ridx)[self.rest:]
        remove_index = np.full((len(unsorted_quality_info)), False)
        
        for i in range(len(t3)):
            remove_index[t3[i]] = True
            
        return remove_index, t3

    def process_files(self):
        """處理所有urf檔案"""
        print(f"開始處理 {self.site_name} 站點...")
        
        # 取得檔案列表
        urffiles = sorted([_ for _ in listdir(self.urf_path) if _.endswith('.urf')])
        ohmfiles = sorted([_ for _ in listdir(self.urf_path) if _.endswith('.ohm')])
        
        # 取得日期資訊
        self.dates, date_count = self.get_datetime_list_and_count(self.urf_path)
        print(f"總共 {len(date_count)} 個不同的日期")
        
        # 處理資料品質過濾
        remove_index, t3 = self.process_quality_filtering()
        
        # 處理每個檔案
        for i, urf_file_name in enumerate(urffiles):
            print(f"處理進度: {i+1}/{len(urffiles)} - {urf_file_name}")
            
            # 檢查是否已有ohm檔案
            if urf_file_name[:-4]+'.ohm' in ohmfiles:
                print(f'{urf_file_name[:-4]}.urf 已處理過，跳過！')
                ohm_file_name = join(self.urf_path, urf_file_name[:-4]+'.ohm')
            else:
                print(f'正在處理: {urf_file_name}')
                ohm_file_name = urf2ohm(join(self.urf_path, urf_file_name), has_trn=False)

            # 第一個檔案需要載入完整資料結構
            if i == 0:
                data = ert.load(ohm_file_name)
                data.remove(remove_index)
                print(data)
                data['k'] = ert.createGeometricFactors(data, numerical=True)
                rhoa = data['k'] * data['r']
                
                # 設定陣列類型分類
                alpha_indices, beta_indices, gamma_indices = self.setup_array_classification(data)
                print(f"Alpha (Wenner/Schlumberger/Gradient) 測量: {len(alpha_indices)}")
                print(f"Beta (Dipole-dipole) 測量: {len(beta_indices)}")
                print(f"Gamma (其他) 測量: {len(gamma_indices)}")
                
            else:
                # 後續檔案只讀取電阻值
                r_values = self.read_ohm_r_column(ohm_file_name)
                removed_r = np.delete(r_values, t3)
                rhoa = data['k'] * removed_r

            # 計算整體中位數
            self.median_RHOA.append(np.median(rhoa))
            self.Q1_RHOA.append(np.percentile(rhoa, 25))
            self.Q3_RHOA.append(np.percentile(rhoa, 75))
            
            # 計算不同陣列類型的中位數
            if len(alpha_indices) > 0:
                rhoa_alpha = rhoa[alpha_indices]
                self.median_RHOA_alpha.append(np.median(rhoa_alpha))
                self.Q1_RHOA_alpha.append(np.percentile(rhoa_alpha, 25))
                self.Q3_RHOA_alpha.append(np.percentile(rhoa_alpha, 75))
            else:
                self.median_RHOA_alpha.append(np.nan)
                self.Q1_RHOA_alpha.append(np.nan)
                self.Q3_RHOA_alpha.append(np.nan)
                
            if len(beta_indices) > 0:
                rhoa_beta = rhoa[beta_indices]
                self.median_RHOA_beta.append(np.median(rhoa_beta))
                self.Q1_RHOA_beta.append(np.percentile(rhoa_beta, 25))
                self.Q3_RHOA_beta.append(np.percentile(rhoa_beta, 75))
            else:
                self.median_RHOA_beta.append(np.nan)
                self.Q1_RHOA_beta.append(np.nan)
                self.Q3_RHOA_beta.append(np.nan)
                
            if len(gamma_indices) > 0:
                rhoa_gamma = rhoa[gamma_indices]
                self.median_RHOA_gamma.append(np.median(rhoa_gamma))
                self.Q1_RHOA_gamma.append(np.percentile(rhoa_gamma, 25))
                self.Q3_RHOA_gamma.append(np.percentile(rhoa_gamma, 75))
            else:
                self.median_RHOA_gamma.append(np.nan)
                self.Q1_RHOA_gamma.append(np.nan)
                self.Q3_RHOA_gamma.append(np.nan)

    def save_results(self):
        """儲存處理結果"""
        print(f"儲存 {self.site_name} 的處理結果...")
        
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.dates, f)
            pickle.dump(self.median_RHOA, f)
            pickle.dump(self.Q1_RHOA, f)
            pickle.dump(self.Q3_RHOA, f)
            pickle.dump(self.median_RHOA_alpha, f)
            pickle.dump(self.Q1_RHOA_alpha, f)
            pickle.dump(self.Q3_RHOA_alpha, f)
            pickle.dump(self.median_RHOA_beta, f)
            pickle.dump(self.Q1_RHOA_beta, f)
            pickle.dump(self.Q3_RHOA_beta, f)
            pickle.dump(self.median_RHOA_gamma, f)
            pickle.dump(self.Q1_RHOA_gamma, f)
            pickle.dump(self.Q3_RHOA_gamma, f)
        
        print(f"結果已儲存至 {self.output_file}")

    def read_hydro_data(self):
        """讀取水文資料"""
        try:
            df = pd.read_excel(HYDRO_DATA_PATH, sheet_name=self.hydro_sheet)

            # 將'TIME'列轉換為日期時間格式
            df['TIME'] = pd.to_datetime(df['TIME'])
            df.set_index('TIME', inplace=True)

            # 將'Rain(mm)'列轉換為數字
            df['Rain(mm)'] = pd.to_numeric(df['Rain(mm)'], errors='coerce')
            df.dropna(subset=['Rain(mm)'], inplace=True)
            
            daily_rainfall = df['Rain(mm)']
            return daily_rainfall
        except Exception as e:
            print(f"讀取水文資料時發生錯誤: {e}")
            return None

    def plot_results(self):
        """繪製結果圖表"""
        daily_rainfall = self.read_hydro_data()
        
        fig, ax1 = plt.subplots(figsize=(25, 6))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # 繪製整體中位數
        ax1.plot(self.dates, self.median_RHOA, 'ko', markersize=2, label='Overall')
        
        # 繪製不同陣列類型的中位數
        ax1.plot(self.dates, self.median_RHOA_alpha, 'ro', markersize=1.5, alpha=0.7, label='Alpha (W/S/G)')
        ax1.plot(self.dates, self.median_RHOA_beta, 'bo', markersize=1.5, alpha=0.7, label='Beta (DD)')
        ax1.plot(self.dates, self.median_RHOA_gamma, 'go', markersize=1.5, alpha=0.7, label='Gamma (Other)')
        
        ax1.set_xlim(self.dates[0], self.dates[-1])
        ax1.set_xlabel('日期')
        ax1.set_ylabel('視電阻率中位數 ($\Omega m$)')
        ax1.grid(linestyle='--', linewidth=0.5)
        ax1.legend()
        
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()

        # 加入降雨量
        if daily_rainfall is not None:
            ax2 = ax1.twinx()
            ax2.bar(daily_rainfall.index, daily_rainfall, width=1, alpha=0.3, color='c', label='降雨量')
            ax2.set_ylabel('降雨量 (mm)', color='c')
            ax2.tick_params(axis='y', labelcolor='c')
        
        plt.title(f'{self.site_name} 站點視電阻率時間序列')
        plt.show()

def main():
    """主程式"""
    # 處理所有站點
    sites_to_process = ['E1', 'E2', 'E3']  # 可以選擇要處理的站點
    
    for site in sites_to_process:
        print(f"\n{'='*50}")
        print(f"開始處理 {site} 站點")
        print(f"{'='*50}")
        
        try:
            processor = ERTProcessor(site)
            processor.process_files()
            processor.save_results()
            processor.plot_results()
            
            # 完成提示音
            try:
                winsound.Beep(440, 1000)
            except:
                print("無法播放提示音")
                
        except Exception as e:
            print(f"處理 {site} 站點時發生錯誤: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("所有站點處理完成！")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 