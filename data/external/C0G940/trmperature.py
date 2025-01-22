# %%
# 請將目錄中所有csv檔案，名稱為G2F820-2024-10.csv.csv, G2F820-2024-11.csv合併成一個檔案，並且只保留以下欄位：

# 1. ObsTime
# 2. Precp
# 請將合併後的觀測時間依照日期組合起來，例如2024-10，因為原本的資料是每天一個觀測值，所以合併後的資料也是每天一個觀測值
# 請將合併後的資料存成一個csv檔案，檔名為merged_data.csv

import os
import pandas as pd

# 取得目錄中的所有開頭為467540-*.csv的csv檔案
csv_files = [f for f in os.listdir() if f.startswith('C0G940') and f.endswith('.csv')]

# 創建一個空的 DataFrame 來存儲所有數據
all_data = pd.DataFrame()

# 讀取每個csv檔案，並將其添加到 all_data
for file in csv_files:
    # 從文件名中提取日期：格式為2024-09-03，日期來自檔名
    date_str = file.split('-')[1] + '-' + file.split('-')[2][:2]
    # 讀取 CSV 文件，使用英文名稱，第一航的中文名稱須將其跳過
    df = pd.read_csv(file, skiprows=1)
    # 只保留觀測時間和降水量這兩個欄位
    df = df[['ObsTime', 'Temperature']]

    # 將觀測時間轉換為日期時間格式，日期來自檔名，時間(day)來自ObsTime
    df['Time'] = pd.to_datetime(date_str) + pd.to_timedelta(df['ObsTime'] - 1, unit='d')
    all_data = pd.concat([all_data, df], ignore_index=True)

# 刪除 ObsTime 這個欄位
all_data = all_data.drop(columns=['ObsTime'])
# 排序 Time 這個欄位
all_data = all_data.sort_values('Time')
# 將 Time 這個欄位移動到第一個欄位
all_data.insert(0, 'Time', all_data.pop('Time'))

# 將合併後的數據保存為 CSV 文件
all_data.to_csv('merged_data.csv', index=False)
