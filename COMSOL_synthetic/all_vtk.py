import os
from datetime import datetime
from paraview.simple import *

def check_files_in_directory(directory_path):
    # 存儲解析出來的日期和文件夾名稱
    folders_with_dates = []

    # 遍歷資料夾中的所有檔案
    for filename in sorted(os.listdir(directory_path)):
        # 檢查檔案名稱是否符合特定格式
        if filename.endswith('_m_E3'):
            date_str = filename[:8]  # 提取日期部分
            try:
                # 轉換日期格式從 'YYMMDDHH' 到 datetime 對象
                date = datetime.strptime(date_str, '%y%m%d%H')
                folders_with_dates.append((filename, date))
            except ValueError:
                # 如果日期格式不正確，忽略此檔案
                continue

    return folders_with_dates

# 資料夾路徑
output_path = r'D:\R2MSDATA\TARI_E3_test\output_318_406_rep2'
folders_with_dates = check_files_in_directory(output_path)

# 創建一個新的 ParaView 項目
Disconnect()
Connect()

# 加載所有的 vtk 檔案並設定時間標記
for folder_name, date in folders_with_dates:
    # 加載 vtk 檔案
    vtk_file = os.path.join(output_path, folder_name, 'ERTManager', 'resistivity.vtk')
    reader = LegacyVTKReader(FileNames=[vtk_file])
    
    # 設定時間屬性
    time_annotation = AnnotateTimeFilter(Input=reader)
    time_annotation.Format = date.strftime('%Y-%m-%d %H:%M:%S')
    
    # 顯示數據
    Show(reader)
    Show(time_annotation)

# 更新時間步長
time_steps = [date.timestamp() for _, date in folders_with_dates]
animation_scene = GetAnimationScene()
animation_scene.TimeKeeper.TimestepValues = time_steps

# 啟動互動式 ParaView
Render()
