# %%
import os
from datetime import datetime
import vtk
import pygimli as pg
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

# 輸出目錄
output_path = r'D:\R2MSDATA\TARI_E3_test\output_318_406_rep2'
folders_with_dates = check_files_in_directory(output_path)

# 創建合併數據集
append_filter = vtk.vtkAppendFilter()
append_filter.MergePointsOn()  # 合併共用節點

# 設置時間索引屬性
for folder_name, date in folders_with_dates[:3]:
    print(f"Processing folder: {folder_name} with date: {date}")
    vtk_path = os.path.join(output_path, folder_name, 'ERTManager', 'resistivity.vtk')
    mesh = pg.load(os.path.join(output_path, folder_name, 'ERTManager', 'resistivity-pd.bms'))
    
    # 讀取VTK文件
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllScalarsOn()
    reader.Update()
    

    # 獲取數據並設置時間屬性
    data = reader.GetOutput()
    num_points = data.GetNumberOfCells()
    print("Number of points:", data.GetNumberOfPoints())
    print("Number of cells:", data.GetNumberOfCells())

    
    
    if num_points > 0:
        # 為所有屬性設置時間索引
        time_array = vtk.vtkFloatArray()
        time_array.SetName("TimeIndex")
        time_array.SetNumberOfComponents(1)
        time_array.SetNumberOfTuples(num_points)
        time_value = date.timestamp()  # 將datetime對象轉換為時間戳
        time_array.FillComponent(0, time_value)
        data.GetCellData().AddArray(time_array)
        
        # 列出所有屬性以進行檢查
        cell_data = data.GetCellData()
        for i in range(cell_data.GetNumberOfArrays()):
            array_name = cell_data.GetArrayName(i)
            print(f"Found array: {array_name}")
        
        # 檢查所有屬性是否存在
        required_arrays = ["Coverage", "Resistivity", "Resistivity_(log10)"]
        for array_name in required_arrays:
            if cell_data.GetArray(array_name) is None:
                print(f"Warning: {array_name} not found in {vtk_path}")
        
        # 添加到合併過濾器
        append_filter.AddInputData(data)

# 更新過濾器並寫入合併後的文件
append_filter.Update()
merged_data = append_filter.GetOutput()

# 檢查合併後的數據集
if merged_data.GetNumberOfPoints() > 0:
    # 檢查合併數據集中是否包含所有屬性
    cell_data = merged_data.GetPointData()
    for i in range(cell_data.GetNumberOfArrays()):
        array_name = cell_data.GetArrayName(i)
        print(f"Merged array: {array_name}")

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName("merged_output.vtk")
    writer.SetInputData(merged_data)
    writer.Write()
else:
    print("No valid data found in the input files.")
