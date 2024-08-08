# %%
import os
from datetime import datetime
import vtk
import pygimli as pg
def check_files_in_directory(directory_path):
    # �s�x�ѪR�X�Ӫ�����M��󧨦W��
    folders_with_dates = []

    # �M����Ƨ������Ҧ��ɮ�
    for filename in sorted(os.listdir(directory_path)):
        # �ˬd�ɮצW�٬O�_�ŦX�S�w�榡
        if filename.endswith('_m_E3'):
            date_str = filename[:8]  # �����������
            try:
                # �ഫ����榡�q 'YYMMDDHH' �� datetime ��H
                date = datetime.strptime(date_str, '%y%m%d%H')
                folders_with_dates.append((filename, date))
            except ValueError:
                # �p�G����榡�����T�A�������ɮ�
                continue

    return folders_with_dates

# ��X�ؿ�
output_path = r'D:\R2MSDATA\TARI_E3_test\output_318_406_rep2'
folders_with_dates = check_files_in_directory(output_path)

# �ЫئX�ּƾڶ�
append_filter = vtk.vtkAppendFilter()
append_filter.MergePointsOn()  # �X�֦@�θ`�I

# �]�m�ɶ������ݩ�
for folder_name, date in folders_with_dates[:3]:
    print(f"Processing folder: {folder_name} with date: {date}")
    vtk_path = os.path.join(output_path, folder_name, 'ERTManager', 'resistivity.vtk')
    mesh = pg.load(os.path.join(output_path, folder_name, 'ERTManager', 'resistivity-pd.bms'))
    
    # Ū��VTK���
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllScalarsOn()
    reader.Update()
    

    # ����ƾڨó]�m�ɶ��ݩ�
    data = reader.GetOutput()
    num_points = data.GetNumberOfCells()
    print("Number of points:", data.GetNumberOfPoints())
    print("Number of cells:", data.GetNumberOfCells())

    
    
    if num_points > 0:
        # ���Ҧ��ݩʳ]�m�ɶ�����
        time_array = vtk.vtkFloatArray()
        time_array.SetName("TimeIndex")
        time_array.SetNumberOfComponents(1)
        time_array.SetNumberOfTuples(num_points)
        time_value = date.timestamp()  # �Ndatetime��H�ഫ���ɶ��W
        time_array.FillComponent(0, time_value)
        data.GetCellData().AddArray(time_array)
        
        # �C�X�Ҧ��ݩʥH�i���ˬd
        cell_data = data.GetCellData()
        for i in range(cell_data.GetNumberOfArrays()):
            array_name = cell_data.GetArrayName(i)
            print(f"Found array: {array_name}")
        
        # �ˬd�Ҧ��ݩʬO�_�s�b
        required_arrays = ["Coverage", "Resistivity", "Resistivity_(log10)"]
        for array_name in required_arrays:
            if cell_data.GetArray(array_name) is None:
                print(f"Warning: {array_name} not found in {vtk_path}")
        
        # �K�[��X�ֹL�o��
        append_filter.AddInputData(data)

# ��s�L�o���üg�J�X�᪺֫���
append_filter.Update()
merged_data = append_filter.GetOutput()

# �ˬd�X�᪺֫�ƾڶ�
if merged_data.GetNumberOfPoints() > 0:
    # �ˬd�X�ּƾڶ����O�_�]�t�Ҧ��ݩ�
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
