import os
from datetime import datetime
from paraview.simple import *

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

# ��Ƨ����|
output_path = r'D:\R2MSDATA\TARI_E3_test\output_318_406_rep2'
folders_with_dates = check_files_in_directory(output_path)

# �Ыؤ@�ӷs�� ParaView ����
Disconnect()
Connect()

# �[���Ҧ��� vtk �ɮרó]�w�ɶ��аO
for folder_name, date in folders_with_dates:
    # �[�� vtk �ɮ�
    vtk_file = os.path.join(output_path, folder_name, 'ERTManager', 'resistivity.vtk')
    reader = LegacyVTKReader(FileNames=[vtk_file])
    
    # �]�w�ɶ��ݩ�
    time_annotation = AnnotateTimeFilter(Input=reader)
    time_annotation.Format = date.strftime('%Y-%m-%d %H:%M:%S')
    
    # ��ܼƾ�
    Show(reader)
    Show(time_annotation)

# ��s�ɶ��B��
time_steps = [date.timestamp() for _, date in folders_with_dates]
animation_scene = GetAnimationScene()
animation_scene.TimeKeeper.TimestepValues = time_steps

# �Ұʤ��ʦ� ParaView
Render()
