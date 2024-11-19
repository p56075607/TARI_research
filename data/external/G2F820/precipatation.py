# %%
# �бN�ؿ����Ҧ�csv�ɮסA�W�٬�G2F820-2024-10.csv.csv, G2F820-2024-11.csv�X�֦��@���ɮסA�åB�u�O�d�H�U���G

# 1. ObsTime
# 2. Precp
# �бN�X�᪺֫�[���ɶ��̷Ӥ���զX�_�ӡA�Ҧp2024-10�A�]���쥻����ƬO�C�Ѥ@���[���ȡA�ҥH�X�᪺֫��Ƥ]�O�C�Ѥ@���[����
# �бN�X�᪺֫��Ʀs���@��csv�ɮסA�ɦW��merged_data.csv

import os
import pandas as pd

# ���o�ؿ������Ҧ��}�Y��467540-*.csv��csv�ɮ�
csv_files = [f for f in os.listdir() if f.startswith('G2F820') and f.endswith('.csv')]

# �Ыؤ@�ӪŪ� DataFrame �Ӧs�x�Ҧ��ƾ�
all_data = pd.DataFrame()

# Ū���C��csv�ɮסA�ñN��K�[�� all_data
for file in csv_files:
    # �q���W����������G�榡��2024-09-03�A����Ӧ��ɦW
    date_str = file.split('-')[1] + '-' + file.split('-')[2][:2]
    # Ū�� CSV ���A�ϥέ^��W�١A�Ĥ@�誺����W�ٶ��N����L
    df = pd.read_csv(file, skiprows=1)
    # �u�O�d�[���ɶ��M�����q�o������
    df = df[['ObsTime', 'Precp']]

    # �N�[���ɶ��ഫ������ɶ��榡�A����Ӧ��ɦW�A�ɶ�(day)�Ӧ�ObsTime
    df['Time'] = pd.to_datetime(date_str) + pd.to_timedelta(df['ObsTime'] - 1, unit='d')
    all_data = pd.concat([all_data, df], ignore_index=True)

# �R�� ObsTime �o�����
all_data = all_data.drop(columns=['ObsTime'])
# �Ƨ� Time �o�����
all_data = all_data.sort_values('Time')
# �N Time �o����첾�ʨ�Ĥ@�����
all_data.insert(0, 'Time', all_data.pop('Time'))

# �N�X�᪺֫�ƾګO�s�� CSV ���
all_data.to_csv('merged_data.csv', index=False)
