# %%
import pygimli as pg
from pygimli.physics import ert
import numpy as np
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from ohm2urf import ohm2urf
from os.path import join
#%%
stg_ph = r"C:\Users\Git\TARI_research\data\AGI\ERT2D\110802"
stg_fname = join(stg_ph,stg_ph[-6:]+".stg")#r'C:\Users\Git\TARI_research\data\AGI\ERT3D\1023\10233D.stg'#r'1_112_DDWSEG(144Min)\trial1\1_112_DDWSEG(144Min)_trial1.stg'
data = ert.load(stg_fname)
print(data)
if stg_ph[-1] == '1':
    time_str = '24'+stg_ph[-6:-1]+'9'
elif stg_ph[-1] == '2':
    time_str = '24'+stg_ph[-6:-2]+'13'
# %%
data['k'] = ert.createGeometricFactors(data, numerical=True)
data['r'] = data['u'] / data['i']
data['rhoa'] = data['r'] * data['k']
# data['err'] = ert.estimateError(data, relativeError=0.02)
data.save(stg_fname[:-4]+'.ohm')
# %%
ax,_ = ert.showERTData(data)
ax.plot(pg.x(data),pg.y(data), 'kv')
ax.set_xlim([min(pg.x(data)), max(pg.x(data))])
# %%
import matplotlib.pyplot as plt
import os
import os
line_1 = [] # 1-28
line_2 = [] # 29-49
line_3 = [] # 50-70
line_4 = [] # 71-91
line_5 = [] # 92-112

ABMN = np.array([data['a'],data['b'],data['m'],data['n']]).T
for i in range(len(ABMN)):
    if np.all(ABMN[i] < 28):
        line_1.append(i)
    elif np.all(ABMN[i] > 27) and np.all(ABMN[i] < 49):
        line_2.append(i)
    elif np.all(ABMN[i] > 48) and np.all(ABMN[i] < 70):
        line_3.append(i)
    elif np.all(ABMN[i] > 69) and np.all(ABMN[i] < 91):
        line_4.append(i)
    elif np.all(ABMN[i] > 90) and np.all(ABMN[i] < 113):
        line_5.append(i)


def seperate_line(data, line):
    boolean_array = np.ones(ABMN.shape[0], dtype=bool)
    boolean_array[line] = False
    data_line = data.copy()
    data_line.remove(boolean_array)
    data_line.removeUnusedSensors()
    print(data_line)
    # fig, ax = plt.subplots()
    # ax.plot(np.array(pg.x(data_line)), np.array(pg.z(data_line)),'kv',label='Current/Potential electrode ($C_i/P_i$)')
    return data_line

seperate_ph = join(stg_ph, 'seperate')
if not os.path.exists(seperate_ph):
    os.makedirs(seperate_ph)
data_line_1 = seperate_line(data, line_1)
ohm_name = join(seperate_ph,time_str+'L1_1_28'+'.ohm')
data_line_1.save(ohm_name)
ohm2urf(ohm_name, ohm_name[:-4]+'.urf')

data_line_2 = seperate_line(data, line_2)
ohm_name = join(seperate_ph, time_str+'L2_29_49'+'.ohm')
data_line_2.save(ohm_name)
ohm2urf(ohm_name, ohm_name[:-4]+'.urf')

data_line_3 = seperate_line(data, line_3)
ohm_name = join(seperate_ph, time_str+'L3_50_70'+'.ohm')
data_line_3.save(ohm_name)
ohm2urf(ohm_name, ohm_name[:-4]+'.urf')

data_line_4 = seperate_line(data, line_4)
ohm_name = join(seperate_ph, time_str+'L4_71_91'+'.ohm')
data_line_4.save(ohm_name)
ohm2urf(ohm_name, ohm_name[:-4]+'.urf')

data_line_5 = seperate_line(data, line_5)
ohm_name = join(seperate_ph, time_str+'L5_92_112'+'.ohm')
data_line_5.save(ohm_name)
ohm2urf(ohm_name, ohm_name[:-4]+'.urf')


# %%
# config = np.array([data['a'],data['b'],data['m'],data['n']])
# from collections import defaultdict
# import matplotlib.pyplot as plt

# def find_duplicate_columns(array):
#     column_dict = defaultdict(list)
#     for i in range(array.shape[1]):
#         col_tuple = tuple(array[:, i])
#         column_dict[col_tuple].append(i)
    
#     duplicates = {key: value for key, value in column_dict.items() if len(value) > 1}
    
#     return duplicates

# duplicates = find_duplicate_columns(config)
# # 繪製重複 column 的索引序列
# if duplicates:
#     # 將所有重複的索引展平為一個列表
#     duplicate_indices = [index for indices in duplicates.values() for index in indices]

#     # 繪製條形圖
#     plt.figure(figsize=(12, 6))
#     plt.hist(duplicate_indices, bins=len(set(duplicate_indices)), edgecolor='black')
#     plt.title('Histogram of Duplicate Column Indices')
#     plt.xlabel('Column Index')
#     plt.ylabel('Frequency')

# else:
#     print("All columns are unique.")
# # %%
# mgr = ert.ERTManager(data)
# mgr.invert(verbose=True)
# mgr.showResultAndFit()