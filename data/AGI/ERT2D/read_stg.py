# %%
import pygimli as pg
from pygimli.physics import ert
import numpy as np

stg_fname = '1_49_DD(22Min)_trial1.stg'
data = ert.load(stg_fname)
data['k'] = ert.createGeometricFactors(data, numerical=True)
data['r'] = data['u'] / data['i']
data['rhoa'] = data['r'] * data['k']
data['err'] = ert.estimateError(data, relativeError=0.02)
data.save(stg_fname[:-4]+'.ohm')
# %%
ax,_ = ert.showERTData(data)
ax.plot(pg.x(data),pg.y(data), 'kv')
ax.set_xlim([min(pg.x(data)), max(pg.x(data))])

# %%
config = np.array([data['a'],data['b'],data['m'],data['n']])
from collections import defaultdict
import matplotlib.pyplot as plt

def find_duplicate_columns(array):
    column_dict = defaultdict(list)
    for i in range(array.shape[1]):
        col_tuple = tuple(array[:, i])
        column_dict[col_tuple].append(i)
    
    duplicates = {key: value for key, value in column_dict.items() if len(value) > 1}
    
    return duplicates

duplicates = find_duplicate_columns(config)
# 繪製重複 column 的索引序列
if duplicates:
    # 將所有重複的索引展平為一個列表
    duplicate_indices = [index for indices in duplicates.values() for index in indices]

    # 繪製條形圖
    plt.figure(figsize=(12, 6))
    plt.hist(duplicate_indices, bins=len(set(duplicate_indices)), edgecolor='black')
    plt.title('Histogram of Duplicate Column Indices')
    plt.xlabel('Column Index')
    plt.ylabel('Frequency')

else:
    print("All columns are unique.")
# %%
mgr = ert.ERTManager(data)
mgr.invert(verbose=True)
mgr.showResultAndFit()