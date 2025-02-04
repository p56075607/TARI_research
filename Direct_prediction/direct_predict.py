# %%
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Microsoft Sans Serif"
# %config InlineBackend.figure_format='svg' # Setting figure format for this notebook
import numpy as np
import pygimli as pg
from pygimli.physics import ert  # the module
import pygimli.meshtools as mt
from datetime import datetime
from os.path import join
from os import listdir
from datetime import timedelta
import matplotlib.dates as mdates
import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
# %%
urf_path = r'D:\R2MSDATA\TARI_E3_test\urf'
ohmfiles = sorted([_ for _ in listdir(urf_path) if _.endswith('.ohm')])

def get_datetime_list_and_count(directory):
    # Initialize an empty list to store datetime objects
    datetime_list = []
    urffiles = sorted([_ for _ in listdir(directory) if _.endswith('.ohm')])
    # Loop through all files in the directory
    for filename in urffiles:
        # Check if the filename matches the expected format
        if len(filename) > 8 and filename[:8].isdigit():
            # Extract the date-time part from the filename
            date_time_str = filename[:8]
            # Convert the string to a datetime object
            date_time_obj = datetime.strptime(date_time_str, '%y%m%d%H')
            # Append the datetime object to the list
            datetime_list.append(date_time_obj)


    return datetime_list

dates = get_datetime_list_and_count(urf_path)

date_lim = [datetime(2024,6,8,1,0),datetime(2024,6,10,23,0)]
picked_date = [date for date in dates if date >= date_lim[0] and date <= date_lim[1]]
picked_date_index = [dates.index(date) for date in picked_date]
picked_ohmfiles = [ohmfiles[i] for i in picked_date_index]
# %%
R = []
for i,urf_file_name in enumerate(picked_ohmfiles):

    ohm_file_name = join(urf_path,urf_file_name[:-4]+'.ohm')
    data = ert.load(ohm_file_name)
    R.append(list(data['r']))

df = pd.DataFrame(R, index=pd.to_datetime(picked_date))
df.index.name = 'DateTime'
df.columns = [f'Data_{i+1}' for i in range(len(df.columns))]
print(df)
# %%
# 資料標準化
scaler = StandardScaler()
data_standardized = scaler.fit_transform(df)

# 建立 PCA 物件，設定要保留的主成分數量
pca = PCA(n_components=0.8, svd_solver='full')  # 保留 % 的變異量
principal_components = pca.fit_transform(data_standardized)

# 將主成分轉換為 DataFrame，方便後續分析
principal_df = pd.DataFrame(data=principal_components)

# %%
hydro_df = pd.read_csv(r'C:\Users\Git\TARI_research\data\external\竹塘水田.csv')

hydro_df["Timestamp"] = pd.to_datetime(hydro_df["Timestamp"])
hydro_df.set_index("Timestamp", inplace=True)
hydro_df = hydro_df.resample("h").mean()
filterd_hydro_data = pd.DataFrame()
for i in range(len(picked_ohmfiles)):
    mask = (hydro_df.index == picked_date[i])
    filterd_hydro_data = pd.concat([filterd_hydro_data, hydro_df.loc[mask]])
# %%
# 資料標準化
scaler = StandardScaler()

# 建立 PCA 物件，設定要保留的主成分數量
pca = PCA(n_components=8, svd_solver='full')  # 保留 % 的變異量
hydro_principal_components = pca.fit_transform(scaler.fit_transform(filterd_hydro_data))

# 將主成分轉換為 DataFrame，方便後續分析
hydro_principal_df = pd.DataFrame(data=hydro_principal_components)

# %%
X_pca = principal_df
Y_pca = hydro_principal_df
# 設定要提取的典型相關變量數量
n_components = 8#min(X_pca.shape[0], Y_pca.shape[0])

# 建立 CCA 物件
cca = CCA(n_components=n_components)

# 擬合模型並轉換資料
X_c, Y_c = cca.fit_transform(X_pca, Y_pca)

# X_c 和 Y_c 現在是典型相關變量
plt.plot(X_c,Y_c,'o')