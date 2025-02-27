# %%
import pickle
from os.path import isdir, join
import pandas as pd
with open(join(r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor','median_RHOA_E2_and_date.pkl'), 'rb') as f:
    dates_E1 = pickle.load(f)
    median_RHOA_E1 = pickle.load(f)
    Q1_RHOA_E1 = pickle.load(f)
    Q3_RHOA_E1 = pickle.load(f)

df_rhoa = pd.DataFrame({'date': dates_E1, 'median_RHOA': median_RHOA_E1, 'Q1_RHOA': Q1_RHOA_E1, 'Q3_RHOA': Q3_RHOA_E1})
df_rhoa['date'] = df_rhoa['date'].dt.strftime('%Y/%m/%d %H:%M')
# output time formate YYYY/MM/DD HH:MM
df_rhoa.to_csv(r'C:\Users\Git\TARI_research\picking\E2_rhoa.csv', index=False)
# %%
