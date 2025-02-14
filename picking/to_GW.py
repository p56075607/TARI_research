# %%
import pandas as pd
df = pd.read_csv(r'C:\Users\Git\TARI_research\picking\E3.csv')
# delete data i%3 == 0
df = df[df.index%3 != 0]
df.to_csv(r'C:\Users\Git\TARI_research\picking\E3_window.csv', index=False)
# %%
import pickle
from os.path import isdir, join
with open(join(r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor','median_RHOA_E3_and_date.pkl'), 'rb') as f:
    pickled_dates_E1 = pickle.load(f)
    pickled_median_RHOA_E1 = pickle.load(f)

df_rhoa = pd.DataFrame({'date': pickled_dates_E1, 'median_RHOA': pickled_median_RHOA_E1})
df_rhoa['date'] = df_rhoa['date'].dt.strftime('%Y/%m/%d %H:%M')
# output time formate YYYY/MM/DD HH:MM
df_rhoa.to_csv(r'C:\Users\Git\TARI_research\picking\E3_rhoa.csv', index=False)