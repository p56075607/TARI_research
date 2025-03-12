# %%
import sys
sys.path.append(r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox')
from ohm2urf import ohm2urf
from os.path import join
from os import listdir

ohm_path = 'syn_output'
ohmfiles = [_ for _ in listdir(ohm_path) if _.endswith('.ohm')]
for i,ohm_file_name in enumerate(ohmfiles):
    ohm_file_path = join(ohm_path, ohm_file_name)
    urf_file_path = ohm_file_path[:-4]+'.urf'
    ohm2urf(ohm_file_path, urf_file_path)