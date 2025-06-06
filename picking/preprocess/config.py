# Configuration file for ERT data processing
import os

# Site configurations
SITES_CONFIG = {
    'E1': {
        'urf_path': r'D:\R2MSDATA_2024\TARI_E1_test\urf',
        'ridx_urf_path': r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor\E1_check\urf_E1_ridx',
        'rest': 50000,
        'output_file': 'median_RHOA_E1_and_date.pkl',
        'hydro_sheet': '農試所(霧峰)雨量資料'
    },
    'E2': {
        'urf_path': r'D:\R2MSDATA\TARI_E2_test\urf',
        'ridx_urf_path': r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor\E2_check\urf_E2_ridx',
        'rest': 100000,
        'output_file': 'median_RHOA_E2_and_date.pkl',
        'hydro_sheet': '農試所(霧峰)雨量資料'
    },
    'E3': {
        'urf_path': r'D:\R2MSDATA\TARI_E3_test\urf',
        'ridx_urf_path': r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor\E3_check\urf_E3_ridx',
        'rest': 100000,
        'output_file': 'median_RHOA_E3_and_date.pkl',
        'hydro_sheet': '彰化竹塘水田'
    }
}

# Common paths
TOOLBOX_PATH = r'C:\Users\Git\masterdeg_programs\pyGIMLi\ToolBox'
HYDRO_DATA_PATH = r'水文站資料彙整_20240731.xlsx'

# Processing parameters
FORMULA_CHOOSE = 'C' 