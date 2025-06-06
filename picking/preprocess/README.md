# ERT 資料處理整合程式

這個整合程式包含了原本 E1、E2、E3 三個站點的資料處理功能，並新增了陣列類型分類功能。

## 檔案結構

```
preprocess/
├── config.py                     # 配置檔案
├── array_classifier.py           # 陣列類型分類器
├── process_rhoa_integrated.py    # 主要處理程式
├── load_and_plot.py             # 資料載入和繪圖工具
├── README.md                    # 說明文件
├── plot_rhoa_E1.py             # 原始E1處理程式
├── plot_rhoa_E2.py             # 原始E2處理程式
└── plot_rhoa_E3.py             # 原始E3處理程式
```

## 主要功能

### 1. 陣列類型分類
程式會自動識別以下陣列類型：
- **Alpha**: Wenner、Schlumberger、Gradient 陣列
- **Beta**: Dipole-dipole 陣列  
- **Gamma**: 其他配置

### 2. 分別計算中位數
針對每種陣列類型分別計算：
- `median_RHOA_alpha`: Alpha 陣列的視電阻率中位數
- `median_RHOA_beta`: Beta 陣列的視電阻率中位數
- `median_RHOA_gamma`: Gamma 陣列的視電阻率中位數

### 3. 統一配置管理
所有站點的路徑和參數都集中在 `config.py` 中管理。

## 使用方法

### 配置設定

在 `config.py` 中調整各站點的路徑和參數：

```python
SITES_CONFIG = {
    'E1': {
        'urf_path': r'D:\R2MSDATA_2024\TARI_E1_test\urf',
        'ridx_urf_path': r'C:\Users\Git\masterdeg_programs\pyGIMLi\field data\TARI_monitor\E1_check\urf_E1_ridx',
        'rest': 50000,
        'output_file': 'median_RHOA_E1_and_date.pkl',
        'hydro_sheet': '農試所(霧峰)雨量資料'
    },
    # ... E2, E3 配置
}
```

### 執行資料處理

```python
# 執行主要處理程式
python process_rhoa_integrated.py

# 或者載入並分析已處理的資料
python load_and_plot.py
```

### 程式化使用

```python
from process_rhoa_integrated import ERTProcessor

# 處理單一站點
processor = ERTProcessor('E1')
processor.process_files()
processor.save_results()
processor.plot_results()

# 載入和分析資料
from load_and_plot import ERTDataLoader

loader = ERTDataLoader('E1')
loader.statistics_summary()
loader.plot_comparison()
loader.plot_time_series()
```

## 輸出資料格式

處理後的 pickle 檔案包含以下資料：
1. `dates`: 測量時間列表
2. `median_RHOA`: 整體視電阻率中位數
3. `Q1_RHOA`, `Q3_RHOA`: 整體四分位數
4. `median_RHOA_alpha`: Alpha 陣列視電阻率中位數
5. `Q1_RHOA_alpha`, `Q3_RHOA_alpha`: Alpha 陣列四分位數
6. `median_RHOA_beta`: Beta 陣列視電阻率中位數
7. `Q1_RHOA_beta`, `Q3_RHOA_beta`: Beta 陣列四分位數
8. `median_RHOA_gamma`: Gamma 陣列視電阻率中位數
9. `Q1_RHOA_gamma`, `Q3_RHOA_gamma`: Gamma 陣列四分位數

## 陣列分類邏輯

**重要更新：分類邏輯現在完全基於 PyGIMLi 的 midconfERT 函數邏輯**

### 分類標準（基於 midconfERT）

#### Alpha 陣列 (包含 Wenner 和 Schlumberger)
**判斷條件：**
1. 四個電極都有效（非NaN）
2. A和B電極在MN中點的兩側：`sign((A-MN_mid)*(B-MN_mid)) < 0`
3. 滿足以下條件之一：
   - MN電極都在AB之間：`min(M,N) > min(A,B) & max(M,N) < max(A,B)`
   - AB電極都在MN之間：`min(A,B) > min(M,N) & max(A,B) < max(M,N)`
4. AB和MN不完全分離

**細分為：**
- **Wenner**: 3×MN = AB 且 AB中點與MN中點對齊
- **Schlumberger**: 其他符合Alpha條件的配置

#### Beta 陣列 (Dipole-dipole)
**判斷條件：**
1. 四個電極都有效
2. AB和MN電極完全分離：
   - `max(A,B) < min(M,N)` 或 `max(M,N) < min(A,B)`

#### Gamma 陣列 (其他配置)
**判斷條件：**
1. 不滿足Alpha或Beta條件的所有四電極配置
2. 包括三電極配置（極-偶極、偶極-極）
3. 包括極-極配置
4. 包括交錯排列（如AMBN、ANMB等）

### midconfERT 分離代碼對應

程式同時支援生成 midconfERT 風格的分離代碼：

- **30000-39999**: Wenner-alpha 陣列
- **40000-49999**: Schlumberger 和 Gradient 陣列  
- **50000-59999**: Dipole-dipole 陣列
- **60000-69999**: Gamma 類型陣列

### 分類器驗證

可使用 `test_array_classification.py` 來驗證分類結果：

```python
python test_array_classification.py
```

這個測試程式會：
1. 使用合成配置測試基本分類功能
2. 與真實資料進行比較驗證
3. 生成分類統計和視覺化結果

### 與原始方法的差異

新的分類邏輯相較於簡單的距離判斷方法，具有以下優勢：
1. **更準確的幾何判斷**：考慮電極的相對位置關係
2. **符合標準定義**：完全遵循 PyGIMLi 的 midconfERT 邏輯
3. **處理邊界情況**：正確處理特殊排列如MABN型配置
4. **詳細分類代碼**：提供與 midconfERT 一致的分離代碼

## 繪圖功能

### 1. 比較圖 (`plot_comparison`)
- 四個子圖顯示不同陣列類型
- 包含四分位距顯示
- 可選擇是否顯示降雨量

### 2. 時間序列圖 (`plot_time_series`)
- 單一站點的完整時間序列
- 包含整體中位數和四分位距
- 雙y軸顯示降雨量

### 3. 多站點比較 (`compare_all_sites`)
- 同時比較所有站點的資料
- 分別顯示不同陣列類型

## 依賴套件

- numpy
- matplotlib
- pandas
- pygimli
- pickle
- datetime
- collections
- itertools

## 注意事項

1. 確保所有路徑在 `config.py` 中正確設定
2. 確保 PyGIMLi ToolBox 在系統路徑中
3. 處理大量資料時請確保有足夠的記憶體
4. 第一次處理時會計算幾何因子，較耗時

## 疑難排解

### 常見錯誤

1. **找不到模組**：檢查 ToolBox 路徑設定
2. **找不到檔案**：檢查 urf_path 和 ridx_urf_path
3. **記憶體不足**：考慮分批處理或增加 rest 參數值
4. **水文資料讀取失敗**：檢查 Excel 檔案路徑和工作表名稱

### 效能優化

- 調整 `rest` 參數來控制資料品質過濾的嚴格程度
- 對於大量資料，可以考慮平行處理多個站點
- 定期清理 .ohm 檔案以節省磁碟空間 