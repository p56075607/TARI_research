import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------
# 參數設定
# ------------------------------
ALPHA_0 = 0.02   # 溫度係數 (per °C)
T0 = 25.0        # 目標校正溫度 (°C)

# 檔案路徑 (可依需求自行修改)
BASE_DIR = Path(__file__).resolve().parent
RHOA_FILE = BASE_DIR / "alpha_one_rhoa_data.csv"
TEMP_FILE = BASE_DIR / "TxSoil50cm.csv"
OUTPUT_FILE = BASE_DIR / "alpha_one_rhoa_data_corrected.csv"

# ------------------------------
# 資料讀取
# ------------------------------
print(f"讀取電阻率資料: {RHOA_FILE}")
rhoa_df = pd.read_csv(RHOA_FILE)

print(f"讀取溫度資料: {TEMP_FILE}")
# 假設溫度檔案第一欄為日期字串，第二欄為溫度值
# 時間格式為 YYYY-MM-DD 或其他，可以交由 pandas 自動解析

temp_df = pd.read_csv(TEMP_FILE)

# 將日期字串轉為 datetime 型別
rhoa_df["datetime"] = pd.to_datetime(rhoa_df["datetime"], format="%Y/%m/%d %H:%M")

# temperature csv 有可能沒有時間 (僅日期)
if "Time" in temp_df.columns:
    temp_df["Time"] = pd.to_datetime(temp_df["Time"], format="%Y-%m-%d")
    temp_df = temp_df.rename(columns={"Time": "date", temp_df.columns[1]: "temperature"})
    # 確保溫度為數值
    temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")
else:
    # 若欄位不同，嘗試自動推測
    date_col = temp_df.columns[0]
    temp_col = temp_df.columns[1]
    temp_df[date_col] = pd.to_datetime(temp_df[date_col])
    temp_df = temp_df.rename(columns={date_col: "date", temp_col: "temperature"})
    temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")

# 從電阻率 df 取得日期欄位以便合併
rhoa_df["date"] = rhoa_df["datetime"].dt.floor("D")

# 左合併以保留所有電阻率紀錄
merged = rhoa_df.merge(temp_df, on="date", how="left")

if merged["temperature"].isna().any():
    missing_cnt = merged["temperature"].isna().sum()
    print(f"警告: 有 {missing_cnt} 筆溫度資料缺失，將以線性內插補值。")
    merged["temperature"] = merged["temperature"].interpolate(limit_direction="both")

# ------------------------------
# 進行溫度修正
# ------------------------------
# 把欲校正的欄位 (除了 datetime/date/temperature) 依序取出
value_cols = [col for col in merged.columns if col not in ("datetime", "date", "temperature")]

# 再確保 merged 中溫度為浮點數
merged["temperature"] = pd.to_numeric(merged["temperature"], errors="coerce")
# 計算修正係數: 1 + alpha0 * (T - T0)
correction_factor = 1 + ALPHA_0 * (merged["temperature"] - T0)

# 對所有電阻率欄位進行修正
for col in value_cols:
    merged[col] = merged[col] / correction_factor

# 移除輔助欄位並輸出
output_cols = ["datetime"] + value_cols
corrected_df = merged[output_cols].copy()
corrected_df.to_csv(OUTPUT_FILE, index=False)

# ------------------------------
# 畫圖：修正前 vs 修正後
# ------------------------------
import matplotlib.pyplot as plt
# Font：Microsoft JhengHei
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
for col in value_cols:
    plt.figure(figsize=(10,4))
    plt.plot(rhoa_df["datetime"], rhoa_df[col], label="原始", alpha=0.7)
    plt.plot(corrected_df["datetime"], corrected_df[col], label="修正後", alpha=0.7)
    plt.title(f"{col} 溫度修正前後對照")
    plt.xlabel("Datetime")
    plt.ylabel("Resistivity")
    plt.legend()
    plt.tight_layout()
    # 可選：將圖檔輸出
    fig_path = BASE_DIR / f"{col}_temp_correct_compare.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

print("溫度修正完成！已輸出至:")
print(f"  {OUTPUT_FILE}\n")
print("比較圖已輸出至相同資料夾")
