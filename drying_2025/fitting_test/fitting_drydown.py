import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 一般模型（全參數自由）
def rho_model(t, rho0, delta_theta, tau, theta_w, n):
    theta = delta_theta * np.exp(-t / tau) + theta_w
    return rho0 * (1 / theta) ** n

# 主擬合函數（可固定任意參數）
def fit_resistivity_model(file_path, label="data", fix_params={}, enforce_initial=True):
    """擬合乾燥電阻率模型

    Parameters
    ----------
    file_path : str
        資料檔路徑，可為 *.txt (預設以 TAB 分隔且無標題) 或一般 *.csv (自帶標題)。
    label : str, optional
        繪圖及輸出用之標籤。
    fix_params : dict, optional
        指定欲固定之參數，例如 {"tau": 500} 代表固定 \tau 不參與優化。
    enforce_initial : bool, optional
        若為 True，則強制滿足 \rho(0)=\rho_0(\Delta\theta+\theta_w)^{-n}，其中 \rho(0)
        取自輸入資料 t=0 (若無則取第一筆)。此時 \rho_0 不再視為獨立參數。
    """

    # 根據副檔名自動判斷分隔符號與標題
    if file_path.lower().endswith(".txt"):
        df = pd.read_csv(file_path, sep="\t", header=None, names=["t", "rho"])
    else:
        df = pd.read_csv(file_path)  # 假設欄位已包含時間與電阻率兩欄
        # 嘗試對應常見欄位名稱
        if df.columns.size >= 2:
            df = df.rename(columns={df.columns[0]: "t", df.columns[1]: "rho"})

    t = df["t"].values
    rho_obs = df["rho"].values

    # 觀測起始電阻率 (t == 0 若不存在則取第一筆)
    idx_zero = np.where(t == 0)[0]
    rho0_obs = rho_obs[idx_zero[0]] if idx_zero.size else rho_obs[0]

    # 所有參數與預設值
    all_param_names = ['rho0', 'delta_theta', 'tau', 'theta_w', 'n']
    default_p0 = {'rho0': 50, 'delta_theta': 0.1, 'tau': 300, 'theta_w': 0.1, 'n': 2.0}
    lower_bounds = {'rho0': 0, 'delta_theta': 0, 'tau': 1, 'theta_w': 0, 'n': 1}
    upper_bounds = {'rho0': 500, 'delta_theta': 0.4, 'tau': 10000, 'theta_w': 0.2, 'n': 3}

    # 依 enforce_initial 決定是否將 rho0 納入優化
    if enforce_initial and 'rho0' in fix_params:
        raise ValueError("當 enforce_initial=True 時，不可於 fix_params 中指定 'rho0'。")

    base_param_names = ['delta_theta', 'tau', 'theta_w', 'n'] if enforce_initial else all_param_names

    # 建立要優化的參數清單
    fit_param_names = [k for k in base_param_names if k not in fix_params]
    p0 = [default_p0[k] for k in fit_param_names]
    bounds = (
        [lower_bounds[k] for k in fit_param_names],
        [upper_bounds[k] for k in fit_param_names]
    )

    # 定義新的函數：將要優化的變數與固定變數合併
    def model(t, *args):
        param_dict = fix_params.copy()
        param_dict.update({k: v for k, v in zip(fit_param_names, args)})

        # 若需要，利用初始條件計算 rho0
        if enforce_initial:
            param_dict['rho0'] = rho0_obs * (param_dict['delta_theta'] + param_dict['theta_w']) ** param_dict['n']

        theta = param_dict["delta_theta"] * np.exp(-t / param_dict["tau"]) + param_dict["theta_w"]
        return param_dict["rho0"] * (1 / theta) ** param_dict["n"]

    # 擬合
    popt, _ = curve_fit(model, t, rho_obs, p0=p0, bounds=bounds)
    fitted_rho = model(t, *popt)

    # 計算 R²
    ss_res = np.sum((rho_obs - fitted_rho) ** 2)
    ss_tot = np.sum((rho_obs - np.mean(rho_obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # 繪圖
    plt.figure(figsize=(8, 4))
    plt.plot(t, rho_obs, 'ko', markersize=3, label='Observed')
    plt.plot(t, fitted_rho, 'b-', label='Fitted')
    plt.title(f"{label} Fit (R² = {r2:.3f})")
    plt.xlabel("Time (hr)")
    plt.ylabel("Resistivity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 整理輸出結果
    final_params = fix_params.copy()
    final_params.update({k: v for k, v in zip(fit_param_names, popt)})

    # 若有強制初始條件，需同步輸出 rho0
    if enforce_initial:
        final_params['rho0'] = rho0_obs * (final_params['delta_theta'] + final_params['theta_w']) ** final_params['n']

    final_params["R²"] = r2

    return final_params

params = fit_resistivity_model(
    r"C:\Users\Git\TARI_research\drying_2025\alpha_one_by_column\dryingtime_alpha_one_RHOA_13_E1.csv",
    label="alpha_one_RHOA_1_E1",
    # fix_params={"tau": 447},
    # enforce_initial=True   # 預設就是 True，可省略
)
print(params)
