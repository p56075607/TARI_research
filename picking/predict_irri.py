# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
plt.rcParams['font.family'] = 'Microsoft YaHei'
import matplotlib
def calculate_irrigation_water(slope, intercept, sat_theta, irrig_theta, a, b, Time):
    """
    計算灌溉過程中的時間和電阻率數據。

    參數:
    slope (float): 斜率
    intercept (float): 截距
    sat_theta (float): 飽和含水量
    irrig_theta (float): 灌溉門檻含水量
    a (float): 參數 a
    b (float): 參數 b
    Time (float): 總時間

    返回:
    total_t (numpy.ndarray): 總時間數據
    total_rhos (numpy.ndarray): 總電阻率數據
    """
    irrig_rhoa = np.log10(a * np.log(irrig_theta) + b)
    rhoa_min = np.log10(100)#np.log10(a * np.log(sat_theta) + b)
    sat_theta = np.exp(((10**rhoa_min)-b)/a)
    rho = np.log10(a * np.log(80) + b)
    t = 0
    rhos = []
    ts = []
    while rho < irrig_rhoa:
        K = (slope * rhoa_min + intercept)
        rho = (K * np.exp(slope * t) - intercept) / slope
        rhos.append(rho)
        ts.append(t)
        t += 1
    irrig_times = Time / t
    total_water_usage = (Time / t) * (sat_theta - irrig_theta) / 100
    print('灌溉次數:', irrig_times, ', 總用水量:', total_water_usage)

    total_t = []
    total_rhos = []

    # 迭代灌溉次數
    for i in range(int(np.ceil(irrig_times))):
        # 計算當前時間段的時間和電阻率，並添加到總列表中
        current_ts = np.array(ts) + i * t
        current_rhos = rhos

        # 過濾超過總時間的部分
        valid_indices = current_ts <= Time
        total_t.extend(current_ts[valid_indices])
        total_rhos.extend(np.array(current_rhos)[valid_indices])

    # 將列表轉換為NumPy陣列
    total_t = np.array(total_t)
    total_rhos = np.array(total_rhos)

    return total_t, total_rhos, irrig_times, total_water_usage
# %%
# 創建圖表
fig, ax = plt.subplots(figsize=(15, 5))

# 計算第一組數據
total_t1, total_rhos1, irrig_times1, total_water_usage1 = calculate_irrigation_water(
    slope=-0.0021842301349774863,
    intercept=0.004869288745739898,
    sat_theta=45,  # 飽和含水量
    irrig_theta=30,  # 灌溉門檻含水量
    a=-28.05, b=226.22,
    Time=100 * 24
)

# 繪製第一條曲線，並在圖例中標示灌溉次數和總用水量
ax.plot(total_t1, 10 ** total_rhos1, label=f'灌溉門檻含水量 30%\n灌溉次數: {irrig_times1:.0f}次, 總用水量: {total_water_usage1:.1f}$m^3$')

# 計算第二組數據
total_t2, total_rhos2, irrig_times2, total_water_usage2 = calculate_irrigation_water(
    slope=-0.0021842301349774863,
    intercept=0.004869288745739898,
    sat_theta=45,  # 飽和含水量
    irrig_theta=15,  # 灌溉門檻含水量
    a=-28.05, b=226.22,
    Time=100 * 24
)

# 繪製第二條曲線，並在圖例中標示灌溉次數和總用水量
ax.plot(total_t2, 10 ** total_rhos2, label=f'灌溉門檻含水量 15%\n灌溉次數: {irrig_times2:.0f}次, 總用水量: {total_water_usage2:.1f}$m^3$')

# 設定 x 軸刻度間隔為每 7 天（168 小時）
ax.xaxis.set_major_locator(MultipleLocator(7 * 24))

# 定義刻度標籤格式化函數，以天數顯示
def hours_to_days(x, pos):
    return f'{int(x // 24)} 天'

ax.xaxis.set_major_formatter(FuncFormatter(hours_to_days))

# 設定 x 軸範圍以確保不超過指定的 Time
ax.set_xlim([0, 100 * 24])



ax.grid(True, which='major', linestyle='--', linewidth=0.5)
fontsize = 20
ax.set_ylabel('電阻率 ($\Omega-m$)', fontsize=fontsize+5,fontweight='bold')
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
# set xy ticks label fontsize 
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
# ax.set_xlabel('Time', fontsize=fontsize, fontweight='bold')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)
plt.yticks(fontsize=fontsize,fontweight='bold')
plt.xticks(fontsize=fontsize-2,fontweight='bold',rotation=45, ha='center')
# 添加圖例，將其放置在圖表外部
font = matplotlib.font_manager.FontProperties(size=fontsize-5, weight='bold')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',prop=font)

# 調整子圖佈局以適應圖例
plt.subplots_adjust(right=0.8)
# 顯示圖表
plt.show()

# %%
# 創建圖表
fig, ax = plt.subplots(figsize=(15, 5))

# 計算第一組數據
total_t1, total_rhos1, irrig_times1, total_water_usage1 = calculate_irrigation_water(
    slope=-0.0025465262282503166,
    intercept=0.0039892514401080985,
    sat_theta=45,  # 飽和含水量
    irrig_theta=30,  # 灌溉門檻含水量
    a=-4.71, b=45.68,
    Time=100 * 24
)

# 繪製第一條曲線，並在圖例中標示灌溉次數和總用水量
ax.plot(total_t1, 10 ** total_rhos1, label=f'灌溉門檻含水量 30%\n灌溉次數: {irrig_times1:.0f}次, 總用水量: {total_water_usage1:.1f}$m^3$')

# 計算第二組數據
total_t2, total_rhos2, irrig_times2, total_water_usage2 = calculate_irrigation_water(
    slope=-0.0025465262282503166,
    intercept=0.0039892514401080985,
    sat_theta=45,  # 飽和含水量
    irrig_theta=15,  # 灌溉門檻含水量
    a=-4.71, b=45.68,
    Time=100 * 24
)

# 繪製第二條曲線，並在圖例中標示灌溉次數和總用水量
ax.plot(total_t2, 10 ** total_rhos2, label=f'灌溉門檻含水量 15%\n灌溉次數: {irrig_times2:.0f}次, 總用水量: {total_water_usage2:.1f}$m^3$')

# 設定 x 軸刻度間隔為每 7 天（168 小時）
ax.xaxis.set_major_locator(MultipleLocator(7 * 24))

# 定義刻度標籤格式化函數，以天數顯示
def hours_to_days(x, pos):
    return f'{int(x // 24)} 天'

ax.xaxis.set_major_formatter(FuncFormatter(hours_to_days))

# 設定 x 軸範圍以確保不超過指定的 Time
ax.set_xlim([0, 100 * 24])



ax.grid(True, which='major', linestyle='--', linewidth=0.5)
fontsize = 20
ax.set_ylabel('電阻率 ($\Omega-m$)', fontsize=fontsize+5,fontweight='bold')
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
# set xy ticks label fontsize 
ax.tick_params(axis='both', which='major', length=10,width=3, direction='in')
ax.tick_params(axis='both', which='minor', length=5,width=1.5, direction='in')
# ax.set_xlabel('Time', fontsize=fontsize, fontweight='bold')
width = 3
ax.spines['top'].set_linewidth(width)
ax.spines['right'].set_linewidth(width)
ax.spines['bottom'].set_linewidth(width)
ax.spines['left'].set_linewidth(width)

plt.yticks(fontsize=fontsize,fontweight='bold')
plt.xticks(fontsize=fontsize-2,fontweight='bold',rotation=45, ha='center')
# 添加圖例，將其放置在圖表外部
font = matplotlib.font_manager.FontProperties(size=fontsize-5, weight='bold')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',prop=font)

# 調整子圖佈局以適應圖例
plt.subplots_adjust(right=0.8)
# 顯示圖表