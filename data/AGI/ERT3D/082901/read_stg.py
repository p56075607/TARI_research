import pandas as pd

def read_and_process_supersting_data(file_path):
    # 定義列標題
    column_names = [
        "Record Number", "USER", "Date", "Time", "V/I", "Error (%)",
        "Output Current (mA)", "Apparent Resistivity (?m or ?ft)", "Command File ID",
        "A-electrode X", "A-electrode Y", "A-electrode Z",
        "B-electrode X", "B-electrode Y", "B-electrode Z",
        "M-electrode X", "M-electrode Y", "M-electrode Z",
        "N-electrode X", "N-electrode Y", "N-electrode Z",
        "Cmd", "HV", "Cyk", "MTime", "Gain", "Ch"
    ]
    
    # 初始化數據列表
    data = []

    with open(file_path, 'r') as file:
        # 保存頭文件信息
        header = [next(file) for _ in range(3)]

        # 逐行讀取數據
        for line in file:
            # 移除前後空白並以逗號分隔欄位
            fields = line.strip().split(',')

            # 將電極位置數據轉換為浮點數
            parsed_data = fields[:21] + [f.split('=')[1] for f in fields[21:]]
            data.append(parsed_data)

    # 創建DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    # 轉換必要的數據類型
    df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
    df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S').dt.time
    df["V/I"] = pd.to_numeric(df["V/I"])
    df["Error (%)"] = pd.to_numeric(df["Error (%)"])
    df["Output Current (mA)"] = pd.to_numeric(df["Output Current (mA)"])
    df["Apparent Resistivity (?m or ?ft)"] = pd.to_numeric(df["Apparent Resistivity (?m or ?ft)"])
    df.iloc[:, 9:21] = df.iloc[:, 9:21].apply(pd.to_numeric)

    # 根據條件調整A和M電極的X和Y座標
    df["A-electrode X"] = df["A-electrode X"].apply(lambda x: x - 11 if x > 10.5 else x - 3)
    df["M-electrode X"] = df["M-electrode X"].apply(lambda x: x - 11 if x > 10.5 else x - 3)
    df["A-electrode Y"] = df["A-electrode Y"] - 13
    df["M-electrode Y"] = df["M-electrode Y"] - 13

    return header, df

def save_processed_data_to_stg(header, df, output_file_path):
    with open(output_file_path, 'w') as file:
        # 寫入頭文件信息
        file.writelines(header)
        
        # 寫入數據
        for index, row in df.iterrows():
            line = f"{int(row['Record Number'])},{row['USER']},{row['Date'].strftime('%Y%m%d')},{row['Time'].strftime('%H:%M:%S')},{row['V/I']: .5E},{int(row['Error (%)'])},{int(row['Output Current (mA)'])},{row['Apparent Resistivity (?m or ?ft)']: .5E},{row['Command File ID']}, {row['A-electrode X']: .5E},{row['A-electrode Y']: .5E},{row['A-electrode Z']: .5E},{row['B-electrode X']: .5E},{row['B-electrode Y']: .5E},{row['B-electrode Z']: .5E},{row['M-electrode X']: .5E},{row['M-electrode Y']: .5E},{row['M-electrode Z']: .5E},{row['N-electrode X']: .5E},{row['N-electrode Y']: .5E},{row['N-electrode Z']: .5E},Cmd={int(row['Cmd'])},HV={int(row['HV'])},Cyk={int(row['Cyk'])},MTime={float(row['MTime'])},Gain={int(row['Gain'])},Ch={int(row['Ch'])}\n"
            file.write(line)

# 使用函式讀取和處理數據
file_path = '082901.stg'
output_file_path = '082901_m.stg'

header, processed_data_frame = read_and_process_supersting_data(file_path)

# 保存處理後的數據
save_processed_data_to_stg(header, processed_data_frame, output_file_path)
