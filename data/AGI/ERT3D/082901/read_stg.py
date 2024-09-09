import pandas as pd

def read_and_process_supersting_data(file_path):
    # �w�q�C���D
    column_names = [
        "Record Number", "USER", "Date", "Time", "V/I", "Error (%)",
        "Output Current (mA)", "Apparent Resistivity (?m or ?ft)", "Command File ID",
        "A-electrode X", "A-electrode Y", "A-electrode Z",
        "B-electrode X", "B-electrode Y", "B-electrode Z",
        "M-electrode X", "M-electrode Y", "M-electrode Z",
        "N-electrode X", "N-electrode Y", "N-electrode Z",
        "Cmd", "HV", "Cyk", "MTime", "Gain", "Ch"
    ]
    
    # ��l�ƼƾڦC��
    data = []

    with open(file_path, 'r') as file:
        # �O�s�Y���H��
        header = [next(file) for _ in range(3)]

        # �v��Ū���ƾ�
        for line in file:
            # �����e��ťըåH�r�����j���
            fields = line.strip().split(',')

            # �N�q����m�ƾ��ഫ���B�I��
            parsed_data = fields[:21] + [f.split('=')[1] for f in fields[21:]]
            data.append(parsed_data)

    # �Ы�DataFrame
    df = pd.DataFrame(data, columns=column_names)
    
    # �ഫ���n���ƾ�����
    df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
    df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S').dt.time
    df["V/I"] = pd.to_numeric(df["V/I"])
    df["Error (%)"] = pd.to_numeric(df["Error (%)"])
    df["Output Current (mA)"] = pd.to_numeric(df["Output Current (mA)"])
    df["Apparent Resistivity (?m or ?ft)"] = pd.to_numeric(df["Apparent Resistivity (?m or ?ft)"])
    df.iloc[:, 9:21] = df.iloc[:, 9:21].apply(pd.to_numeric)

    # �ھڱ���վ�A�MM�q����X�MY�y��
    df["A-electrode X"] = df["A-electrode X"].apply(lambda x: x - 11 if x > 10.5 else x - 3)
    df["M-electrode X"] = df["M-electrode X"].apply(lambda x: x - 11 if x > 10.5 else x - 3)
    df["A-electrode Y"] = df["A-electrode Y"] - 13
    df["M-electrode Y"] = df["M-electrode Y"] - 13

    return header, df

def save_processed_data_to_stg(header, df, output_file_path):
    with open(output_file_path, 'w') as file:
        # �g�J�Y���H��
        file.writelines(header)
        
        # �g�J�ƾ�
        for index, row in df.iterrows():
            line = f"{int(row['Record Number'])},{row['USER']},{row['Date'].strftime('%Y%m%d')},{row['Time'].strftime('%H:%M:%S')},{row['V/I']: .5E},{int(row['Error (%)'])},{int(row['Output Current (mA)'])},{row['Apparent Resistivity (?m or ?ft)']: .5E},{row['Command File ID']}, {row['A-electrode X']: .5E},{row['A-electrode Y']: .5E},{row['A-electrode Z']: .5E},{row['B-electrode X']: .5E},{row['B-electrode Y']: .5E},{row['B-electrode Z']: .5E},{row['M-electrode X']: .5E},{row['M-electrode Y']: .5E},{row['M-electrode Z']: .5E},{row['N-electrode X']: .5E},{row['N-electrode Y']: .5E},{row['N-electrode Z']: .5E},Cmd={int(row['Cmd'])},HV={int(row['HV'])},Cyk={int(row['Cyk'])},MTime={float(row['MTime'])},Gain={int(row['Gain'])},Ch={int(row['Ch'])}\n"
            file.write(line)

# �ϥΨ禡Ū���M�B�z�ƾ�
file_path = '082901.stg'
output_file_path = '082901_m.stg'

header, processed_data_frame = read_and_process_supersting_data(file_path)

# �O�s�B�z�᪺�ƾ�
save_processed_data_to_stg(header, processed_data_frame, output_file_path)
