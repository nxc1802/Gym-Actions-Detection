import os
import pandas as pd

# Đường dẫn tới thư mục lớn chứa các thư mục con chứa các file CSV
parent_folder = "./data/csv_output"

# Đường dẫn tới thư mục lớn chứa các thư mục con lưu kết quả
output_parent_folder = "./data/train_data"

# Duyệt qua tất cả các thư mục con trong thư mục lớn
for subdir, dirs, files in os.walk(parent_folder):
    for csv_file in files:
        if csv_file.endswith('.csv'):
            # Đường dẫn đầy đủ đến file CSV
            csv_file_path = os.path.join(subdir, csv_file)

            # Đọc file CSV
            df = pd.read_csv(csv_file_path, header=None)
            
            # Khởi tạo danh sách để lưu các DataFrame con
            df_list = []

            # Duyệt qua các dòng của DataFrame và lấy các nhóm dòng cần thiết
            for start in range(0, len(df), 20):
                # Lấy 10 dòng từ vị trí start
                sub_df = df.iloc[start:start + 60]
                if len(sub_df) == 60:
                    df_list.append(sub_df)
            
            # Tạo thư mục con trong thư mục đích nếu chưa tồn tại
            output_subfolder = os.path.join(output_parent_folder, os.path.basename(subdir))
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # Ghi từng DataFrame con ra file CSV riêng biệt trong thư mục con của thư mục lớn
            for i, sub_df in enumerate(df_list):
                output_csv_file = f"{os.path.splitext(csv_file)[0]}_part_{i+1}.csv"
                output_csv_path = os.path.join(output_subfolder, output_csv_file)
                sub_df.to_csv(output_csv_path, index=False, header=False)

print("Processing completed.")
