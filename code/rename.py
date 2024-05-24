import os

# Đường dẫn tới thư mục chứa các thư mục con
parent_folder = "./data/image/clean_data"

# Hàm tạo tên tệp duy nhất
def get_unique_filename(directory, base_name, extension):
    counter = 0
    new_file_name = f"{base_name}_{counter}{extension}"
    new_file_path = os.path.join(directory, new_file_name)
    while os.path.exists(new_file_path):
        counter += 1
        new_file_name = f"{base_name}_{counter}{extension}"
        new_file_path = os.path.join(directory, new_file_name)
    return new_file_path

# Duyệt qua tất cả các thư mục con
for subdir, dirs, files in os.walk(parent_folder):
    for idx, file in enumerate(files):
        file_path = os.path.join(subdir, file)
        if os.path.isfile(file_path):
            # Lấy tên thư mục con
            subdir_name = os.path.basename(subdir)
            # Lấy phần mở rộng của tệp
            file_extension = os.path.splitext(file)[1]
            # Tạo tên mới cho file
            new_file_path = get_unique_filename(subdir, subdir_name, file_extension)
            # Đổi tên file
            os.rename(file_path, new_file_path)

print("File renaming completed.")
