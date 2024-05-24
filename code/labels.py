import os

# Đường dẫn tới thư mục chính
parent_directory = "./data/image/clean_data"  # Thay thế bằng đường dẫn của bạn

# Lấy danh sách các thư mục con
subdirectories = [name for name in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, name))]

# In ra danh sách các thư mục con
print("Danh sách các thư mục con trong", parent_directory, ":")
print(subdirectories)

# subdirectories giờ đã là một list chứa tên các thư mục con
