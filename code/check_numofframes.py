import cv2
import os

# Đường dẫn tới thư mục chứa video
video_folder = "./data/image/clean_data/test"

# Đọc các video từ thư mục
video_files = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]

for video_file in video_files:
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
    
    # Kiểm tra FPS của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_file} - FPS: {fps}")

    cap.release()
