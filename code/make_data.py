import cv2
import mediapipe as mp
import pandas as pd
import os

# Đường dẫn tới thư mục chứa các thư mục con với video
input_parent_folder = "./data/image/clean_data"
# Đường dẫn tới thư mục lớn để lưu các file CSV
output_parent_folder = "./data/csv_output"

# Tạo thư mục lưu trữ file CSV nếu chưa tồn tại
if not os.path.exists(output_parent_folder):
    os.makedirs(output_parent_folder)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Danh sách các chỉ số của những điểm không cần thiết
unnecessary_points = {1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32}

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if id not in unnecessary_points:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối chỉ cho các điểm cần thiết
    for connection in mpPose.POSE_CONNECTIONS:
        if connection[0] not in unnecessary_points and connection[1] not in unnecessary_points:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = results.pose_landmarks.landmark[start_idx]
            end_point = results.pose_landmarks.landmark[end_idx]
            h, w, _ = img.shape
            start_coords = (int(start_point.x * w), int(start_point.y * h))
            end_coords = (int(end_point.x * w), int(end_point.y * h))
            cv2.line(img, start_coords, end_coords, (0, 255, 0), 2)
    
    # Vẽ các điểm nút cần thiết
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if id not in unnecessary_points:
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
    return img

# Duyệt qua tất cả các thư mục con
for subdir, dirs, files in os.walk(input_parent_folder):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov')):  # Lọc các tệp video
            video_file_path = os.path.join(subdir, file)
            relative_path = os.path.relpath(subdir, input_parent_folder)
            output_subdir = os.path.join(output_parent_folder, relative_path)
            
            # Tạo thư mục con tương ứng trong thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            cap = cv2.VideoCapture(video_file_path)
            lm_list = []  # Reset landmark list for each video

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Nhận diện pose
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frameRGB)

                if results.pose_landmarks:
                    # Ghi nhận thông số khung xương
                    lm = make_landmark_timestep(results)
                    lm_list.append(lm)
                    # Vẽ khung xương lên ảnh
                    frame = draw_landmark_on_image(mpDraw, results, frame)

                cv2.imshow("image", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # Write landmark data to CSV for each video
            df = pd.DataFrame(lm_list)
            csv_file_name = os.path.splitext(file)[0] + ".csv"
            csv_file_path = os.path.join(output_subdir, csv_file_name)
            df.to_csv(csv_file_path, index=False, header=False)  # Đảm bảo rằng index và header không được lưu

            cap.release()

cv2.destroyAllWindows()
