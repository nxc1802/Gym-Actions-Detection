import cv2
import numpy as np
import os
import mediapipe as mp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Đọc mô hình đã lưu
model = load_model("model.h5")

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Danh sách các chỉ số của những điểm không cần thiết
unnecessary_points = {1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32}

# Labels
actions = ['barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift', 'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press', 'lat pulldown', 'lateral raise', 'leg extension', 'leg raises', 'plank', 'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 'shoulder press', 'squat', 't bar row', 'tricep dips', 'tricep Pushdown']

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

# Hàm dự đoán hành động từ video
def predict_action(model, pose, mpDraw, unnecessary_points, actions, video_path):
    cap = cv2.VideoCapture(video_path)
    lm_list = []
    frame_count = 0
    number_of_prediction = 0
    label_counts = {action: 0 for action in actions}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            frame = draw_landmark_on_image(mpDraw, results, frame)
            frame_count += 1

            if frame_count % 15 == 0:
                prediction = model.predict(np.expand_dims(np.array(lm_list), axis=0))
                action_idx = np.argmax(prediction)
                action = actions[action_idx]
                label_counts[action] += 1
                lm_list = []
                number_of_prediction += 1

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    max_count = max(label_counts.values())
    for action, count in label_counts.items():
        if count == max_count:
            print(f"Predicted Action: {action} (Count: {count}/{number_of_prediction})")

# Sử dụng hàm dự đoán hành động
predict_action(model, pose, mpDraw, unnecessary_points, actions, "./data/predict_data/CFM.mp4")
