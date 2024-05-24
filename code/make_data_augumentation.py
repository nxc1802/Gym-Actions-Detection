import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

# Đường dẫn tới thư mục chứa video
video_folder = "./data/image/clean_data/barbell biceps curl"
output_folder = "./data/image/augmented_data/barbell biceps curl"
os.makedirs(output_folder, exist_ok=True)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append((lm.x, lm.y, lm.z, lm.visibility))
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
    return img

def flip_image(image, lm_list):
    flipped_image = cv2.flip(image, 1)
    h, w, _ = image.shape
    new_lm_list = []
    for lm in lm_list:
        new_lm_list.append((1 - lm[0], lm[1], lm[2], lm[3]))
    return flipped_image, new_lm_list

def rotate_image(image, lm_list, angle):
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    new_lm_list = []
    for lm in lm_list:
        x, y = lm[0] * w, lm[1] * h
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        new_lm_list.append((new_x / w, new_y / h, lm[2], lm[3]))
    return rotated_image, new_lm_list

def translate_image(image, lm_list, tx, ty):
    h, w, _ = image.shape
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, T, (w, h))

    new_lm_list = []
    for lm in lm_list:
        new_x = lm[0] + tx / w
        new_y = lm[1] + ty / h
        new_x = min(max(new_x, 0), 1)
        new_y = min(max(new_y, 0), 1)
        new_lm_list.append((new_x, new_y, lm[2], lm[3]))
    return translated_image, new_lm_list

# Đọc các video từ thư mục
video_files = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]

for video_file in video_files:
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
    lm_list = []
    stop = 1

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

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Augmentation and save
    for augment, augment_name in [
        (flip_image, 'flipping'),
        (lambda img, lm: rotate_image(img, lm, -5), 'rotation_-5'),
        (lambda img, lm: rotate_image(img, lm, 5), 'rotation_5'),
        (lambda img, lm: translate_image(img, lm, 10, 10), 'translation_10_10'),
        (lambda img, lm: translate_image(img, lm, -10, -10), 'translation_-10_-10')
    ]:
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        out_video_file = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{augment_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        aug_lm_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)

            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                aug_frame, aug_lm = augment(frame, lm)
                aug_lm_list.append(aug_lm)
                for i, aug_lm_step in enumerate(aug_lm):
                    results.pose_landmarks.landmark[i].x = aug_lm_step[0]
                    results.pose_landmarks.landmark[i].y = aug_lm_step[1]
                aug_frame = draw_landmark_on_image(mpDraw, results, aug_frame)
                out.write(aug_frame)

            cv2.imshow("augmented image", aug_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        df = pd.DataFrame(aug_lm_list)
        csv_file_name = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{augment_name}.csv")
        df.to_csv(csv_file_name, index=False)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if stop: break
