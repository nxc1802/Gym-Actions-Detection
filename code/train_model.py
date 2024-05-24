import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Đường dẫn tới thư mục chứa các thư mục CSV
csv_input_folder = "./data/train_data"

# Danh sách các hành động và nhãn tương ứng (các thư mục con)
actions = ['barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift', 'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press', 'lat pulldown', 'lateral raise', 'leg extension', 'leg raises', 'plank', 'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 'shoulder press', 'squat', 't bar row', 'tricep dips', 'tricep Pushdown']
label_map = {action: idx for idx, action in enumerate(actions)}

X = []
y = []
no_of_timesteps = 60

# Duyệt qua từng thư mục con
for action in actions:
    action_folder_path = os.path.join(csv_input_folder, action)
    csv_files = [f for f in os.listdir(action_folder_path) if f.endswith('.csv')]

    # Duyệt qua từng file CSV trong thư mục con
    for csv_file in csv_files:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(os.path.join(action_folder_path, csv_file), header=None)
        dataset = df.values
        
        # Thêm dữ liệu vào danh sách
        X.append(dataset)
        y.append(label_map[action])

X, y = np.array(X), np.array(y)

# print(X.shape, y.shape)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode nhãn
y_train = to_categorical(y_train, num_classes=len(actions))
y_test = to_categorical(y_test, num_classes=len(actions))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=len(actions), activation="softmax"))
model.compile(optimizer="adam", metrics=['accuracy'], loss="categorical_crossentropy")

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.save("model.h5")
