import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


DATASET_PATH = "OACE"   # <-- adjust if your folder name is different
classes = ["Close", "Open"]

def preprocess(img_path, size=64):
    """Load image, grayscale, resize, normalize"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img

X, y = [], []

for label, category in enumerate(classes):
    folder = os.path.join(DATASET_PATH, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        try:
            features = preprocess(img_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print("Error loading:", img_path, e)

X = np.array(X).reshape(-1, 64, 64, 1)  # CNN needs (H,W,channels)
y = np.array(y)
print("Dataset loaded:", X.shape, "Labels:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=classes))

test_img_path = os.path.join(DATASET_PATH, "Open", os.listdir(os.path.join(DATASET_PATH, "Open"))[0])
test_img = preprocess(test_img_path).reshape(1, 64, 64, 1)

prediction = model.predict(test_img)[0][0]
print("\nSample test image:", test_img_path)
print("Prediction:", "Open" if prediction > 0.5 else "Close")
