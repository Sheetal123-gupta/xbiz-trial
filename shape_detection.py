
import os
path="C:\\Users\\ASUS\\Downloads\\shapes\\signature"
count=0
all_files=os.listdir(path)
ext=('.png','.jpg','.jpeg','.bmp')
for ans in all_files:
  if ans.lower().endswith(ext):
    count=count+1
print(count)


#-------------------------------------------- lets see ------------------------ 
# ✅ 1️⃣ IMPORTS
from importlib.metadata import files
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import zipfile
import os
#from google.colab import files



DATA_DIR = "C:\\Users\\ASUS\\Downloads\\shapes"  # Adjust if your folder structure differs
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(CLASS_NAMES)
print("Detected classes:", CLASS_NAMES)

# ✅ 3️⃣ DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ✅ 4️⃣ CLASS WEIGHTS
labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# ✅ 5️⃣ BUILD MODEL (TRANSFER LEARNING)
base_model = MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ✅ 6️⃣ TRAIN MODEL WITH EARLY STOPPING
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# ✅ 7️⃣ PLOT TRAINING HISTORY
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train', color='red')
plt.plot(history.history['val_accuracy'], label='Validation', color='blue')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ✅ 8️⃣ EVALUATE MODEL
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ✅ 9️⃣ TEST ON A NEW IMAGE
print("\n--- Upload an image to test ---")
uploaded = files.upload()  # Upload your image

img_path = list(uploaded.keys())[0]
img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred_prob = model.predict(img_array)
pred_class = np.argmax(pred_prob, axis=1)[0]
pred_class_name = CLASS_NAMES[pred_class]

print(f"Predicted output--- : {pred_class_name}")
plt.imshow(img)
plt.title(f"Predicted: {pred_class_name}")
plt.axis('off')
plt.show()
