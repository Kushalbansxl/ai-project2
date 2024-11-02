# I was unable to push my dataset on Github due to size of Dataset
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

base_dir = '/home/aryan-dhanuka/SML_Lab/Project/chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

classes = os.listdir(train_dir)

for cls in classes:
    cls_train_dir = os.path.join(train_dir, cls)
    cls_val_dir = os.path.join(val_dir, cls)

    if not os.path.exists(cls_val_dir):
        os.makedirs(cls_val_dir)

    images = os.listdir(cls_train_dir)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for img in val_images:
        shutil.move(os.path.join(cls_train_dir, img), os.path.join(cls_val_dir, img))

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
history = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

cnn_accuracy = history.history['val_accuracy'][-1]
print(f"CNN Validation Accuracy: {cnn_accuracy * 100:.2f}%")


def load_image_data(image_dir, target_size=(150, 150)):
    images = []
    labels = []
    class_names = os.listdir(image_dir)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = plt.imread(image_path)
            if img.ndim == 2:  # grayscale to RGB
                img = np.stack((img,) * 3, axis=-1)
            img = np.resize(img, (target_size[0], target_size[1], img.shape[-1]))
            images.append(img.flatten())
            labels.append(label)
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), class_names

X_train, y_train, _ = load_image_data(train_dir)
X_val, y_val, _ = load_image_data(val_dir)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, y_pred)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

import pickle
pickle.dump(rf_model, open("model.pkl", 'wb'))
