#
# Zeynep Ferah Akkurt
#
# Image Classification Modeli Oluşturulması
# train.py
#

import os

import pathlib
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tensorflow_version = float(tf.__version__[0:3])
print(f"Your tensorflow version is : {tensorflow_version}\n")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset
mainPath = os.getcwd()
pathTrain = pathlib.Path(mainPath + "/dataset/NEU-CLS-64_tf_mode/train")
pathTest = pathlib.Path(mainPath + "/dataset/NEU-CLS-64_tf_mode/test")

image_count2 = len(list(pathTest.glob('*/*.jpg')))
print(image_count2)

batch_size = 8
img_height = 64
img_width = 64

# train dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathTrain,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# validation dataset (%25)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathTest,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create the model
num_classes = 6

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

model = Sequential([
    # data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training-validation.png')
plt.show()

model.save("model-artifact")
