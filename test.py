##
## Zeynep Ferah Akkurt
##
## Image Classification Modelinin test edilmesi
## test.py
##

import os

import tensorflow as tf
from tensorflow import keras

import pathlib

model = keras.models.load_model("model artifact")

## Testing
mainPath = os.getcwd()
path = pathlib.Path(mainPath + "/dataset/NEU-CLS-64_tf_mode/finalTest")


image_count2 = len(list(path.glob('*/*.jpg')))
print(image_count2)

batch_size = 8
img_height = 64
img_width = 64

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  path,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


acc = model.evaluate(test_ds)