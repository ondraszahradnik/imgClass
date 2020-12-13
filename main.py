# IMPORTS EXTERNAL MODULES
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# IMPORTS BUILT-IN MODULES
import os
import pathlib

# STORES THE PATH OF THE IMAGES DIRECTORY
images_dir = pathlib.Path('images')

batch_size = 32
img_height = 256 # ???
img_width = 256 # ???

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  images_dir,
  validation_split = 0.2,
  subset = 'training',
  seed = 123, # ???
  image_size = (img_height, img_width),
  batch_size = batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  images_dir,
  validation_split = 0.2,
  subset = 'validation',
  seed = 123, # ???
  image_size = (img_height, img_width),
  batch_size = batch_size)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10)) # ???
for images, labels in train_ds.take(1):
  for i in range(9): # ???
    ax = plt.subplot(3, 3, i + 1) # ???
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 7 # ???

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer = 'adam', # ???
  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics = ['accuracy'])

model.fit(
  train_ds,
  validation_data = val_ds,
  epochs = 3 # ???
)

model.save('model')
