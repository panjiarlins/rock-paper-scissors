import tensorflow as tf
print(tf.__version__)

!wget --no-check-certificate \
  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip

import zipfile, os

zip_dir = os.path.join('/tmp', 'rockpaperscissors.zip')
zip_file = zipfile.ZipFile(zip_dir, 'r')
zip_file.extractall('/tmp')
zip_file.close()

base_dir = os.path.join('/tmp', 'rockpaperscissors', 'rps-cv-images')
os.listdir(base_dir)

dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.4)
 
train_gen = dataGenerator.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        subset='training')
 
validation_gen = dataGenerator.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=32,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical',
        subset='validation')

from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

my_model = tf.keras.models.Sequential([
    Conv2D(16, (5,5), strides=(2,2), activation='relu', input_shape=(150, 150, 3)),
    ZeroPadding2D(padding=(1,1)),
    Conv2D(32, (3,3), strides=(2,2), activation='relu'),
    MaxPooling2D((2,2), strides=(2,2)),
    Conv2D(64, (3,3), strides=(1,1), activation='relu'),
    MaxPooling2D((2,2), strides=(1,1)),
    Conv2D(128, (3,3), strides=(1,1), activation='relu'),
    MaxPooling2D((2,2), strides=(1,1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
my_model.summary()

from tensorflow.keras.callbacks import EarlyStopping, History

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = History()

my_model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])

my_model.fit(
      train_gen,
      steps_per_epoch = train_gen.n // train_gen.batch_size,
      epochs = 25,
      validation_data = validation_gen, 
      validation_steps = validation_gen.n // validation_gen.batch_size,  
      callbacks = [early_stop, history],
      verbose = 2)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
 
uploaded = files.upload()
 
for fn in uploaded.keys(): 
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])

  resultProb = my_model.predict(images) 
  resultLabel = resultProb.argmax(axis=-1)

  if resultLabel == 0:
    plt.title("Result: Paper")
  elif resultLabel == 1:
    plt.title("Result: Rock")
  elif resultLabel == 2:
    plt.title("Result: Scissor")