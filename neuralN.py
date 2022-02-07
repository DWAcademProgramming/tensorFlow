# Import the image set
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O ./cats_and_dogs_filtered.zip

# Import Project Dependencies
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator
%matplotlib inline 

#Unzipping the dataset
dataset_path = "./cats_and_dogs_filtered.zip"
zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall("./")
zip_object.close()

#Setting up the database paths
dataset_path_new = "./cats_and_dogs_filtered/"
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

#Building the base model
IMG_SHAPE = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

# Freezing the base model is an essential aspect of this code
base_model.trainable = False
base_model.output

#Global and Prediction Layers
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

#Defining the model
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

#Compiling the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=["accuracy"])
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

#Building the generator
train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_train.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

#Training the generator
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

#Transfer the learning model evaluation
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

#Fine tuning
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]: 
  layer.trainable = False

#Compiling the model for fine tuning
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 
              loss='binary_crossentropy', 
              metrics=["accuracy"])

#Fine tuning the model
model.fit_generator(train_generator, 
                    epochs=5, 
                    validation_data=valid_generator)

#Generating the final results
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)

#Displaying the final results
valid_accuracy
