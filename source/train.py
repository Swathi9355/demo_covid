import numpy as np
import os
from build import build_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

basedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")

# Data generators
train_img = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_img = ImageDataGenerator(rescale=1. / 255)

# Training data
train_data = train_img.flow_from_directory(
    os.path.join(basedir, "train"),
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32
)

# Test/Validation data
test_data = test_img.flow_from_directory(
    os.path.join(basedir, "test"),
    target_size=(224, 224),
    class_mode="categorical",
    batch_size=32
)

# Build and train model
model = build_model()
model.fit(train_data, epochs=10, validation_data=test_data)

# Save the model
model.save(os.path.join(basedir, "covid_pneu_model.h5"))