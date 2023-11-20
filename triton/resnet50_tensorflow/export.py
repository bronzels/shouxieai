import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights="imagenet")
model.save("./1/model.savedmodel")
