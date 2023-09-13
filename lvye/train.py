import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model1 = resnet50.ResNet50(weights='imagenet')
model1.save('weights.h5')
model = tf.keras.models.load_model("weights.h5")

img = image.load_img('2008_002682.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = preprocess_input(img)
print("img: ", img.shape)

img = np.expand_dims(img, axis=0)
print(img.shape)

t_model = time.perf_counter()
pred_class = model.predict(img)
print((pred_class.shape))
print(f'do inference cost:{time.perf_counter() - t_model:.8f}s')
print('Predicted:', decode_predictions(pred_class, top=5)[0])

