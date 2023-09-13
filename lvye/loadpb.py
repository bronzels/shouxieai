import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img = image.load_img('2008_002682.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = preprocess_input(img)
print(img.shape)

PATH_TO_CKPT = "weights.pb"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default() as graph:
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        inp = detection_graph.get_tensor_by_name(('Input:0'))
        predictions = detection_graph.get_tensor_by_name('resnet50/predictions/Softmax:0')
        t_model = time.perf_counter()
        x = predictions.eval(feed_dict={inp: img})
        print(f'do inference cost:{time.perf_counter() - t_model:.8f}s')
        print(x.shape)
        print('Predicted:', decode_predictions(x, top=5)[0])

"""
convert-to-uff -o test.uff --input_file weights.pb
先报错gfile，手工修改以后还是报错：
NOTE: UFF has been tested with TensorFlow 1.15.0.
WARNING: The version of TensorFlow installed on this system is not guaranteed to work with UFF.
UFF Version 0.6.9
"""

