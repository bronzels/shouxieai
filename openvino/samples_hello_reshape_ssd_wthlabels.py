"""
omz_downloader --name mobilenet-ssd

omz_converter --name mobilenet-ssd
ls public/mobilenet-ssd/FP16
ls public/mobilenet-ssd/FP32

"""

"""
"""
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
with open('labels-voc.txt', 'w') as fp:
    fp.write('\n'.join(voc_labels))
    fp.close()
"""

"""
import logging as log
import os
import sys

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Layout, PartialShape, Type

import time

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # Parsing and validation of input arguments
    if len(sys.argv) != 5:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name>')
        return 1

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device_name = sys.argv[3]
    label_file_name = sys.argv[4]
    with open(label_file_name, 'r') as fp:
        labels = [line.replace('\n', '') for line in fp.readlines()]

    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    log.info(f'Reading the model: {model_path}')
    model = core.read_model(model_path)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    image = cv2.imread(image_path)
    input_tensor = np.expand_dims(image, 0)

    log.info('Reshaping the model to the height and width of the input image')
    n, h, w, c = input_tensor.shape
    model.reshape({model.input().get_any_name(): PartialShape((n, c, h, w))})

    ppp = PrePostProcessor(model)

    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))

    ppp.input().model().set_layout(Layout('NCHW'))

    ppp.output().tensor().set_element_type(Type.f32)

    model = ppp.build()

    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, device_name)

    log.info('Starting inference in synchronous mode')
    a = time.time()
    results = compiled_model.infer_new_request({0: input_tensor})
    b = time.time()
    log.info(f'---debug---infer time:{b-a}')

    predictions = next(iter(results.values()))

    detections = predictions.reshape(-1, 7)

    for detection in detections:
        confidence = detection[2]

        if confidence > 0.5:
            class_id = int(detection[1])

            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)

            log.info(f'Found: class = {labels[class_id-1]}, confidence = {confidence:.2f}, ' f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imwrite('out.bmp', image)

    if os.path.exists('out.bmp'):
        log.info('Image out.bmp was created!')
    else:
        log.error('Image out.bmp was not created. Check your permissions.')

# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
python samples_hello_reshape_ssd_wthlabels.py public/mobilenet-ssd/FP32/mobilenet-ssd.xml car.jpeg CPU labels-voc.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: public/mobilenet-ssd/FP32/mobilenet-ssd.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] ---debug---infer time:0.11163520812988281
[ INFO ] Found: class = car, confidence = 0.99, coords = (17, 23), (117, 101)
[ INFO ] Image out.bmp was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_reshape_ssd_wthlabels.py public/mobilenet-ssd/FP32/mobilenet-ssd.xml laptop.jpeg CPU labels-voc.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: public/mobilenet-ssd/FP32/mobilenet-ssd.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] ---debug---infer time:0.04282188415527344
[ INFO ] Image out.bmp was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_reshape_ssd_wthlabels.py public/mobilenet-ssd/FP32/mobilenet-ssd.xml dog.jpeg CPU labels-voc.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: public/mobilenet-ssd/FP32/mobilenet-ssd.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] ---debug---infer time:0.012285947799682617
[ INFO ] Found: class = dog, confidence = 0.94, coords = (0, 22), (101, 108)
[ INFO ] Image out.bmp was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

python samples_hello_reshape_ssd_wthlabels.py public/mobilenet-ssd/FP16/mobilenet-ssd.xml car.jpeg CPU labels-voc.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: public/mobilenet-ssd/FP16/mobilenet-ssd.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] ---debug---infer time:0.032166242599487305
[ INFO ] Found: class = car, confidence = 0.99, coords = (17, 23), (117, 101)
[ INFO ] Image out.bmp was created!
[ INFO
python samples_hello_reshape_ssd_wthlabels.py public/mobilenet-ssd/FP32/mobilenet-ssd.xml car.jpeg CPU labels-voc.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: public/mobilenet-ssd/FP32/mobilenet-ssd.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] ---debug---infer time:0.01745891571044922
[ INFO ] Found: class = car, confidence = 0.99, coords = (17, 23), (117, 101)
[ INFO ] Image out.bmp was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
python samples_hello_reshape_ssd_wthlabels.py public/mobilenet-ssd/FP16/mobilenet-ssd.xml car.jpeg CPU labels-voc.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: public/mobilenet-ssd/FP16/mobilenet-ssd.xml
[ INFO ] Reshaping the model to the height and width of the input image
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] ---debug---infer time:0.011810064315795898
[ INFO ] Found: class = car, confidence = 0.99, coords = (17, 23), (117, 101)
[ INFO ] Image out.bmp was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

"""