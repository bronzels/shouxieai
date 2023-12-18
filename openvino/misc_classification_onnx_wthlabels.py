"""
"""
import logging as log
import sys

from openvino.inference_engine import IECore
from openvino.runtime import Core
import cv2
import numpy as np

import time

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    if len(sys.argv) != 5:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name> <label_file_name>')
        return 1

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device_name = sys.argv[3]
    label_file_name = sys.argv[4]
    with open(label_file_name, 'r') as fp:
        labels = [line.replace('\n', '') for line in fp.readlines()]

    if model_path.endswith("-static.onnx"):
        log.info('Creating OpenVINO Runtime Core')
        ie = IECore()
        log.info(f'Reading the model:{model_path}')
        net = ie.read_network(model=model_path)
        exec_net = ie.load_network(network=net, device_name=device_name)
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))
        _, _, h, w = net.input_info[input_blob].input_data.shape
    else:
        ie = Core()
        compiled_model = ie.compile_model(model=model_path, device_name=device_name)
        output_blob = compiled_model.output(0)
        h, w = 224, 224

    a1 = time.time()
    image_paths = image_path.split(',')
    images = []
    for image_path in image_paths:
        src = cv2.imread(image_path)
        sub_image = cv2.resize(src, (w, h))
        sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
        sub_image = np.float32(sub_image) / 255.0
        sub_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406),)
        sub_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225),)
        sub_image = sub_image.transpose((2, 0, 1))
        images.append(sub_image)

    image = np.stack(images, axis=0)

    #log.info('Starting inference in synchronous mode')
    a2 = time.time()
    if model_path.endswith("-static.onnx"):
        #res = exec_net.infer(inputs={input_blob: [image]})
        res = exec_net.infer(inputs={input_blob: image})[output_blob]
    else:
        #image = np.expand_dims(image, 0)
        res = compiled_model([image])[output_blob]
    b = time.time()
    log.info(f'---debug---infer time:{b-a2}')
    log.info(f'---debug---preprocess + infer time:{b-a1}')

    for index, image_path in enumerate(image_paths):
        probs = res[index]
        top_10 = np.argsort(probs)[-10:][::-1]

        header = 'class_id        probability'

        log.info(f'Image path: {image_path}')
        log.info('Top 10 results: ')
        log.info(header)
        log.info('-' * len(header))

        for class_id in top_10:
            number_indent = len('class_id       ') - len(str(labels[class_id])) + 1
            probability_indent = ' ' * number_indent
            class_str = str(labels[class_id])
            log.info(f'{class_str}{probability_indent}{probs[class_id]:.7f}')

    log.info('')

    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())


"""
/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-static.onnx car.jpeg CPU labels-imagenet.txt 
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet50-static.onnx
[ INFO ] ---debug---infer time:0.09180426597595215
[ INFO ] ---debug---preprocess + infer time:0.27508020401000977
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1131859
[ INFO ] CANOE           8.7460232
[ INFO ] SCREWDRIVER     8.6130886
[ INFO ] PADDLE          7.5107813
[ INFO ] HAND-HELD COMPUTER7.2564745
[ INFO ] SAFETY PIN      6.8813004
[ INFO ] TRAILER TRUCK   6.7991319
[ INFO ] CONVERTIBLE     6.7498298
[ INFO ] CAN OPENER      6.6727839
[ INFO ] LAWN MOWER      6.6678133
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-static.onnx laptop.jpeg CPU labels-imagenet.txt 
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet50-static.onnx
[ INFO ] ---debug---infer time:0.09436917304992676
[ INFO ] ---debug---preprocess + infer time:0.11474394798278809
[ INFO ] Image path: laptop.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] NOTEBOOK        12.9896173
[ INFO ] LAPTOP          12.3024321
[ INFO ] DESKTOP COMPUTER11.4877920
[ INFO ] SCREEN          11.0541496
[ INFO ] SPACE BAR       10.8488340
[ INFO ] HAND-HELD COMPUTER10.1742487
[ INFO ] MOUSE           8.8540459
[ INFO ] MONITOR         8.7412739
[ INFO ] DESK            8.2771416
[ INFO ] WEB SITE        8.1865005
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-static.onnx dog.jpeg CPU labels-imagenet.txt 
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:resnet50-static.onnx
[ INFO ] ---debug---infer time:0.1067812442779541
[ INFO ] ---debug---preprocess + infer time:0.127913236618042
[ INFO ] Image path: dog.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] PEMBROKE        16.7579746
[ INFO ] CARDIGAN        15.4459257
[ INFO ] DINGO           8.4533911
[ INFO ] BASENJI         8.1822948
[ INFO ] CHIHUAHUA       8.1629667
[ INFO ] COLLIE          8.1357021
[ INFO ] PAPILLON        8.0254154
[ INFO ] KELPIE          7.6495848
[ INFO ] BORDER COLLIE   7.4995995
[ INFO ] NORWICH TERRIER 7.4799156
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-dynamic.onnx car.jpeg CPU labels-imagenet.txt 
[ INFO ] ---debug---infer time:0.18404603004455566
[ INFO ] ---debug---preprocess + infer time:0.18706202507019043
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1131859
[ INFO ] CANOE           8.7460232
[ INFO ] SCREWDRIVER     8.6130886
[ INFO ] PADDLE          7.5107813
[ INFO ] HAND-HELD COMPUTER7.2564745
[ INFO ] SAFETY PIN      6.8813004
[ INFO ] TRAILER TRUCK   6.7991319
[ INFO ] CONVERTIBLE     6.7498298
[ INFO ] CAN OPENER      6.6727839
[ INFO ] LAWN MOWER      6.6678133
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-dynamic.onnx laptop.jpeg CPU labels-imagenet.txt 
[ INFO ] ---debug---infer time:0.14465093612670898
[ INFO ] ---debug---preprocess + infer time:0.16127467155456543
[ INFO ] Image path: laptop.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] NOTEBOOK        12.9896173
[ INFO ] LAPTOP          12.3024321
[ INFO ] DESKTOP COMPUTER11.4877920
[ INFO ] SCREEN          11.0541496
[ INFO ] SPACE BAR       10.8488340
[ INFO ] HAND-HELD COMPUTER10.1742487
[ INFO ] MOUSE           8.8540459
[ INFO ] MONITOR         8.7412739
[ INFO ] DESK            8.2771416
[ INFO ] WEB SITE        8.1865005
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-dynamic.onnx dog.jpeg CPU labels-imagenet.txt 
[ INFO ] ---debug---infer time:0.15482878684997559
[ INFO ] ---debug---preprocess + infer time:0.17514395713806152
[ INFO ] Image path: dog.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] PEMBROKE        16.7579746
[ INFO ] CARDIGAN        15.4459257
[ INFO ] DINGO           8.4533911
[ INFO ] BASENJI         8.1822948
[ INFO ] CHIHUAHUA       8.1629667
[ INFO ] COLLIE          8.1357021
[ INFO ] PAPILLON        8.0254154
[ INFO ] KELPIE          7.6495848
[ INFO ] BORDER COLLIE   7.4995995
[ INFO ] NORWICH TERRIER 7.4799156
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-dynamic.onnx car.jpeg,dog.jpeg CPU labels-imagenet.txt 
[ INFO ] ---debug---infer time:0.5754067897796631
[ INFO ] ---debug---preprocess + infer time:0.8114719390869141
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1131868
[ INFO ] CANOE           8.7460279
[ INFO ] SCREWDRIVER     8.6130886
[ INFO ] PADDLE          7.5107803
[ INFO ] HAND-HELD COMPUTER7.2564735
[ INFO ] SAFETY PIN      6.8812990
[ INFO ] TRAILER TRUCK   6.7991319
[ INFO ] CONVERTIBLE     6.7498288
[ INFO ] CAN OPENER      6.6727829
[ INFO ] LAWN MOWER      6.6678143
[ INFO ] Image path: dog.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] PEMBROKE        16.7579746
[ INFO ] CARDIGAN        15.4459219
[ INFO ] DINGO           8.4533911
[ INFO ] BASENJI         8.1822968
[ INFO ] CHIHUAHUA       8.1629658
[ INFO ] COLLIE          8.1357012
[ INFO ] PAPILLON        8.0254145
[ INFO ] KELPIE          7.6495843
[ INFO ] BORDER COLLIE   7.4995999
[ INFO ] NORWICH TERRIER 7.4799147
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/classification_onnx_wthlabels.py resnet50-dynamic.onnx car.jpeg,laptop.jpeg,dog.jpeg CPU labels-imagenet.txt 
[ INFO ] ---debug---infer time:0.3268129825592041
[ INFO ] ---debug---preprocess + infer time:0.3421509265899658
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] SPORTS CAR      9.1131859
[ INFO ] CANOE           8.7460270
[ INFO ] SCREWDRIVER     8.6130886
[ INFO ] PADDLE          7.5107803
[ INFO ] HAND-HELD COMPUTER7.2564735
[ INFO ] SAFETY PIN      6.8812995
[ INFO ] TRAILER TRUCK   6.7991319
[ INFO ] CONVERTIBLE     6.7498288
[ INFO ] CAN OPENER      6.6727839
[ INFO ] LAWN MOWER      6.6678138
[ INFO ] Image path: laptop.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] NOTEBOOK        12.9896173
[ INFO ] LAPTOP          12.3024311
[ INFO ] DESKTOP COMPUTER11.4877920
[ INFO ] SCREEN          11.0541477
[ INFO ] SPACE BAR       10.8488359
[ INFO ] HAND-HELD COMPUTER10.1742477
[ INFO ] MOUSE           8.8540468
[ INFO ] MONITOR         8.7412729
[ INFO ] DESK            8.2771416
[ INFO ] WEB SITE        8.1865005
[ INFO ] Image path: dog.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id        probability
[ INFO ] ---------------------------
[ INFO ] PEMBROKE        16.7579727
[ INFO ] CARDIGAN        15.4459219
[ INFO ] DINGO           8.4533911
[ INFO ] BASENJI         8.1822958
[ INFO ] CHIHUAHUA       8.1629667
[ INFO ] COLLIE          8.1357012
[ INFO ] PAPILLON        8.0254145
[ INFO ] KELPIE          7.6495843
[ INFO ] BORDER COLLIE   7.4995999
[ INFO ] NORWICH TERRIER 7.4799147
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

"""