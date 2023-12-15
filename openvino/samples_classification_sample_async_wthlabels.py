import argparse
import logging as log
import sys

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import AsyncInferQueue, Core, InferRequest, Layout, Type

import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', type=str, required=True, nargs='+',
                      help='Required. Path to an image file(s).')
    args.add_argument('-d', '--device', type=str, default='CPU',
                      help='Optional. Specify the target device to infer on; CPU, GPU, GNA or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    return parser.parse_args()


with open('labels-resnet50.txt', 'r') as fp:
    labels = [line.replace('\n', '') for line in fp.readlines()]

def completion_callback(infer_request: InferRequest, image_path: str) -> None:
    predictions = next(iter(infer_request.results.values()))

    probs = predictions.reshape(-1)

    top_10 = np.argsort(probs)[-10:][::-1]

    header = 'class_id probability'

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


def main() -> int:
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    log.info(f'Reading the model: {args.model}')
    model = core.read_model(args.model)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    images = [cv2.imread(image_path) for image_path in args.input]

    _, _, h, w = model.input().shape
    resized_images = [cv2.resize(image, (w, h)) for image in images]

    input_tensors = [np.expand_dims(image, 0) for image in resized_images]

    ppp = PrePostProcessor(model)

    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))  # noqa: N400

    ppp.input().model().set_layout(Layout('NCHW'))

    ppp.output().tensor().set_element_type(Type.f32)

    model = ppp.build()

    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, args.device)

    log.info('Starting inference in asynchronous mode')
    infer_queue = AsyncInferQueue(compiled_model)
    infer_queue.set_callback(completion_callback)

    a = time.time()
    for i, input_tensor in enumerate(input_tensors):
        infer_queue.start_async({0: input_tensor}, args.input[i])

    infer_queue.wait_all()
    b = time.time()
    log.info(f'---debug---infer time:{b-a}')
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
python samples_classification_sample_async_wthlabels.py --model alexnet.xml --input car.jpeg laptop.jpeg dog.jpeg 
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model: alexnet.xml
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in asynchronous mode
[ INFO ] Image path: car.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id probability
[ INFO ] --------------------
[ INFO ] BOBSLED         0.2662014
[ INFO ] SPORTS CAR      0.1130833
[ INFO ] AIRSHIP         0.0516823
[ INFO ] OCARINA         0.0461483
[ INFO ] WAFFLE IRON     0.0321314
[ INFO ] PROJECTILE      0.0320070
[ INFO ] TOASTER         0.0305676
[ INFO ] HARMONICA       0.0292178
[ INFO ] WHISTLE         0.0288203
[ INFO ] CONVERTIBLE     0.0287943
[ INFO ] 
[ INFO ] Image path: laptop.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id probability
[ INFO ] --------------------
[ INFO ] HAND-HELD COMPUTER0.4617566
[ INFO ] SPACE BAR       0.3990846
[ INFO ] DESKTOP COMPUTER0.0428079
[ INFO ] LAPTOP          0.0393335
[ INFO ] TYPEWRITER KEYBOARD0.0334197
[ INFO ] SCREEN          0.0105551
[ INFO ] NOTEBOOK        0.0055327
[ INFO ] MODEM           0.0031176
[ INFO ] COMPUTER KEYBOARD0.0015323
[ INFO ] CELLULAR TELEPHONE0.0008552
[ INFO ] 
[ INFO ] Image path: dog.jpeg
[ INFO ] Top 10 results: 
[ INFO ] class_id probability
[ INFO ] --------------------
[ INFO ] PEMBROKE        0.7276618
[ INFO ] CARDIGAN        0.2694458
[ INFO ] BASENJI         0.0011975
[ INFO ] CHIHUAHUA       0.0004558
[ INFO ] ESKIMO DOG      0.0003778
[ INFO ] SIBERIAN HUSKY  0.0002267
[ INFO ] PAPILLON        0.0001588
[ INFO ] DINGO           0.0001495
[ INFO ] COLLIE          0.0000540
[ INFO ] BORDER COLLIE   0.0000540
[ INFO ] 
[ INFO ] ---debug---infer time:0.23291611671447754
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
"""