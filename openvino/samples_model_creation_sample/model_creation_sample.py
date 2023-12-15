import logging as log
import sys
import typing
from functools import reduce

import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import (Core, Layout, Model, Shape, Type, op, opset1,
                              opset8, set_batch)

from data import digits


def create_ngraph_function(model_path: str) -> Model:

    def shape_and_length(shape: list) -> typing.Tuple[list, int]:
        length = reduce(lambda x, y: x * y, shape)
        return shape, length

    weights = np.fromfile(model_path, dtype=np.float32)
    weights_offset = 0
    padding_begin = padding_end = [0, 0]

    # input
    input_shape = [64, 1, 28, 28]
    param_node = op.Parameter(Type.f32, Shape(input_shape))

    # convolution 1
    conv_1_kernel_shape, conv_1_kernel_length = shape_and_length([20, 1, 5, 5])
    conv_1_kernel = op.Constant(Type.f32, Shape(conv_1_kernel_shape), weights[0:conv_1_kernel_length].tolist())
    weights_offset += conv_1_kernel_length
    conv_1_node = opset8.convolution(param_node, conv_1_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 1
    add_1_kernel_shape, add_1_kernel_length = shape_and_length([1, 20, 1, 1])
    add_1_kernel = op.Constant(Type.f32, Shape(add_1_kernel_shape),
                               weights[weights_offset : weights_offset + add_1_kernel_length])
    weights_offset += add_1_kernel_length
    add_1_node = opset8.add(conv_1_node, add_1_kernel)

    # maxpool 1
    maxpool_1_node = opset1.max_pool(add_1_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

    # convolution 2
    conv_2_kernel_shape, conv_2_kernel_length = shape_and_length([50, 20, 5, 5])
    conv_2_kernel = op.Constant(Type.f32, Shape(conv_2_kernel_shape),
                                weights[weights_offset : weights_offset + conv_2_kernel_length],
                                )
    weights_offset += conv_2_kernel_length
    conv_2_node = opset8.convolution(maxpool_1_node, conv_2_kernel, [1, 1], padding_begin, padding_end, [1, 1])

    # add 2
    add_2_kernel_shape, add_2_kernel_length = shape_and_length([1, 50, 1, 1])
    add_2_kernel = op.Constant(Type.f32, Shape(add_2_kernel_shape),
                               weights[weights_offset : weights_offset + add_2_kernel_length],
                               )
    weights_offset += add_2_kernel_length
    add_2_node = opset8.add(conv_2_node, add_2_kernel)

    # maxpool 2
    maxpool_2_node = opset1.max_pool(add_2_node, [2, 2], padding_begin, padding_end, [2, 2], 'ceil')

    # reshape 1
    reshape_1_dims, reshape_1_length = shape_and_length([2])
    dtype_weights = np.frombuffer(
        weights[weights_offset : weights_offset + 2 * reshape_1_length],
        dtype=np.int64,
    )
    reshape_1_kernel = op.Constant(Type.i64, Shape(list(dtype_weights.shape)), dtype_weights)
    weights_offset += 2 * reshape_1_length
    reshape_1_node = opset8.reshape(maxpool_2_node, reshape_1_kernel, True)

    # matmul 1
    matmul_1_kernel_shape, matmul_1_kernel_length = shape_and_length([500, 800])
    matmul_1_kernel = op.Constant(Type.f32, Shape(matmul_1_kernel_shape),
                                  weights[weights_offset : weights_offset + matmul_1_kernel_length],
                                  )
    weights_offset += matmul_1_kernel_length
    matmul_1_node = opset8.matmul(reshape_1_node, matmul_1_kernel, False, True)

    # add 3
    add_3_kernel_shape, add_3_kernel_length = shape_and_length([1, 500])
    add_3_kernel = op.Constant(Type.f32, Shape(add_3_kernel_shape),
                               weights[weights_offset : weights_offset + add_3_kernel_length],
                               )
    weights_offset += add_3_kernel_length
    add_3_node = opset8.add(matmul_1_node, add_3_kernel)

    # ReLU
    relu_node = opset8.relu(add_3_node)

    # reshape 2
    reshape_2_kernel = op.Constant(Type.i64, Shape(list(dtype_weights.shape)), dtype_weights)
    reshape_2_node = opset8.reshape(relu_node, reshape_2_kernel, True)

    # matmul 2
    matmul_2_kernel_shape, matmul_2_kernel_length = shape_and_length([10, 500])
    matmul_2_kernel = op.Constant(Type.f32, Shape(matmul_2_kernel_shape),
                                  weights[weights_offset : weights_offset + matmul_2_kernel_length],
                                  )
    weights_offset += matmul_2_kernel_length
    matmul_2_node = opset8.matmul(reshape_2_node, matmul_2_kernel, False, True)

    # add 4
    add_4_kernel_shape, add_4_kernel_length = shape_and_length([1, 10])
    add_4_kernel = op.Constant(Type.f32, Shape(add_4_kernel_shape),
                               weights[weights_offset : weights_offset + add_4_kernel_length],
                               )
    weights_offset += add_4_kernel_length
    add_4_node = opset8.add(matmul_2_node, add_4_kernel)

    # softmax
    softmax_axis = 1
    softmax_node = opset8.softmax(add_4_node, softmax_axis)

    return Model(softmax_node, [param_node], 'lenet')


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    # Parsing and validation of input arguments
    if len(sys.argv) != 3:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <device_name>')
        return 1

    model_path = sys.argv[1]
    device_name = sys.argv[2]
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    number_top = 1
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

    log.info(f'Loading the model using ngraph function with weights from {model_path}')
    model = create_ngraph_function(model_path)
    ppp = PrePostProcessor(model)
    ppp.input().tensor() \
        .set_element_type(Type.u8) \
        .set_layout(Layout('NHWC'))  # noqa: N400

    ppp.input().model().set_layout(Layout('NCHW'))
    ppp.output().tensor().set_element_type(Type.f32)

    model = ppp.build()

    set_batch(model, digits.shape[0])
    compiled_model = core.compile_model(model, device_name)

    n, c, h, w = model.input().shape
    input_data = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = digits[i].reshape(28, 28)
        image = image[:, :, np.newaxis]
        input_data[i] = image

    results = compiled_model.infer_new_request({0: input_data})

    predictions = next(iter(results.values()))

    log.info(f'Top {number_top} results: ')
    for i in range(n):
        probs = predictions[i]
        top_n_indexes = np.argsort(probs)[-number_top:][::-1]

        header = 'classid probability'
        header = header + ' label' if labels else header

        log.info(f'Image {i}')
        log.info('')
        log.info(header)
        log.info('-' * len(header))

        for class_id in top_n_indexes:
            probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
            label_indent = ' ' * (len('probability') - 8) if labels else ''
            label = labels[class_id] if labels else ''
            log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}{label_indent}{label}')
        log.info('')

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/samples_model_creation_sample/model_creation_sample.py lenet.bin CPU 
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Loading the model using ngraph function with weights from lenet.bin
[ INFO ] Top 1 results: 
[ INFO ] Image 0
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 0       1.0000000   0
[ INFO ] 
[ INFO ] Image 1
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 1       1.0000000   1
[ INFO ] 
[ INFO ] Image 2
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 2       1.0000000   2
[ INFO ] 
[ INFO ] Image 3
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 3       1.0000000   3
[ INFO ] 
[ INFO ] Image 4
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 4       1.0000000   4
[ INFO ] 
[ INFO ] Image 5
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 5       1.0000000   5
[ INFO ] 
[ INFO ] Image 6
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 6       1.0000000   6
[ INFO ] 
[ INFO ] Image 7
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 7       1.0000000   7
[ INFO ] 
[ INFO ] Image 8
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 8       1.0000000   8
[ INFO ] 
[ INFO ] Image 9
[ INFO ] 
[ INFO ] classid probability label
[ INFO ] -------------------------
[ INFO ] 9       1.0000000   9
[ INFO ] 
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

"""