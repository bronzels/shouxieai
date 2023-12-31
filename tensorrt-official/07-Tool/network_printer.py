import numpy as np
import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

inputTensor = network.add_input("inputT0", trt.float32, [-1, 1, 28, 28])
profile.set_shape(inputTensor.name, [1, 1, 28, 28], [4, 1, 28, 28], [16, 1, 28, 28])
config.add_optimization_profile(profile)

w = np.random.rand(32, 1, 5, 5).astype(np.float32).reshape(-1)
b = np.random.rand(32).astype(np.float32).reshape(-1)
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], w, b)
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.random.rand(64, 32, 5, 5).astype(np.float32).reshape(-1)
b = np.random.rand(64).astype(np.float32).reshape(-1)
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], w, b)
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
_6.first_transpose = (0, 2, 3, 1)
_6.reshape_dims = (-1, 64 * 7 * 7)

w = np.random.rand(64 * 7 * 7, 1024).astype(np.float32)
b = np.random.rand(1, 1024).astype(np.float32)
_7 = network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
_8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
_9 = network.add_constant(b.shape, trt.Weights(np.ascontiguousarray(b)))
_10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
_11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

w = np.random.rand(1024, 10).astype(np.float32)
b = np.random.rand(1, 10).astype(np.float32)
_12 = network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
_13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
_14 = network.add_constant(b.shape, trt.Weights(np.ascontiguousarray(b)))
_15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

_16 = network.add_softmax(_15.get_output(0))
_16.axes = 1 << 1

_17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

network.mark_output(_17.get_output(1))

for i in range(network.num_layers):
    layer = network.get_layer(i)
    print("%4d->%s,in=%d,out=%d,%s" % (i, str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
    for j in range(layer.num_inputs):
        tensor = layer.get_input(j)
        print("\tInput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
    for j in range(layer.num_outputs):
        tensor = layer.get_output(j)
        print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))

#engineString = builder.build_serialized_network(network, config)
