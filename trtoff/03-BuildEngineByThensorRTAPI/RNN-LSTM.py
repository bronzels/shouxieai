import os
from datetime import datetime as dt

import numpy as np
import tensorrt as trt
from cuda import cudart

os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
np.random.seed(31193)
tf.compat.v1.set_random_seed(97)
epsilon = 1e-6
nBatchSize, nSequenceLength, nInputDim, nHiddenDim = 2, 4, 7, 5
inputX = np.random.rand(nBatchSize, nSequenceLength, nInputDim).astype(np.float32).reshape([nBatchSize, nSequenceLength, nInputDim])
inputH = np.random.rand(nBatchSize, nHiddenDim).astype(np.float32).reshape([nBatchSize, nHiddenDim])
inputC = np.random.rand(nBatchSize, nHiddenDim).astype(np.float32).reshape([nBatchSize, nHiddenDim])

def printArrayInformation(x, info="", n=5):
    if 0 in x.shape:
        print('%s:%s' % (info, str(x.shape)))
        print()
        return
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1))))))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5, name=''):
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("name:%s, check:%s, absDiff=%f, relDiff=%f" % (name, res, diff0, diff1))

def smallTest(x0, h0, c0):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    para = np.load("test?.npz")
    weight = [np.split(i, [nInputDim], axis=0) for i in np.split(para["?/kernel:0"], 4, axis=1)]
    bias = np.split(para["?/bias:0"], 4)
    h, c = h0, c0
    for t in range(nSequenceLength):
        x = x0[:, t, :]
        it = sigmoid(np.matmul(x, weight[0][0]) + np.matmul(h, weight[0][1]) + bias[0])
        ct_ = np.tanh(np.matmul(x, weight[1][0]) + np.matmul(h, weight[1][1]) + bias[1])
        ft = sigmoid(np.matmul(x, weight[2][0]) + np.matmul(h, weight[2][1]) + bias[2])
        ot = sigmoid(np.matmul(x, weight[3][0]) + np.matmul(h, weight[3][1]) + bias[3])
        ct = ft * c0 + it * ct_
        ht = ot * np.tanh(ct)
        print("ht=\n", ht, "\nct=\n", ct)
        h = ht
        c = ct

    print("here")
    return

def _test1():
    print("\ntf.keras.layers.LSTM or tf.keras.layers.LSTMCell + tf.keras.layers.RNN")
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="c0")
    if True:
        lstm    = tf.compat.v1.keras.layers.LSTM( \
                    nHiddenDim,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(mean=0,stddev=0.1),
                    recurrent_initializer=tf.compat.v1.truncated_normal_initializer(mean=0,stddev=0.1),
                    bias_initializer=tf.compat.v1.truncated_normal_initializer(mean=0,stddev=0.1),
                    unit_forget_bias=False,
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    recurrent_constraint=None,
                    bias_constraint=None,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    implementation=1,
                    return_sequences=True,
                    return_state=True,
                    go_backwards=False,
                    stateful=False,
                    unroll=False,
                    time_major=False
                    )
    else:
        cell    = tf.keras.layers.LSTMCell( \
                    nHiddenDim,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(mean=0,stddev=0.1),
                    recurrent_initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.1),
                    bias_initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=0.1),
                    unit_forget_bias=False,
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    kernel_constraint=None,
                    recurrent_constraint=None,
                    bias_constraint=None,
                    dropout=0.0,
                    recurrent_dropout=0.0
                    )
        lstm    = tf.keras.layers.RNN( \
                    cell,
                    return_sequences=True,
                    return_state=True,
                    go_backwards=False,
                    stateful=False,
                    unroll=False,
                    time_major=False
                    )
    y, h1, c1 = lstm(x, initial_state=[h0, c0])

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFh1, outputTFc1 = sess.run([y, h1, c1], feed_dict={x:inputX, h0: inputH, c0: inputC})

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test1.npz", **tfPara)
    sess.close()

    if True:
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
        inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
        inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
        profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
        profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
        profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
        config.add_optimization_profile(profile)

        para = np.load("test1.npz")
        weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], np.ascontiguousarray(i.reshape(-1))) for i in np.split(para["lstm/lstm_cell/kernel:0"], 4, axis=1)]
        weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], np.ascontiguousarray(i.reshape(-1))) for i in np.split(para["lstm/lstm_cell/recurrent_kernel:0"], 4, axis=1)]
        biasLayerList = [network.add_constant([1, nHiddenDim], np.ascontiguousarray(i.reshape(-1))) for i in np.split(para["lstm/lstm_cell/bias:0"], 4)]

        loop = network.add_loop()

        def gate(network, xTensor, wx, hTensor, wh, b, isSigmoid):
            _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
            _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
            _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
            _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
            _h4 = network.add_activation(_h3.get_output(0), trt.ActivationType.SIGMOID if isSigmoid else trt.ActivationType.TANH)
            return _h4

        _t0 = network.add_shape(inputT0)
        _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
        _t2 = network.add_shuffle(_t1.get_output(0))
        _t2.reshape_dims = ()
        loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
        iteratorLayer = loop.add_iterator(inputT0, 1, False)
        hiddenStateLayer = loop.add_recurrence(inputT1)
        cellStateLayer = loop.add_recurrence(inputT2)

        gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), True)
        gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), True)
        gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), False)
        gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), True)

        _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
        _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
        newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
        _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
        newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

        hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
        cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

        loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
        loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)
        loopOutput1.set_input(1, _t2.get_output(0))
        loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)

        network.mark_output(loopOutput0.get_output(0))
        network.mark_output(loopOutput1.get_output(0))
        network.mark_output(loopOutput2.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        context = engine.create_execution_context()
        context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
        context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
        context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
        _, stream = cudart.cudaStreamCreate()

        inputH0 = np.ascontiguousarray(inputX.reshape(-1))
        inputH1 = np.ascontiguousarray(inputH.reshape(-1))
        inputH2 = np.ascontiguousarray(inputC.reshape(-1))

        nIO = engine.num_io_tensors
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        bufferOutH = []
        for i in range(nInput, nIO):
            bufferOutH.append(np.empty(context.get_tensor_shape(lTensorName[i]),
                                    dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        outputH0 = bufferOutH[0]
        outputH1 = bufferOutH[1]
        outputH2 = bufferOutH[2]
        """
        outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
        outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
        outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
        """
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
        _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
        _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
        _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
        _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

        printArrayInformation(inputX, "x")
        print(inputX)
        printArrayInformation(inputH, "h0")
        print(inputH)
        printArrayInformation(inputC, "c0")
        print(inputC)
        printArrayInformation(outputTFh1, "TF h1")
        printArrayInformation(outputH0, "TRT h1")
        printArrayInformation(outputTFc1, "TF c1")
        printArrayInformation(outputH2, "TRT c1")
        printArrayInformation(outputTF, "TF AllOutput")
        printArrayInformation(outputH1, "TRT AllOutput")
        check(outputTFh1, outputH0, weak=True, name="h1")
        check(outputTFc1, outputH2, weak=True, name="c1")
        check(outputTF, outputH1, weak=True, name="AllOutput")

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(inputD1)
        cudart.cudaFree(inputD2)
        cudart.cudaFree(outputD0)
        cudart.cudaFree(outputD1)
        cudart.cudaFree(outputD2)


if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    np.set_printoptions(precision=3, linewidth=200, suppress=True)

    _test1()

    print("\ntest finish!")
