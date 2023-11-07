
import os
from time import time

import numpy as np
import tensorrt as trt
from cuda import cudart

trtFile = "./model.plan"
np.random.seed(31193)

nWarmUp = 10
nTest = 30

# There are 3 scenarios of the inference
# 1. HtoD-bound

nB, nC, nH, nW = 8, 64, 256, 256
nCOut, nKernelHeight, nKernelWidth = 1, 3, 3

# 2. Calculation-bound
"""
nB,nC,nH,nW = 8,64,128,128
nCOut,nKernelHeight,nKernelWidth    = 64,9,9
"""
# 3. DtoH-bound
"""
nB,nC,nH,nW = 8,64,128,128
nCOut,nKernelHeight,nKernelWidth    = 256,3,3
"""

def getEngine():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputTensor = network.add_input("inputT0", trt.float32, [-1, nC, nH, nW])
        profile.set_shape(inputTensor.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
        config.add_optimization_profile(profile)

        w = np.ascontiguousarray(np.random.rand(nCOut, nC, nKernelHeight, nKernelWidth).astype(np.float32) * 2 - 1)
        b = np.ascontiguousarray(np.random.rand(nCOut).astype(np.float32) * 2 - 1)
        _0 = network.add_convolution_nd(inputTensor, nCOut, [nKernelHeight, nKernelWidth], trt.Weights(w), trt.Weights(b))
        _0.padding_nd = (nKernelHeight // 2, nKernelWidth // 2)
        _1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)

        network.mark_output(_1.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file!")

    return trt.Runtime(logger).deserialize_cuda_engine(engineString)

def run1(engine):
    context = engine.create_execution_context()
    context.set_binding_shape(0, [nB, nC, nH, nW])
    _, stream = cudart.cudaStreamCreate()

    data = np.random.rand(nB * nC * nH * nW).astype(np.float32).reshape(nB, nC, nH, nW)
    inputH0 = np.ascontiguousarray(data.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

   # Count time of memory copy from host to device
    for i in range(nWarmUp):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    trtTimeStart = time()
    for i in range(nTest):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, DataCopyHtoD" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    # Count time of inference
    for i in range(nWarmUp):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)

    trtTimeStart = time()
    for i in range(nTest):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, Inference" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    # Count time of memory copy from device to host
    for i in range(nWarmUp):
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    trtTimeStart = time()
    for i in range(nTest):
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, DataCopyDtoH" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    # Count time of end to end
    for i in range(nWarmUp):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)

    trtTimeStart = time()
    for i in range(nTest):
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(outputD0)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    trtTimeEnd = time()
    print("%6.3fms - 1 stream, DataCopy + Inference" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

def run2(engine):
    context = engine.create_execution_context()
    context.set_binding_shape(0, [nB, nC, nH, nW])
    _, stream0 = cudart.cudaStreamCreate()
    _, stream1 = cudart.cudaStreamCreate()
    _, event0 = cudart.cudaEventCreate()
    _, event1 = cudart.cudaEventCreate()

    data = np.random.rand(nB * nC * nH * nW).astype(np.float32).reshape(nB, nC, nH, nW)
    inputSize = trt.volume(context.get_binding_shape(0)) * np.array([0], dtype=trt.nptype(engine.get_binding_dtype(0))).nbytes
    outputSize = trt.volume(context.get_binding_shape(1)) * np.array([0], dtype=trt.nptype(engine.get_binding_dtype(1))).nbytes
    _, inputH0 = cudart.cudaHostAlloc(inputSize, cudart.cudaHostAllocWriteCombined)
    _, inputH1 = cudart.cudaHostAlloc(inputSize, cudart.cudaHostAllocWriteCombined)
    _, outputH0 = cudart.cudaHostAlloc(outputSize, cudart.cudaHostAllocWriteCombined)
    _, outputH1 = cudart.cudaHostAlloc(outputSize, cudart.cudaHostAllocWriteCombined)
    _, inputD0 = cudart.cudaMallocAsync(inputSize, stream0)
    _, inputD1 = cudart.cudaMallocAsync(inputSize, stream1)
    _, outputD0 = cudart.cudaMallocAsync(outputSize, stream0)
    _, outputD1 = cudart.cudaMallocAsync(outputSize, stream1)

    # Count time of end to end
    for i in range(nWarmUp):
        context.execute_async_v2([int(inputD0), int(outputD0)], stream0)

    trtTimeStart = time()
    cudart.cudaEventRecord(event1, stream1)

    for i in range(nTest):
        inputH, outputH = [inputH1, outputH1] if i & 1 else [inputH0, outputH0]
        inputD, outputD = [inputD1, outputD1] if i & 1 else [inputD0, outputD0]
        eventBefore, eventAfter = [event0, event1] if i & 1 else [event1, event0]
        stream = stream1 if i & 1 else stream0

        cudart.cudaMemcpyAsync(inputD, inputH, inputSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaStreamWaitEvent(stream, eventBefore, cudart.cudaEventWaitDefault)
        context.execute_async_v2([int(inputD), int(outputD)], stream)
        cudart.cudaEventRecord(eventAfter, stream)
        cudart.cudaMemcpyAsync(outputH, outputD, outputSize, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaEventSynchronize(event1)
    trtTimeEnd = time()
    print("%6.3fms - 2 stream, DataCopy + Inference" % ((trtTimeEnd - trtTimeStart) / nTest * 1000))

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    cudart.cudaDeviceSynchronize()
    engine = getEngine()
    run1(engine)
    run2(engine)

"""
13.515ms - 1 stream, DataCopyHtoD
 1.456ms - 1 stream, Inference
 0.233ms - 1 stream, DataCopyDtoH
15.174ms - 1 stream, DataCopy + Inference

5.791ms - 2 stream, DataCopy + Inference
"""

