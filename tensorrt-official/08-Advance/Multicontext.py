import numpy as np
import tensorrt as trt
from cuda import cudart

shape = [2, 3, 4, 5]
nContext = 2
np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
inputT1 = network.add_input("inputT1", trt.float32, [-1, -1, -1, -1])
layer = network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.SUM)
network.mark_output(layer.get_output(0))

profile.set_shape(inputT0.name, shape, shape, [k * nContext for k in shape])
profile.set_shape(inputT1.name, shape, shape, [k * nContext for k in shape])
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
nOutput = nIO - nInput

streamList = [cudart.cudaStreamCreate()[1] for _ in range(nContext)]
contextList = [engine.create_execution_context() for _ in range(nContext)]

for index in range(nContext):
    stream = streamList[index]
    context = contextList[index]
    context.set_optimization_profile_async(0, stream)
    inputShape = [k * (index + 1) for k in shape]
    context.set_input_shape(lTensorName[0], inputShape)
    context.set_input_shape(lTensorName[1], inputShape)
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])
    bufferH = []
    for i in range(nInput):
        bufferH.append(np.arange(np.prod(inputShape)).astype(np.float32).reshape(inputShape))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[nInput + i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[nInput + i]))))

    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(stream)


    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)

    print("check result of context %d: %s" % (index, np.all(bufferH[2] == bufferH[0] + bufferH[1])))

    for b in bufferD:
        cudart.cudaFree(b)

"""
[ 0]Input -> DataType.FLOAT (-1, -1, -1, -1) (2, 3, 4, 5) inputT0
[ 1]Input -> DataType.FLOAT (-1, -1, -1, -1) (2, 3, 4, 5) inputT1
[ 2]Output-> DataType.FLOAT (-1, -1, -1, -1) (2, 3, 4, 5) (Unnamed Layer* 0) [ElementWise]_output
check result of context 0: True
[ 0]Input -> DataType.FLOAT (-1, -1, -1, -1) (4, 6, 8, 10) inputT0
[ 1]Input -> DataType.FLOAT (-1, -1, -1, -1) (4, 6, 8, 10) inputT1
[ 2]Output-> DataType.FLOAT (-1, -1, -1, -1) (4, 6, 8, 10) (Unnamed Layer* 0) [ElementWise]_output
check result of context 1: True
"""