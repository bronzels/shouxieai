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
profileList = [builder.create_optimization_profile() for _ in range(nContext)]
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
inputT1 = network.add_input("inputT1", trt.float32, [-1, -1, -1, -1])
layer = network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.SUM)
network.mark_output(layer.get_output(0))

for profile in profileList:
    profile.set_shape(inputT0.name, shape, shape, [k * nContext for k in shape])
    profile.set_shape(inputT1.name, shape, shape, [k * nContext for k in shape])
    config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_bindings
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = nIO - nInput
nIO, nInput, nOutput = nIO // nContext, nInput // nContext, nOutput // nContext

streamList = [cudart.cudaStreamCreate()[1] for _ in range(nContext)]
contextList = [engine.create_execution_context() for index in range(nContext)]

bufferH = []
for index in range(nContext):
    stream = streamList[index]
    context = contextList[index]
    context.set_optimization_profile_async(index, stream)
    bindingPad = nIO * index
    inputShape = [k * (index + 1) for k in shape]
    context.set_binding_shape(bindingPad + 0, inputShape)
    context.set_binding_shape(bindingPad + 1, inputShape)
    print("Context%d binding all? %s" % (index, "Yes" if context.all_binding_shapes_specified else "No"))
    for i in range(nIO):
        print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))
    for i in range(nInput):
        bufferH.append(np.arange(np.prod(inputShape)).astype(np.float32).reshape(inputShape))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(bindingPad + nInput + i), dtype=trt.nptype(engine.get_binding_dtype(bindingPad + nInput + i))))

bufferD = []
for i in range(len(bufferH)):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for index in range(nContext):
    print("Use Context %d" % index)
    stream = streamList[index]
    context = contextList[index]
    context.set_optimization_profile_async(index, stream)
    bindingPad = nIO * index
    inputShape = [k * (index + 1) for k in shape]
    context.set_binding_shape(bindingPad + 0, inputShape)
    context.set_binding_shape(bindingPad + 1, inputShape)
    for i in range(nIO * nContext):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[bindingPad + i], bufferH[bindingPad + i].ctypes.data, bufferH[bindingPad + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    bufferList = [int(0) for b in bufferD[:bindingPad]] + [int(b) for b in bufferD[bindingPad:(bindingPad + nInput + nOutput)]] + [int(0) for b in bufferD[(bindingPad + nInput + nOutput):]]

    context.execute_async_v2(bufferList, stream)

    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[bindingPad + nInput + i].ctypes.data, bufferD[bindingPad + nInput + i], bufferH[bindingPad + nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

for index in range(nContext):
    stream = streamList[index]
    cudart.cudaStreamSynchronize(stream)

for index in range(nContext):
    bindingPad = nIO * index
    print("check result of context %d: %s" % (index, np.all(bufferH[bindingPad + 2] == bufferH[bindingPad + 0] + bufferH[bindingPad + 1])))

for stream in streamList:
    cudart.cudaStreamDestroy(stream)

for b in bufferD:
    cudart.cudaFree(b)

"""
Context0 binding all? Yes
0 Input  (-1, -1, -1, -1) (2, 3, 4, 5)
1 Input  (-1, -1, -1, -1) (2, 3, 4, 5)
2 Output (-1, -1, -1, -1) (2, 3, 4, 5)
Context1 binding all? Yes
0 Input  (-1, -1, -1, -1) (-1, -1, -1, -1)
1 Input  (-1, -1, -1, -1) (-1, -1, -1, -1)
2 Output (-1, -1, -1, -1) (-1, -1, -1, -1)
Use Context 0
[ 0]Input -> DataType.FLOAT (-1, -1, -1, -1) (2, 3, 4, 5) inputT0
[ 1]Input -> DataType.FLOAT (-1, -1, -1, -1) (2, 3, 4, 5) inputT1
[ 2]Output-> DataType.FLOAT (-1, -1, -1, -1) (2, 3, 4, 5) (Unnamed Layer* 0) [ElementWise]_output
[ 3]Output-> DataType.FLOAT (-1, -1, -1, -1) (-1, -1, -1, -1) inputT0 [profile 1]
[ 4]Output-> DataType.FLOAT (-1, -1, -1, -1) (-1, -1, -1, -1) inputT1 [profile 1]
[ 5]Output-> DataType.FLOAT (-1, -1, -1, -1) (-1, -1, -1, -1) (Unnamed Layer* 0) [ElementWise]_output [profile 1]
Use Context 1
[ 0]Input -> DataType.FLOAT (-1, -1, -1, -1) (-1, -1, -1, -1) inputT0
[ 1]Input -> DataType.FLOAT (-1, -1, -1, -1) (-1, -1, -1, -1) inputT1
[ 2]Output-> DataType.FLOAT (-1, -1, -1, -1) (-1, -1, -1, -1) (Unnamed Layer* 0) [ElementWise]_output
[ 3]Output-> DataType.FLOAT (-1, -1, -1, -1) (4, 6, 8, 10) inputT0 [profile 1]
[ 4]Output-> DataType.FLOAT (-1, -1, -1, -1) (4, 6, 8, 10) inputT1 [profile 1]
[ 5]Output-> DataType.FLOAT (-1, -1, -1, -1) (4, 6, 8, 10) (Unnamed Layer* 0) [ElementWise]_output [profile 1]
check result of context 0: True
check result of context 1: True
"""