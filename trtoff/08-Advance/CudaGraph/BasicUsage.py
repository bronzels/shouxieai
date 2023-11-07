import os

import numpy as np
import tensorrt as trt
from cuda import cudart

trtFile = "./model.plan"

def run():
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

        inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
        profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
        config.add_optimization_profile(profile)

        identityLayer = network.add_identity(inputTensor)
        network.mark_output(identityLayer.get_output(0))

        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building serialized engine!")
            return
        print("Succeeded building serialized engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    if engine == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
    #nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)

    context = engine.create_execution_context()
    # get a CUDA stream for CUDA graph and inference
    _, stream = cudart.cudaStreamCreate()

    context.set_input_shape(lTensorName[0], [3, 4, 5])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    # do inference before CUDA graph capture
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    context.execute_async_v3(stream)
    for i in range(nInput, nIO):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    # CUDA Graph capture
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    #for i in range(nIO):  # no need to reset the address if unchanged
    #    context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    context.execute_async_v3(stream)
    for i in range(nInput, nIO):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    #cudart.cudaStreamSynchronize(stream)  # no need to synchronize within the CUDA graph capture
    _, graph = cudart.cudaStreamEndCapture(stream)
    #_, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)  # for CUDA < 12
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)  # for CUDA >= 12

    # do inference with CUDA graph
    bufferH[1] *= 0
    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)

    # when the input shape changed, inference is also needed before CUDA graph capture
    context.set_input_shape(lTensorName[0], [2, 3, 4])

    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))  # set address of all input and output data in device buffer
    context.execute_async_v3(stream)
    for i in range(nInput, nIO):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    # CUDA Graph capture again
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    #for i in range(nIO):  # no need to reset the address if unchanged
    #    context.set_tensor_address(lTensorName[i], int(bufferD[i]))
    context.execute_async_v3(stream)
    for i in range(nInput, nIO):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    #cudart.cudaStreamSynchronize(stream)  # no need to synchronize within the CUDA graph capture
    _, graph = cudart.cudaStreamEndCapture(stream)
    #_, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)  # for CUDA < 12
    _, graphExe = cudart.cudaGraphInstantiate(graph, 0)  # for CUDA >= 12

    # do inference with CUDA graph
    bufferH[1] *= 0  # set output buffer as 0 to see the real output of inference
    cudart.cudaGraphLaunch(graphExe, stream)
    cudart.cudaStreamSynchronize(stream)

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)

    cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    cudart.cudaDeviceSynchronize()
    run()
    run()

"""
Succeeded building serialized engine!
Succeeded saving .plan file!
Succeeded building engine!
[ 0]Input -> DataType.FLOAT (-1, -1, -1) (3, 4, 5) inputT0
[ 1]Output-> DataType.FLOAT (-1, -1, -1) (3, 4, 5) (Unnamed Layer* 0) [Identity]_output
inputT0
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
(Unnamed Layer* 0) [Identity]_output
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
inputT0
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
(Unnamed Layer* 0) [Identity]_output
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
Succeeded getting serialized engine!
Succeeded building engine!
[ 0]Input -> DataType.FLOAT (-1, -1, -1) (3, 4, 5) inputT0
[ 1]Output-> DataType.FLOAT (-1, -1, -1) (3, 4, 5) (Unnamed Layer* 0) [Identity]_output
inputT0
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
(Unnamed Layer* 0) [Identity]_output
[[[ 0.  1.  2.  3.  4.]
  [ 5.  6.  7.  8.  9.]
  [10. 11. 12. 13. 14.]
  [15. 16. 17. 18. 19.]]

 [[20. 21. 22. 23. 24.]
  [25. 26. 27. 28. 29.]
  [30. 31. 32. 33. 34.]
  [35. 36. 37. 38. 39.]]

 [[40. 41. 42. 43. 44.]
  [45. 46. 47. 48. 49.]
  [50. 51. 52. 53. 54.]
  [55. 56. 57. 58. 59.]]]
inputT0
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
(Unnamed Layer* 0) [Identity]_output
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
"""