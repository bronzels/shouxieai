import os
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

import time

onnxFile0 = "model-0-backup.onnx"
onnxFile1 = "model-1.onnx"
"""
# extract subgraph from wenet encoder, should not be used in this example, TODO: rewrite this part by ONNX
onnxFileS = "./encoder.onnx"

graph = gs.import_onnx(onnx.load(onnxFileS))

graph.outputs = []
for node in graph.nodes:

    if node.op == "Slice" and node.name == "Slice_74":
        table1x5000x256 = node.inputs[0].values
        constantData = gs.Constant("constantData", np.ascontiguousarray(table1x5000x256[:,:512,:]))  # keep only 512 elements to reduce the volume of the tensor
        inputTensor = node.inputs[2]
        inputTensor.name = "inputT0"
        graph.inputs = [inputTensor]

        node.inputs[0] = constantData

        for i in range(1, 24, 2):
            graph.outputs.append(node.o(i).o().o().outputs[0])  # Transpose
        continue

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxFile0)
"""

graph = gs.import_onnx(onnx.load(onnxFile0))

wiliConstant0 = gs.Constant("wiliConstant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
wiliConstant1 = gs.Constant("wiliConstant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
wiliConstant3 = gs.Constant("wiliConstant3", np.ascontiguousarray(np.array([3], dtype=np.int64)))

nSlice = 0
graph.outputs = []
for node in graph.nodes:
    if node.op == "Slice" and node.name == "Slice_74":
        table512x256 = node.inputs[0].values[0]
        for i in range(1, 24, 2):
            factor256x256 = node.o(i).inputs[1].values
            transposeNode = node.o(i).o().o()

            newTable = np.matmul(table512x256, factor256x256).transpose().reshape(1, 4, 64, 512)
            constantData = gs.Constant("wiliConstant-" + str(nSlice), np.ascontiguousarray(newTable))
            sliceV = gs.Variable(transposeNode.outputs[0].name, np.dtype(np.float32), [1, 4, 64, "t4"])
            sliceN = gs.Node(
                "Slice",
                "wiliSliceN-" + str(nSlice),
                inputs=[
                    constantData,  # data
                    wiliConstant0,  # start=0
                    graph.inputs[0],  # end
                    wiliConstant3,  # axes=3
                    wiliConstant1,  # step=1
                ],
                outputs=[sliceV]
            )
            graph.nodes.append(sliceN)
            graph.outputs.append(sliceV)
            nSlice += 1
            transposeNode.outputs = []
        continue

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxFile1)

def run(onnxFile):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 22 << 30

    parser = trt.OnnxParser(network, logger)
    with open(onnxFile, "rb") as model:
        parser.parse(model.read())

    inputT0 = network.get_input(0)
    inputT0.shape = [1]
    profile.set_shape_input(inputT0.name, [32], [32], [32])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    trtFile = onnxFile.split(".")[0] + ".plan"
    with open(trtFile, "wb") as f:
        f.write(engineString)

    print("Succeeded building %s!" % (trtFile))

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers --shapes=inputTensor:32" % trtFile)

a = time.time()
run(onnxFile0)
b = time.time()
print("run(onnxFile0) time = %f" % (b - a))
a = time.time()
run(onnxFile1)
b = time.time()
print("run(onnxFile1) time = %f" % (b - a))

"""
run(onnxFile0) time = 9.044213
run(onnxFile1) time = 7.304539
"""