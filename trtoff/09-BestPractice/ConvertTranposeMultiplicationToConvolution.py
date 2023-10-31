import os
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

onnxFile0 = "model-0.onnx-backup"
onnxFile1 = "model-1.onnx"

if True:  # model bindind parameters, not change
    t0 = 19
    t1 = 256
    t2 = 256

nBS = 16
nSL = 64

graph = gs.import_onnx(onnx.load(onnxFile0))

for node in graph.nodes:
    if node.op == "MatMul" and node.name == "MatMul_61":
        convKernel = node.inputs[1].values.transpose(1, 0).reshape(256, t1, 1, t0).astype(np.float32)
        convKernelV = gs.Constant("ConvKernelV", np.ascontiguousarray(convKernel))
        continue

    if node.op == "Add" and node.name == "Add_62":
        convBias = node.inputs[0].values
        convBiasV = gs.Constant("ConvBiasV", np.ascontiguousarray(convBias))
        continue

convV = gs.Variable("ConvV", np.dtype(np.float32), ["B", t1, "t4", 1])
convN = gs.Node("Conv", "ConvN", inputs=[graph.inputs[0], convKernelV, convBiasV], outputs=[convV])
convN.attrs = OrderedDict([
    ("dilations", [1, 1]),
    ("kernel_shape", [1, t0]),
    ("pads", [0, 0, 0, 0]),
    ("strides", [1, 1]),
])
graph.nodes.append(convN)

constant3 = gs.Constant("constant3", np.ascontiguousarray(np.array([3], dtype=np.int64)))

squeezeV = gs.Variable("SqueezeV", np.dtype(np.float32), ["B", t2, "t4"])
squeezeN = gs.Node("Squeeze", "SqueezeN", inputs=[convV, constant3], outputs=[squeezeV])
graph.nodes.append(squeezeN)

transposeV = gs.Variable("TransposeV", np.dtype(np.float32), ["B", "t4", t2])
transposeN = gs.Node("Transpose", "TransposeN", inputs=[squeezeV], outputs=[transposeV], attrs=OrderedDict([("perm", [0, 2, 1])]))
graph.nodes.append(transposeN)

graph.outputs = [transposeV]

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxFile1)


def run(onnxFile):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, logger)
    with open(onnxFile, "rb") as model:
        parser.parse(model.read())

    inputT0 = network.get_input(0)
    inputT0.shape = [-1, t1, -1, t0]
    profile.set_shape(inputT0.name, [1, t1, 1, t0], [nBS, t1, nSL, t0], [nBS, t1, nSL, t0])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    trtFile = onnxFile.split(".")[0] + ".plan"
    with open(trtFile, "wb") as f:
        f.write(engineString)

    print("Succeeded building %s!" % (trtFile))

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers --shapes=inputTensor:%dx%dx%dx%d" % (trtFile, nBS, t1, nSL, t0))

run(onnxFile0)
run(onnxFile1)

"""
[10/30/2023-15:53:05] [I] === Performance summary ===
[10/30/2023-15:53:05] [I] Throughput: 2702.39 qps
[10/30/2023-15:53:05] [I] Latency: min = 0.364502 ms, max = 0.391174 ms, mean = 0.36848 ms, median = 0.366699 ms, percentile(90%) = 0.372681 ms, percentile(95%) = 0.373779 ms, percentile(99%) = 0.387085 ms
[10/30/2023-15:53:05] [I] Enqueue Time: min = 0.0078125 ms, max = 0.0350037 ms, mean = 0.00867348 ms, median = 0.00854492 ms, percentile(90%) = 0.00939941 ms, percentile(95%) = 0.00973511 ms, percentile(99%) = 0.0125427 ms
[10/30/2023-15:53:05] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/30/2023-15:53:05] [I] GPU Compute Time: min = 0.364502 ms, max = 0.391174 ms, mean = 0.36848 ms, median = 0.366699 ms, percentile(90%) = 0.372681 ms, percentile(95%) = 0.373779 ms, percentile(99%) = 0.387085 ms
[10/30/2023-15:53:05] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/30/2023-15:53:05] [I] Total Host Walltime: 3.00068 s
[10/30/2023-15:53:05] [I] Total GPU Compute Time: 2.98801 s
[10/30/2023-15:53:05] [I] Explanations of the performance metrics are printed in the verbose logs.

[10/30/2023-15:53:13] [I] === Performance summary ===
[10/30/2023-15:53:13] [I] Throughput: 2067.13 qps
[10/30/2023-15:53:13] [I] Latency: min = 0.480225 ms, max = 0.489471 ms, mean = 0.482211 ms, median = 0.4823 ms, percentile(90%) = 0.482422 ms, percentile(95%) = 0.485382 ms, percentile(99%) = 0.48642 ms
[10/30/2023-15:53:13] [I] Enqueue Time: min = 0.00463867 ms, max = 0.0129395 ms, mean = 0.00519538 ms, median = 0.00512695 ms, percentile(90%) = 0.00561523 ms, percentile(95%) = 0.00610352 ms, percentile(99%) = 0.0065918 ms
[10/30/2023-15:53:13] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/30/2023-15:53:13] [I] GPU Compute Time: min = 0.480225 ms, max = 0.489471 ms, mean = 0.482211 ms, median = 0.4823 ms, percentile(90%) = 0.482422 ms, percentile(95%) = 0.485382 ms, percentile(99%) = 0.48642 ms
[10/30/2023-15:53:13] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/30/2023-15:53:13] [I] Total Host Walltime: 3.00078 s
[10/30/2023-15:53:13] [I] Total GPU Compute Time: 2.99115 s
[10/30/2023-15:53:13] [I] Explanations of the performance metrics are printed in the verbose logs.
"""
