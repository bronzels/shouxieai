import os
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

nLoop = 10
nC = 32
onnxFile0 = "model-0.onnx"
onnxFile1 = "model-1.onnx"

tensor0 = gs.Variable("tensor-0", np.float32, ["B", 1, 16, 16])
constant32x1 = gs.Constant("constant32x1", np.ascontiguousarray(np.random.rand(nC, 1, 3, 3).reshape(nC, 1, 3, 3).astype(np.float32) * 2 - 1))
constant32x32 = gs.Constant("constant32x32", np.ascontiguousarray(np.random.rand(nC, nC, 3, 3).reshape(nC, nC, 3, 3).astype(np.float32) * 2 - 1))
constant32 = gs.Constant("constant32", np.ascontiguousarray(np.random.rand(1, nC, 1, 1).reshape(1, nC, 1, 1).astype(np.float32) * 2 - 1))
constant32t = gs.Constant("constant32t", np.ascontiguousarray(np.random.rand(1, 1, 1, nC).reshape(1, 1, 1, nC).astype(np.float32) * 2 - 1))
constant1x32 = gs.Constant("constant1x32", np.ascontiguousarray(np.random.rand(1, nC, 3, 3).reshape(1, nC, 3, 3).astype(np.float32) * 2 - 1))
constant1 = gs.Constant("constant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
constant32r = gs.Constant("constant32r", np.ascontiguousarray(np.random.rand(1, nC, 1, 1).reshape(1, nC, 1, 1).astype(np.float32) * 2 - 1))

graphNodeList = []

tensor1 = gs.Variable("tensor-1", np.float32, None)
node1 = gs.Node("Conv", "Conv0", inputs=[tensor0, constant32x1], outputs=[tensor1])
node1.attrs = OrderedDict([("kernel_shape", [3, 3]), ("pads", [1, 1, 1, 1])])
graphNodeList.append(node1)

tensorLoop = tensor1
for i in range(nLoop // 2):
    tensor2 = gs.Variable("tensor-%d-1" % i, np.float32, None)
    node2 = gs.Node("Conv", "Conv-" + str(i), inputs=[tensorLoop, constant32x32], outputs=[tensor2])
    node2.attrs = OrderedDict([("kernel_shape", [3, 3]), ("pads", [1, 1, 1, 1])])
    graphNodeList.append(node2)

    tensor3 = gs.Variable("tensor-%d-2" % i, dtype=np.float32, shape=None)
    node3 = gs.Node("Unsqueeze", "Unsqueeze-" + str(i), inputs=[tensor2, constant1], outputs=[tensor3])
    graphNodeList.append(node3)

    tensor4 = gs.Variable("tensor-%d-3" % i, dtype=np.float32, shape=None)
    node4 = gs.Node("Add", "Add-" + str(i), inputs=[tensor3, constant32], outputs=[tensor4])
    graphNodeList.append(node4)

    tensor5 = gs.Variable("tensor-%d-4" % i, dtype=np.float32, shape=None)
    node5 = gs.Node("Squeeze", "Squeeze-" + str(i), inputs=[tensor4, constant1], outputs=[tensor5])
    graphNodeList.append(node5)

    tensor6 = gs.Variable("tensor-%d-5" % i, dtype=np.float32, shape=None)
    node6 = gs.Node("Relu", "Relu-" + str(i), inputs=[tensor5], outputs=[tensor6])
    graphNodeList.append(node6)

    tensorLoop = tensor6

for i in range(nLoop // 2, nLoop):
    tensor2 = gs.Variable("tensor-%d-1" % i, np.float32, None)
    node2 = gs.Node("Conv", "Conv-" + str(i), inputs=[tensorLoop, constant32x32], outputs=[tensor2])
    node2.attrs = OrderedDict([("kernel_shape", [3, 3]), ("pads", [1, 1, 1, 1])])
    graphNodeList.append(node2)

    tensor3 = gs.Variable("tensor-%d-2" % i, dtype=np.float32, shape=None)
    node3 = gs.Node("Transpose", "Transpose-" + str(i), inputs=[tensor2], outputs=[tensor3], attrs=OrderedDict([("perm", [0, 2, 3, 1])]))
    graphNodeList.append(node3)

    tensor4 = gs.Variable("tensor-%d-3" % i, dtype=np.float32, shape=None)
    node4 = gs.Node("Add", "Add-" + str(i), inputs=[tensor3, constant32t], outputs=[tensor4])
    graphNodeList.append(node4)

    tensor5 = gs.Variable("tensor-%d-4" % i, dtype=np.float32, shape=None)
    node5 = gs.Node("Transpose", "Transpose-" + str(i), inputs=[tensor4], outputs=[tensor5], attrs=OrderedDict([("perm", [0, 3, 1, 2])]))
    graphNodeList.append(node5)

    tensor6 = gs.Variable("tensor-%d-5" % i, dtype=np.float32, shape=None)
    node6 = gs.Node("Relu", "Relu-" + str(i), inputs=[tensor5], outputs=[tensor6])
    graphNodeList.append(node6)

    tensorLoop = tensor6

tensor7 = gs.Variable("tensor7", dtype=np.float32, shape=None)
node7 = gs.Node("Conv", "Conv1", inputs=[tensorLoop, constant1x32], outputs=[tensor7])
graphNodeList.append(node7)

graph = gs.Graph(nodes=graphNodeList, inputs=[tensor0], outputs=[tensor7], opset=13)

onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile0)
print("Succeeded building %s!" % (onnxFile0))

# Remove pairs of Transpose or Squeeze/Unsqueeze ndoes
graph = gs.import_onnx(onnx.load(onnxFile0))

for node in graph.nodes:
    if node.op in ["Squeeze", "Unsqueeze"]:
        node.o().inputs[0] = node.inputs[0]

    if node.op == "Transpose":
        if node.o().op == "Add":
            node.o().inputs[1] = constant32r
        node.o().inputs[0] = node.inputs[0]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxFile1)
print("Succeeded building %s!" % (onnxFile1))

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
    inputT0.shape = [-1, 1, 16, 16]
    profile.set_shape(inputT0.name, [1, 1, 16, 16], [8, 1, 16, 16], [8, 1, 16, 16])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    trtFile = onnxFile.split(".")[0] + ".plan"
    with open(trtFile, "wb") as f:
        f.write(engineString)
    print("Succeeded building %s!" % (trtFile))

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers --shapes=tensor-0:8x1x16x16" % trtFile)

run(onnxFile0)
run(onnxFile1)

"""
[10/31/2023-15:06:14] [I] === Performance summary ===
[10/31/2023-15:06:14] [I] Throughput: 8208.49 qps
[10/31/2023-15:06:14] [I] Latency: min = 0.118652 ms, max = 0.180206 ms, mean = 0.120218 ms, median = 0.119873 ms, percentile(90%) = 0.12085 ms, percentile(95%) = 0.12085 ms, percentile(99%) = 0.12085 ms
[10/31/2023-15:06:14] [I] Enqueue Time: min = 0.00170898 ms, max = 0.0882263 ms, mean = 0.00756336 ms, median = 0.00854492 ms, percentile(90%) = 0.00891113 ms, percentile(95%) = 0.0090332 ms, percentile(99%) = 0.00976562 ms
[10/31/2023-15:06:14] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/31/2023-15:06:14] [I] GPU Compute Time: min = 0.118652 ms, max = 0.180206 ms, mean = 0.120218 ms, median = 0.119873 ms, percentile(90%) = 0.12085 ms, percentile(95%) = 0.12085 ms, percentile(99%) = 0.12085 ms
[10/31/2023-15:06:14] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/31/2023-15:06:14] [I] Total Host Walltime: 3.00018 s
[10/31/2023-15:06:14] [I] Total GPU Compute Time: 2.9606 s
[10/31/2023-15:06:14] [I] Explanations of the performance metrics are printed in the verbose logs.

[10/30/2023-18:24:31] [I] === Performance summary ===
[10/30/2023-18:24:31] [I] Throughput: 9582.57 qps
[10/30/2023-18:24:31] [I] Latency: min = 0.101318 ms, max = 0.109619 ms, mean = 0.102722 ms, median = 0.102417 ms, percentile(90%) = 0.103455 ms, percentile(95%) = 0.103516 ms, percentile(99%) = 0.104462 ms
[10/30/2023-18:24:31] [I] Enqueue Time: min = 0.000976562 ms, max = 0.0148926 ms, mean = 0.00570865 ms, median = 0.0057373 ms, percentile(90%) = 0.00610352 ms, percentile(95%) = 0.00610352 ms, percentile(99%) = 0.00692749 ms
[10/30/2023-18:24:31] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/30/2023-18:24:31] [I] GPU Compute Time: min = 0.101318 ms, max = 0.109619 ms, mean = 0.102722 ms, median = 0.102417 ms, percentile(90%) = 0.103455 ms, percentile(95%) = 0.103516 ms, percentile(99%) = 0.104462 ms
[10/30/2023-18:24:31] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/30/2023-18:24:31] [I] Total Host Walltime: 3.00024 s
[10/30/2023-18:24:31] [I] Total GPU Compute Time: 2.95327 s
[10/30/2023-18:24:31] [I] Explanations of the performance metrics are printed in the verbose logs.
"""