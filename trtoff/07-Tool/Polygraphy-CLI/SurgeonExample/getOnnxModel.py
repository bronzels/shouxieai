from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs

def addNode(graph, nodeType, prefix, number, inputList, attribution=None, suffix="", dtype=None, shape=None):
    tensorName = prefix + "-V-" + str(number) + "-" + nodeType
    nodeName = prefix + "-N-" + str(number) + "-" + nodeType
    if attribution == None:
        attribution = OrderedDict()
    if len(suffix) > 0:
        tensorName += "-" + suffix

    tensor = gs.Variable(tensorName, dtype, shape)
    node = gs.Node(nodeType, nodeName, inputs=inputList, outputs=[tensor], attrs=attribution)
    graph.nodes.append(node)
    return tensor, number + 1

# ModelA -----------------------------------------------------------------------
graph = gs.Graph(nodes=[], inputs=[], outputs=[])

batchName = 12

tensorX = gs.Variable("tensorX", np.float32, [batchName, 2, 3, 4])
constant0C1 = gs.Constant("constant0C1", np.ascontiguousarray(np.array([0, 1], dtype=np.int64)))
constant2C3 = gs.Constant("constant2C3", np.ascontiguousarray(np.array([2, 3], dtype=np.int64)))

n = 0
scopeName = "A"

tensor1, n = addNode(graph, "Shape", scopeName, n, [tensorX], None, "", np.int64, [4])

tensor2, n = addNode(graph, "ReduceProd", scopeName, n, [tensor1], OrderedDict([["axes", [0]], ["keepdims", 1]]), "", np.int64, [1])

tensor3, n = addNode(graph, "Reshape", scopeName, n, [tensorX, tensor2], None, "", np.float32, ["%s*24" % str(batchName)])

tensor4, n = addNode(graph, "Gather", scopeName, n, [tensor1, constant0C1], None, "", np.int64, [2])

tensor5, n = addNode(graph, "Gather", scopeName, n, [tensor1, constant2C3], None, "", np.int64, [2])

tensor6, n = addNode(graph, "ReduceProd", scopeName, n, [tensor5], OrderedDict([["axes", [0]], ["keepdims", 1]]), "", np.int64, [1])

tensor7, n = addNode(graph, "Concat", scopeName, n, [tensor4, tensor6], OrderedDict([["axis", 0]]), "", np.int64, [4])

tensor8, n = addNode(graph, "Reshape", scopeName, n, [tensorX, tensor7], None, "", np.float32, [batchName, 2, 12])

graph.inputs = [tensorX]
graph.outputs = [tensor3, tensor8]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "modelA.onnx")
print("Succeeded create %s" % "modelA.onnx")