from torch.onnx.symbolic_registry import register_op

def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

register_op('asinh', asinh_symbolic, '', 9)

import onnxruntime
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

model = Model()
input = torch.rand([1, 3, 10, 10])
torch_output = model(input).detach().numpy()
#print('torch_output: ', torch_output)

with torch.no_grad():
    torch.onnx.export(
        model,
        input,
        'asinh.onnx',
        opset_version=9,
        input_names=['input'],
        output_names=['output']
    )

sess = onnxruntime.InferenceSession('asinh.onnx')
ort_output = sess.run(None, {'input': input.numpy()})[0]
#print('ort_output: ', ort_output)

assert np.allclose(torch_output, ort_output)
