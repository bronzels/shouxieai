import torch
import my_lib

class MyAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    @staticmethod
    def symbolic(g, a, b):
        two = g.op("Constant", value_t=torch.tensor([2]))
        a = g.op('Mul', a, two)
        return g.op('Add', a, b)

my_add = MyAddFunction.apply

class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)

model = MyAdd()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, (input, input), 'my_add.onnx')
torch_output = model(input, input).detach().numpy()
#print('torch_output:', torch_output)

import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('my_add.onnx')
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0]
#print('ort_output:', ort_output)

assert np.allclose(torch_output, ort_output)