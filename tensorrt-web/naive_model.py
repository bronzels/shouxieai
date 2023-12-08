import torch
import onnx

onnx_model = 'naive_model.onnx'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

model = NaiveModel()
x = torch.randn(1, 3, 224, 224)
print(x.shape)
y = model(x)
print(y.shape)
torch.onnx.export(model, x, onnx_model, input_names=['input'], output_names=['output'], opset_version=11)