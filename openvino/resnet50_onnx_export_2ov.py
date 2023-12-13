import torch
import torchvision
from openvino.tools.mo import convert_model
from openvino.runtime import serialize
import numpy as np

onnx_file_path_static="resnet50-fp16-static.onnx"
onnx_file_path_dynamic="resnet50-fp16-dynamic.onnx"

model = torchvision.models.resnet50(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
#dummy_input = torch.rand((1, 3, 224, 224))
#都可以

#input_names = ["image"],
#output_names = ["prob"],
print("---static start---")
#都可以
#torch.onnx.export(model, dummy_input, onnx_file_path_static,
torch.onnx.export(model, (dummy_input,), onnx_file_path_static,
                  verbose=True,
                  opset_version=11)
ov_model = convert_model(onnx_file_path_static,
                         reverse_input_channels=True,
                         compress_to_fp16=True,
                         mean_values=np.multiply([0.485,0.456,0.406],255),
                         scale_values=np.multiply([0.229,0.224,0.225],255)
                         )
"""
mo --input_model resnet50-static.onnx \
   --reverse_input_channels \
   --mean_values=[123.675,116.28,103.53] \
   --scale_values=[58.395,57.12,57.375] \
   --output_dir ./\
   --model_name resnet_fr_torch_onnx-static
"""
#ov_model = convert_model(onnx_file_path_static)
#, example_input=torch.zeros(1, 3, 224, 224)
serialize(ov_model, 'resnet_fr_torch_onnx-fp16-static.xml')
print("---static end---")

print("---dynamic start---")
#都可以
#torch.onnx.export(model, dummy_input, onnx_file_path_dynamic,
torch.onnx.export(model, (dummy_input,), onnx_file_path_dynamic,
                  input_names=["image"],
                  output_names=["prob"],
                  verbose=True,
                  dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}},
                  opset_version=11)
ov_model = convert_model(onnx_file_path_static,
                         reverse_input_channels=True,
                         compress_to_fp16=True,
                         mean_values=np.multiply([0.485,0.456,0.406],255),
                         scale_values=np.multiply([0.229,0.224,0.225],255)
                         )
#ov_model = convert_model(onnx_file_path_static)
#, example_input=torch.zeros(1, 3, 224, 224)
serialize(ov_model, 'resnet_fr_torch_onnx-fp16-dynamic.xml')
print("---dynamic end---")



