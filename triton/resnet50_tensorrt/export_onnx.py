import torch
import torchvision.models as models

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = models.resnet50(pretrained=True).eval()
x = torch.randn(1, 3, 224, 224, requires_grad=True)

torch.onnx.export(model,
                  x,
                  "resnet50.onnx",
                  export_params=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  )