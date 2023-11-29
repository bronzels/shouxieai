import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval().to("cuda")

image = torch.randn(1, 3, 224, 224).to("cuda")
traced_model = torch.jit.trace(model, image)
model(image)
traced_model.save("./1/model.pt")
