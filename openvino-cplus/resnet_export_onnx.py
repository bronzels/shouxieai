import torch
import torchvision
import cv2
import numpy as np


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        feature     = self.backbone(x)
        probability = torch.softmax(feature, dim=1)
        return probability

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

image = cv2.imread("car.jpeg")
image = cv2.resize(image, (224, 224))
image = image[..., ::-1]
image = image / 255.0
image = (image - imagenet_mean) / imagenet_std
image = image.astype(np.float32)
image = image.transpose(2, 0, 1)
image = np.ascontiguousarray(image)
image = image[None, ...]
image = torch.from_numpy(image)
model = Classifier().eval()
#model = torchvision.models.resnet50(pretrained=True).eval()

with torch.no_grad():
    probability = model(image)

predict_class = probability.argmax(dim=1).item()
confidence    = probability[0, predict_class]

labels = open("labels-imagenet.txt").readlines()
labels = [item.strip() for item in labels]

print(f"Predict: {predict_class}, {confidence}, {labels[predict_class]}")

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    model, (dummy,), "resnet50-dynamic.onnx",
    input_names=["image"],
    output_names=["prob"],
    dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "prob": {0: "batch"}},
    opset_version=11
)
