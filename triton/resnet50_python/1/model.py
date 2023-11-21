import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
import torchvision.models as models

"""
        self.model = (
            torch.hub.load(
                "pytorch/vision:v0.13.0",
                "resnet50",
                weights="IMAGENET1K_V2",
                skip_validation=True,
            )
            .to(self.device)
            .eval()
        )
用hub.load方式，gpu的权重总是下载超时出错
"""

class TritonPythonModel:
    def initialize(self, args):
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        self.model = models.resnet50(pretrained=True).to(self.device).eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            with torch.no_grad():
                result = self.model(
                    torch.as_tensor(input_tensor.as_numpy(), device=self.device)
                )
            out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT", to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
