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

    #/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    def execute(self, requests):
        responses = []
        print("Batch Size Is:{}".format(len(requests)), flush=True)
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_tensor_is_cpu = input_tensor.is_cpu()
            print('input_tensor.is_cpu():{}'.format(str(input_tensor_is_cpu)), flush=True)
            #print("Batch Size:", input_tensor)
            #print(input_tensor.Dims()[0], flush=True)
            #print("input_tensor:", input_tensor)
            #print(dir(input_tensor), flush=True)
            #print("input_tensor.to_dlpack():", input_tensor.to_dlpack())
            #print(dir(input_tensor.to_dlpack()), flush=True)
            #print(dir(input_tensor.shape), flush=True)
            #print("Dim0 Size:{}".format(input_tensor.shape()[0]), flush=True)
            with torch.no_grad():
                if input_tensor_is_cpu:
                    result = self.model(
                        torch.as_tensor(input_tensor.as_numpy(), device=self.device)
                    )
                else:
                    result = self.model(
                        torch.from_dlpack(input_tensor.to_dlpack())
                    )
            out_tensor = pb_utils.Tensor.from_dlpack("OUTPUT", to_dlpack(result))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
