import json

import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "model_2_input_string")
            inp2 = pb_utils.get_input_tensor_by_name(
                request, "model_2_input_UINT8_array"
            )
            inp3 = pb_utils.get_input_tensor_by_name(
                request, "model_2_input_INT8_array"
            )
            inp4 = pb_utils.get_input_tensor_by_name(
                request, "model_2_input_FP32_image"
            )
            inp5 = pb_utils.get_input_tensor_by_name(request, "model_2_input_bool")

            print("Model 2 received", flush=True)
            print(inp.as_numpy(), flush=True)
            print(inp2.as_numpy(), flush=True)
            print(inp3.as_numpy(), flush=True)
            print(inp4.as_numpy(), flush=True)
            print(inp5.as_numpy(), flush=True)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "model_2_output_string",
                        inp.as_numpy(),
                    ),
                    pb_utils.Tensor(
                        "model_2_output_UINT8_array",
                        inp2.as_numpy(),
                    ),
                    pb_utils.Tensor(
                        "model_2_output_INT8_array",
                        inp3.as_numpy(),
                    ),
                    pb_utils.Tensor(
                        "model_2_output_FP32_image",
                        inp4.as_numpy(),
                    ),
                    pb_utils.Tensor(
                        "model_2_output_bool",
                        inp5.as_numpy(),
                    ),
                ]
            )
            responses.append(inference_response)
        return responses
