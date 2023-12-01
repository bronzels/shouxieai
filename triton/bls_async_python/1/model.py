import asyncio
import json

import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    async def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            inference_response_awaits = []
            for model_name in ["simple_python", "add_sub_python"]:
                infer_request = pb_utils.InferenceRequest(
                    model_name=model_name,
                    requested_output_names=["OUTPUT0", "OUTPUT1"],
                    inputs=[in_0, in_1],
                )
                inference_response_awaits.append(infer_request.async_exec())

            inference_responses = await asyncio.gather(*inference_response_awaits)

            for infer_response in inference_responses:
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_response.error().message()
                    )

            simple_output0_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[0], "OUTPUT0"
            )

            addsub_output1_tensor = pb_utils.get_output_tensor_by_name(
                inference_responses[1], "OUTPUT1"
            )

            inference_response = pb_utils.InferenceResponse(
                    output_tensors=[simple_output0_tensor, addsub_output1_tensor]
                )
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")


