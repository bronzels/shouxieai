import json
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            model_name = pb_utils.get_input_tensor_by_name(request, "MODEL_NAME")
            model_name_string = model_name.as_numpy()[0]

            infer_request = pb_utils.InferenceRequest(
                model_name=model_name_string,
                requested_output_names=["OUTPUT0", "OUTPUT1"],
                inputs=[in_0, in_1],
            )

            infer_response = infer_request.exec()

            if infer_response.has_error():
                raise pb_utils.TritonModelException(infer_response.error().message())

            inference_response = pb_utils.InferenceResponse(
                output_tensors=infer_response.output_tensors()
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")


