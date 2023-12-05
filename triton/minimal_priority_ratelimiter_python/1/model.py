import triton_python_backend_utils as pb_utils
import json
import time

class TritonPythonModel:
    def initialize(self, args):
        model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        print("Batch Size Is:{}".format(len(requests)), flush=True)
        for request in requests:
            input_tensor0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            output_tensor0 = pb_utils.Tensor("OUTPUT0", input_tensor0.as_numpy().astype(self.output0_dtype))
            responses.append(pb_utils.InferenceResponse([output_tensor0]))
        time.sleep(1)
        return responses
