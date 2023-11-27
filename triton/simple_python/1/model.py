import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        print("Batch Size Is:{}".format(len(requests)), flush=True)
        for request in requests:
            input_tensor0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_tensor1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            output_tensor0 = pb_utils.Tensor("OUTPUT0", input_tensor0.as_numpy() + input_tensor1.as_numpy())
            output_tensor1 = pb_utils.Tensor("OUTPUT1", input_tensor0.as_numpy() - input_tensor1.as_numpy())
            responses.append(pb_utils.InferenceResponse([output_tensor0, output_tensor1]))
        return responses
