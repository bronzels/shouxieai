import triton_python_backend_utils as pb_utils
import onnxruntime
import json
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type']
        )
        model_directory = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(model_directory, 'model.onnx')
        if not os.path.exists(model_path):
            raise pb_utils.TritonModeException("Can't find the onnx model")
        self.session = onnxruntime.InferenceSession(model_path)
        print('Initialized...')

    def execute(self, requests):
        output_dtype = self.output_dtype
        responses = []
        print("Batch Size Is:{}".format(len(requests)), flush=True)
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            print('input_tensor.is_cpu():{}'.format(str(input_tensor.is_cpu())), flush=True)
            input_array = input_tensor.as_numpy()
            if input_array.shape[2] > 1000 or input_array.shape[3] > 1000:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError("Image shape should not be larger than 1000")))

            prediction = self.session.run(None, {"input": input_array})
            #print("len(prediction):{}".format(len(prediction)), flush=True)
            #print("len(prediction[0]):{}".format(len(prediction[0])), flush=True)
            #print("len(prediction[0][0]):{}".format(len(prediction[0][0])), flush=True)
            #out_tensor = pb_utils.Tensor("OUTPUT", prediction[0].astype(output_dtype))
            #message: AttributeError: 'list' object has no attribute 'astype'
            out_tensor = pb_utils.Tensor("OUTPUT", prediction[0])

            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        print('Cleaning up...')