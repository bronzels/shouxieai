import json
import threading
import time

import numpy as np

import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        in_config = pb_utils.get_input_config_by_name(model_config, "IN")

        in_shape = in_config["dims"]
        if (len(in_shape) != 1) or (in_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'IN' to be
                [1], got {}""".format(
                    args["model_name"], in_shape
                )
            )
        if in_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'IN' to be
                'TYPE_INT32', got {}""".format(
                    args["model_name"], in_config["data_type"]
                )
            )

        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        out_shape = out_config["dims"]
        if (len(out_shape) != 1) or (out_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'OUT' to be
                [1], got {}""".format(
                    args["model_name"], out_shape
                )
            )
        if out_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'OUT' to be
                'TYPE_INT32', got {}""".format(
                    args["model_name"], out_config["data_type"]
                )
            )

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        for request in requests:
            self.process_request(request)

        return None

    def process_request(self, request):
        thread = threading.Thread(
            target=self.response_thread,
            args=(
                request.get_response_sender(),
                pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
            ),
        )

        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, in_input):
        for idx in range(in_input[0]):
            out_output = pb_utils.Tensor("OUT", np.array([in_input[0]], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
            response_sender.send(response)

        response_sender.send(flags=pb_utils.TRITONSERVER_REPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        print("Finalize invoked")

        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = logging_time_sec / sleep_time_sec
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
                if cycles % cycle_to_log == 0:
                    print(
                        f"Waiting for {self.inflight_thread_count} response threads to complete..."
                    )
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1

        print("Finalize complete...")
