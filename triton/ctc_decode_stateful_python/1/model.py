import numpy as np
import json

import triton_python_backend_utils as pb_utils
from multiprocessing.pool import ThreadPool

class Decoder(object):
    def __init__(self, blank):
        self.prev = ''
        self.result = ''
        self.blank_symbol = blank

    def decode(self, input, start, ready):
        if start:
            self.prev = ''
            self.result = ''

        if ready:
            for li in input.decode("utf-8"):
                if li != self.prev:
                    if li != self.blank_symbol:
                        self.result += li
                    self.prev = li
        r = np.array([[self.result]])
        return r

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        max_batch_size = max(model_config["max_batch_size"], 1)
        blank = model_config.get("blank_id", "-")
        self.decoders = [Decoder(blank) for _ in range(max_batch_size)]
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def batch_decode(self, batch_input, batch_start, batch_ready):
        args = []
        idx = 0
        for i,r,s in zip(batch_input, batch_ready, batch_start):
            args.append([idx, i, r, s])
            idx += 1

        with ThreadPool() as p:
            responses = p.map(self.process_single_request, args)

        return responses

    def process_single_request(self, inp):
        decoder_idx, input, ready, start = inp
        response = self.decoders[decoder_idx].decode(input[0], start[0], ready[0])
        out_tensor_0 = pb_utils.Tensor("OUTPUT0", response.astype(self.output0_dtype))
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_0])
        return inference_response

    def execute(self, requests):
        batch_input = []
        batch_ready = []
        batch_start = []
        batch_corrid = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            batch_input += in_0.as_numpy().tolist()
            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            batch_start += in_start.as_numpy().tolist()
            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            batch_ready += in_ready.as_numpy().tolist()
            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            batch_corrid += in_corrid.as_numpy().tolist()
        responses = self.batch_decode(batch_input, batch_start, batch_ready)
        assert len(requests) == len(responses)
        return responses

    def finalize(self):
        print("Cleaning up...")

"""
Get response from 0th chunk: [[b'']]
Get response from 1th chunk: [[b'c']]
Get response from 2th chunk: [[b'c']]
Get response from 3th chunk: [[b'c']]
Get response from 4th chunk: [[b'c']]
Get response from 5th chunk: [[b'ca']]
Get response from 6th chunk: [[b'ca']]
Get response from 7th chunk: [[b'cat']]
Get response from 8th chunk: [[b'cat']]

"""




