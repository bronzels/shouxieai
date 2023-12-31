#!/usr/bin/env python
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys

import numpy as np
import tritonclient.http as httpclient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    FLAGS = parser.parse_args()

    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # First send a single request to the nonbatching model.
    print("=========")
    input0_data = np.array([1, 2, 3, 4], dtype=np.int32)
    print("Sending request to nonbatching model: IN0 = {}".format(input0_data))

    inputs = [httpclient.InferInput("IN0", [4], "INT32")]
    inputs[0].set_data_from_numpy(input0_data)
    result = triton_client.infer("nonbatching_minimal", inputs)

    print("Response: {}".format(result.get_response()))
    print("OUT0 = {}".format(result.as_numpy("OUT0")))

    # Send 2 requests to the batching model. Because these are sent
    # asynchronously and Triton's dynamic batcher is configured to
    # delay up to 5 seconds when forming a batch for this model, we
    # expect these 2 requests to be batched within Triton and sent to
    # the minimal backend as a single batch.
    print("\n=========")
    async_requests = []

    input0_data = np.array([[10, 11, 12, 13]], dtype=np.int32)
    print("Sending request to batching model: IN0 = {}".format(input0_data))
    inputs = [httpclient.InferInput("IN0", [1, 4], "INT32")]
    inputs[0].set_data_from_numpy(input0_data)
    async_requests.append(triton_client.async_infer("batching_minimal", inputs))

    input0_data = np.array([[20, 21, 22, 23],[30, 31, 32, 33]], dtype=np.int32)
    print("Sending request to batching model: IN0 = {}".format(input0_data))
    inputs = [httpclient.InferInput("IN0", [2, 4], "INT32")]
    inputs[0].set_data_from_numpy(input0_data)
    async_requests.append(triton_client.async_infer("batching_minimal", inputs))

    for async_request in async_requests:
        # Get the result from the initiated asynchronous inference
        # request. This call will block till the server responds.
        result = async_request.get_result()
        print("Response: {}".format(result.get_response()))
        print("OUT0 = {}".format(result.as_numpy("OUT0")))
