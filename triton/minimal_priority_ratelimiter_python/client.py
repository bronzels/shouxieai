import sys

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue
from functools import partial

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()
user_data = UserData()

def completion_callback(user_data, result, error):
    user_data._completed_requests.put((result, error))

model_name = "minimal_priority_ratelimiter_python"
#model_name = "minimal_priority_ratelimiter_python_ensemble"
model_yat_name = "minimal_priority_ratelimiter_yat_python"
shape = [1]

with grpcclient.InferenceServerClient("localhost:8001") as client:
    input0_data = np.random.rand(*shape).astype(np.int32)
    inputs = [grpcclient.InferInput("INPUT0", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype))]
    inputs[0].set_data_from_numpy(input0_data)
    outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

    inference_count = 10
    for i in range(inference_count):
        if i % 2 == 0:
            priority = 1
        else:
            priority = 2
        client.async_infer(
                    model_name,
                    inputs,
                    partial(completion_callback, user_data),
                    request_id=model_name + "_" + str(i),
                    model_version=str(1),
                    outputs=outputs)
        client.async_infer(
                    model_yat_name,
                    inputs,
                    partial(completion_callback, user_data),
                    request_id=model_yat_name + "_" + str(i),
                    model_version=str(1),
                    outputs=outputs)
#        client.async_infer(
#                    model_name,
#                    inputs,
#                    partial(completion_callback, user_data),
#                    request_id=str(i),
#                    model_version=str(1),
#                    timeout=1000,
#                    #client_timeout=
#                    outputs=outputs,
#                    priority=priority)
    processed_count = 0
    #while processed_count < inference_count:
    while processed_count < inference_count * 2:
        (result, error) = user_data._completed_requests.get()
        processed_count += 1
        if error is not None:
            print("inference failed: " + str(error))
            continue
        this_id = result.get_response().id
        print("processed count {} request id {} has been executed".format(processed_count, this_id))
#        if int(this_id) % 2 == 0:
#            this_priority = 1
#        else:
#            this_priority = 2
#        print("Request id {} with priority {} has been executed".format(this_id, this_priority))

sys.exit(0)
