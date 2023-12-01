import queue
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

model_name = "square_int32_python"
in_values = [4, 2, 0, 1]
inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.int32))]
outputs = [grpcclient.InferRequestedOutput("OUT")]

user_data = UserData()

with grpcclient.InferenceServerClient(
    url="localhost:8001", verbose=True
) as triton_client:
    triton_client.start_stream(callback=partial(callback, user_data))

    for i in range(len(in_values)):
        in_data = np.array([in_values[i]], dtype=np.int32)
        inputs[0].set_data_from_numpy(in_data)

        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            request_id=str(i),
            outputs=outputs,
        )

    recv_count = 0
    expected_count = sum(in_values)
    result_dict = {}
    while recv_count < expected_count:
        data_item = user_data._completed_requests.get()
        if type(data_item) == InferenceServerException:
            raise data_item
        else:
            this_id = data_item.get_response().id
            if this_id not in result_dict.keys():
                result_dict[this_id] = []
            result_dict[this_id].append((recv_count, data_item))

        recv_count += 1

    for i in range(len(in_values)):
        this_id = str(i)
        if in_values[i] != 0 and this_id not in result_dict.keys():
            print("response for request id {} not received".format(this_id))
            sys.exit(1)
        elif in_values[i] == 0 and this_id in result_dict.keys():
            print("received unexpected response for request id {}".format(this_id))
            sys.exit(1)
        if in_values[i] != 0:
            if len(result_dict[this_id]) != in_values[i]:
                print(
                    "expected {} many responses for request id {}, got {}".format(
                        in_values[i], this_id, result_dict[this_id]
                    )
                )
                sys.exit(1)

        if in_values[i] != 0:
            result_list = result_dict[this_id]
            expected_data = np.array([in_values[i]], dtype=np.int32)
            for j in range(len(result_list)):
                this_data = result_list[j][1].as_numpy("OUT")
                if not np.array_equal(expected_data, this_data):
                    print(
                        "incorrect data: expected {}, got {}".format(
                            expected_data, this_data
                        )
                    )
                    sys.exit(1)

    print("PASS: square_int32")
    sys.exit(0)

"""
python client.py
stream started...
async_stream_infer
model_name: "square_int32_python"
id: "0"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\004\000\000\000"

enqueued request 0 to stream...
async_stream_infer
model_name: "square_int32_python"
id: "1"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\002\000\000\000"

enqueued request 1 to stream...
async_stream_infer
model_name: "square_int32_python"
id: "2"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\000\000\000\000"

enqueued request 2 to stream...
async_stream_infer
model_name: "square_int32_python"
id: "3"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\001\000\000\000"

enqueued request 3 to stream...
infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "1"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\002\000\000\000"
}

infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "3"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\001\000\000\000"
}

infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "1"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\002\000\000\000"
}

infer_response {
  model_name: "square_int32_python"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

"""