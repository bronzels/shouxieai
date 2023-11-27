"""
instance_group [
    {
        count: 1
        kind: KIND_CPU
    }
"""
import argparse
import sys
from builtins import range

import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient import utils

FLAGS = None

import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-i",
        "--protocol",
        type=str,
        required=False,
        default="HTTP",
        help="Protocol (HTTP/gRPC) used to communicate with "
        + "the inference service. Default is HTTP.",
    )

    FLAGS = parser.parse_args()

    try:
        if FLAGS.protocol.lower() == "grpc":
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
            client = grpcclient
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
            client = httpclient
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()

    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input1_data = np.ones(shape=16, dtype=np.int32)

    input_byte_size = input0_data.size * input0_data.itemsize
    output_byte_size = input_byte_size

    shm_op_handle = shm.create_shared_memory_region(
        "output_data", "/output_simple", output_byte_size * 2
    )

    triton_client.register_system_shared_memory(
        "output_data", "/output_simple", output_byte_size * 2
    )

    shm_ip_handle = shm.create_shared_memory_region(
        "input_data", "/input_simple", input_byte_size * 2
    )

    shm.set_shared_memory_region(shm_ip_handle, [input0_data])
    shm.set_shared_memory_region(shm_ip_handle, [input1_data], offset=input_byte_size)

    triton_client.register_system_shared_memory(
        "input_data", "/input_simple", input_byte_size * 2
    )

    inputs = []
    inputs.append(client.InferInput("INPUT0", [1, 16], "INT32"))
    inputs[-1].set_shared_memory("input_data", input_byte_size)

    inputs.append(client.InferInput("INPUT1", [1, 16], "INT32"))
    inputs[-1].set_shared_memory("input_data", input_byte_size, offset=input_byte_size)

    outputs = []
    if FLAGS.protocol.lower() == "grpc":
        outputs.append(client.InferRequestedOutput("OUTPUT0"))
    else:
        outputs.append(client.InferRequestedOutput("OUTPUT0", binary_data=True))
    outputs[-1].set_shared_memory("output_data", output_byte_size)

    if FLAGS.protocol.lower() == "grpc":
        outputs.append(client.InferRequestedOutput("OUTPUT1"))
    else:
        outputs.append(client.InferRequestedOutput("OUTPUT1", binary_data=True))
    outputs[-1].set_shared_memory(
        "output_data", output_byte_size, offset=output_byte_size
    )

    results = triton_client.infer(model_name="simple_python", inputs=inputs, outputs=outputs)

    output0 = results.get_output("OUTPUT0")
    if output0 is not None:
        if FLAGS.protocol.lower() == "grpc":
            output0_data = shm.get_contents_as_numpy(
                shm_op_handle, utils.triton_to_np_dtype(output0.datatype), output0.shape
            )
        else:
            output0_data = shm.get_contents_as_numpy(
                shm_op_handle,
                utils.triton_to_np_dtype(output0["datatype"]),
                output0["shape"],
            )
    else:
        print("OUTPUT0 is missing in the response.")
        sys.exit(1)

    output1 = results.get_output("OUTPUT1")
    if output1 is not None:
        if FLAGS.protocol.lower() == "grpc":
            output1_data = shm.get_contents_as_numpy(
                shm_op_handle, utils.triton_to_np_dtype(output0.datatype), output0.shape,
                offset=output_byte_size,
            )
        else:
            output1_data = shm.get_contents_as_numpy(
                shm_op_handle,
                utils.triton_to_np_dtype(output1["datatype"]),
                output1["shape"],
                offset=output_byte_size,
            )
    else:
        print("OUTPUT1 is missing in the response.")
        sys.exit(1)

    for i in range(16):
        print(
            str(input0_data[i])
            + " + "
            + str(input1_data[i])
            + " = "
            + str(output0_data[0][i])
        )
        print(
            str(input0_data[i])
            + " - "
            + str(input1_data[i])
            + " = "
            + str(output1_data[0][i])
        )
        if (input0_data[i] + input1_data[i]) != output0_data[0][i]:
            print("shm infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[i] - input1_data[i]) != output1_data[0][i]:
            print("shm infer error: incorrect difference")
            sys.exit(1)

    print(triton_client.get_system_shared_memory_status())
    triton_client.unregister_system_shared_memory()
    assert len(shm.mapped_shared_memory_regions()) == 2
    shm.destroy_shared_memory_region(shm_ip_handle)
    shm.destroy_shared_memory_region(shm_op_handle)
    assert len(shm.mapped_shared_memory_regions()) == 0

    print("PASS: system shared memory")

"""
python3 client-shm.py 
0 + 1 = 1
0 - 1 = -1
1 + 1 = 2
1 - 1 = 0
2 + 1 = 3
2 - 1 = 1
3 + 1 = 4
3 - 1 = 2
4 + 1 = 5
4 - 1 = 3
5 + 1 = 6
5 - 1 = 4
6 + 1 = 7
6 - 1 = 5
7 + 1 = 8
7 - 1 = 6
8 + 1 = 9
8 - 1 = 7
9 + 1 = 10
9 - 1 = 8
10 + 1 = 11
10 - 1 = 9
11 + 1 = 12
11 - 1 = 10
12 + 1 = 13
12 - 1 = 11
13 + 1 = 14
13 - 1 = 12
14 + 1 = 15
14 - 1 = 13
15 + 1 = 16
15 - 1 = 14
[{'name': 'input_data', 'key': '/input_simple', 'offset': 0, 'byte_size': 128}, {'name': 'output_data', 'key': '/output_simple', 'offset': 0, 'byte_size': 128}]
PASS: system shared memory

python3 client-shm.py -i gRPC -u localhost:8001
0 + 1 = 1
0 - 1 = -1
1 + 1 = 2
1 - 1 = 0
2 + 1 = 3
2 - 1 = 1
3 + 1 = 4
3 - 1 = 2
4 + 1 = 5
4 - 1 = 3
5 + 1 = 6
5 - 1 = 4
6 + 1 = 7
6 - 1 = 5
7 + 1 = 8
7 - 1 = 6
8 + 1 = 9
8 - 1 = 7
9 + 1 = 10
9 - 1 = 8
10 + 1 = 11
10 - 1 = 9
11 + 1 = 12
11 - 1 = 10
12 + 1 = 13
12 - 1 = 11
13 + 1 = 14
13 - 1 = 12
14 + 1 = 15
14 - 1 = 13
15 + 1 = 16
15 - 1 = 14
regions {
  key: "input_data"
  value {
    name: "input_data"
    key: "/input_simple"
    byte_size: 128
  }
}
regions {
  key: "output_data"
  value {
    name: "output_data"
    key: "/output_simple"
    byte_size: 128
  }
}

PASS: system shared memory



"""
