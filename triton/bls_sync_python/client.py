import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "bls_sync_python"
shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.random.rand(*shape).astype(np.float32)
    input1_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
        httpclient.InferInput("MODEL_NAME", [1], np_to_triton_dtype(np.object_)),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    inputs[2].set_data_from_numpy(np.array(["add_sub_python"], dtype=np.object_))

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")
    print("=========='add_sub_python' model result==========")
    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, output0_data
        )
    )
    print(
        "INPUT0 ({}) - INPUT1 ({}) = OUTPUT1 ({})".format(
            input0_data, input1_data, output1_data
        )
    )

    if not np.allclose(input0_data + input1_data, output0_data):
        print("BLS sync example error: incorrect sum")
        sys.exit(1)

    if not np.allclose(input0_data - input1_data, output1_data):
        print("BLS sync example error: incorrect difference")
        sys.exit(1)

    # Will perform the inference request on the pytorch model:
    inputs[2].set_data_from_numpy(np.array(["simple_python"], dtype=np.object_))
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")
    print("\n")
    print("=========='simple_python' model result==========")
    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, output0_data
        )
    )
    print(
        "INPUT0 ({}) - INPUT1 ({}) = OUTPUT1 ({})".format(
            input0_data, input1_data, output1_data
        )
    )
    if not np.allclose(input0_data + input1_data, output0_data):
        print("BLS sync example error: incorrect sum")
        sys.exit(1)

    if not np.allclose(input0_data - input1_data, output1_data):
        print("BLS sync example error: incorrect difference")
        sys.exit(1)

    # Will perform the same inference request on an undefined model. This leads
    # to an exception:
    print("\n")
    print("=========='undefined' model result==========")
    try:
        inputs[2].set_data_from_numpy(np.array(["undefined_model"], dtype=np.object_))
        _ = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    except InferenceServerException as e:
        print(e.message())

    print("PASS: BLS Sync")
    sys.exit(0)

"""
python client.py 
=========='add_sub_python' model result==========
INPUT0 ([0.22471994 0.28275132 0.8998241  0.7857285 ]) + INPUT1 ([0.31401908 0.21049406 0.79283434 0.04973269]) = OUTPUT0 ([0.538739   0.49324536 1.6926584  0.8354612 ])
INPUT0 ([0.22471994 0.28275132 0.8998241  0.7857285 ]) - INPUT1 ([0.31401908 0.21049406 0.79283434 0.04973269]) = OUTPUT1 ([-0.08929914  0.07225727  0.10698974  0.7359958 ])


=========='simple_python' model result==========
INPUT0 ([0.22471994 0.28275132 0.8998241  0.7857285 ]) + INPUT1 ([0.31401908 0.21049406 0.79283434 0.04973269]) = OUTPUT0 ([0.538739   0.49324536 1.6926584  0.8354612 ])
INPUT0 ([0.22471994 0.28275132 0.8998241  0.7857285 ]) - INPUT1 ([0.31401908 0.21049406 0.79283434 0.04973269]) = OUTPUT1 ([-0.08929914  0.07225727  0.10698974  0.7359958 ])


=========='undefined' model result==========
Failed to process the request(s) for model instance 'bls_sync_python_0', message: TritonModelException: Failed for execute the inference request. Model 'undefined_model' is not ready.

At:
  /models/bls_sync_python/1/model.py(26): execute

PASS: BLS Sync

"""