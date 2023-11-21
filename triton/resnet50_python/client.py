import argparse
import json
import sys
import warnings

import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import *

import time

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="resnet50_python",
        help="Model name",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        required=False,
        default="http://images.cocodataset.org/test2017/000000557146.jpg",
        help="Image URL. Default is:\
                            http://images.cocodataset.org/test2017/000000557146.jpg",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        required=False,
        default="./labels.txt",
        help="Path to the file with text representation \
                        of available labels",
    )
    args = parser.parse_args()

    utils = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_convnets_processing_utils",
        skip_validation=True,
    )

    try:
        triton_client = httpclient.InferenceServerClient(args.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(args.label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    if args.verbose:
        print(json.dumps(triton_client.get_model_config(args.model_name), indent=4))

    input_name = "INPUT"
    output_name = "OUTPUT"
    batch = np.asarray(utils.prepare_input_from_uri(args.image_url))

    input = httpclient.InferInput(input_name, batch.shape, "FP32")
    output = httpclient.InferRequestedOutput(output_name)

    input.set_data_from_numpy(batch)
    a = time.time()
    results = triton_client.infer(
        model_name=args.model_name, inputs=[input], outputs=[output]
    )
    b = time.time()
    print("infer time = %f" % (b - a))

    output_data = results.as_numpy(output_name)
    max_id = np.argmax(output_data, axis=1)[0]
    print("Results is class: {}".format(labels_dict[max_id]))

    print("PASS: ResNet50 instance kind")
    sys.exit(0)


"""
python client.py

#CPU
#hub方式
infer time = 0.049290
Results is class: TABBY

#pretrain方式
infer time = 0.106938
Results is class: TABBY
infer time = 0.059960
Results is class: TABBY
infer time = 0.049828
Results is class: TABBY
infer time = 0.062158
Results is class: TABBY
infer time = 0.048690
Results is class: TABBY


#GPU
#pretrain方式
infer time = 1.352630
Results is class: TABBY
infer time = 0.027162
Results is class: TABBY
infer time = 0.010920
Results is class: TABBY
infer time = 0.023055
Results is class: TABBY
infer time = 0.045977
Results is class: TABBY

#hub方式
infer time = 1.374819
Results is class: TABBY
infer time = 0.013180
Results is class: TABBY
infer time = 0.024770
Results is class: TABBY
infer time = 0.009720
Results is class: TABBY
infer time = 0.019853
Results is class: TABBY

"""
