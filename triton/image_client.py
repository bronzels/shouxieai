import argparse
import os
import sys
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
from torchvision import transforms

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

import time

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def completion_callback(user_data, result, error):
    user_data._completed_requests.put((result, error))

FLAGS = None

def parse_model(model_metadata, model_config):
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception(
            "expecting 1 output, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output_metadata.datatype
        )

    output_batch_dim = model_config.max_batch_size > 0
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    input_batch_dim = model_config.max_batch_size > 0
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(input_metadata.shape)
            )
        )

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if (input_config.format != mc.ModelInput.FORMAT_NCHW) and (
        input_config.format != mc.ModelInput.FORMAT_NHWC
    ):
        raise Exception(
            "unexpected input format "
            + mc.ModelInput.Format.Name(input_config.format)
            + ", expecting "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
            + " or "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
        )

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size,
        input_metadata.name,
        output_metadata.name,
        c,
        h,
        w,
        input_config.format,
        input_metadata.datatype
    )

def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    if c == 1:
        sample_img = img.convert("L")
    else:
        sample_img = img.convert("RGB")

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == "INCEPTION":
        scaled = (typed / 127.5) - 1
    elif scaling == "VGG":
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    return ordered

def preprocess_tch(img):
    _preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return _preprocess(img).numpy()


def postprocess(results, output_name, batch_size, supports_batching):
    output_array = results.as_numpy(output_name)
    if supports_batching and len(output_array) != batch_size:
        raise Exception(
            "expected {} results, got {}".format(batch_size, len(output_array))
        )

    for results in output_array:
        if not supports_batching:
            results = [results]
        #print("results:", results[:5])
        for result in results[:1]:
            if output_array.dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(":")
            else:
                cls = result.split(":")
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))

def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    #outputs = [client.InferRequestedOutput(output_name, binary_data=True, class_count=FLAGS.classes)]
    outputs = [client.InferRequestedOutput(output_name, class_count=FLAGS.classes)]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version

def convert_http_metadata_config(_metadata, _config):
    try:
        from attrdict import AttrDict
    except ImportError:
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)

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
        "-a",
        "--async",
        dest="async_set",
        action="store_true",
        required=False,
        default=False,
        help="Use asynchronous inference API",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Use streaming inference API. "
        + "The flag is only available with gRPC protocol.",
    )
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="Name of model"
    )
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="Batch size. Default is 1.",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=False,
        default=1,
        help="Number of class results to report. Default is 1.",
    )
    parser.add_argument(
        "-s",
        "--scaling",
        type=str,
        choices=["NONE", "INCEPTION", "VGG"],
        required=False,
        default="NONE",
        help="Type of scaling to apply to image pixels. Default is NONE.",
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
    parser.add_argument(
        "image_filename",
        type=str,
        nargs="?",
        default=None,
        help="Input image / Input folder.",
    )
    FLAGS = parser.parse_args()

    if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
        raise Exception("Streaming is only allowed with gRPC protocol")

    try:
        if FLAGS.protocol.lower() == "grpc":
            triton_client = grpcclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose
            )
        else:
            concurrency = 20 if FLAGS.async_set else 1
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency
            )
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    try:
        model_metadata = triton_client.get_model_metadata(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version
        )
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)

    try:
        model_config = triton_client.get_model_config(
            model_name=FLAGS.model_name, model_version=FLAGS.model_version
        )
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)

    if FLAGS.protocol.lower() == "grpc":
        model_config = model_config.config
    else:
        model_metadata, model_config = convert_http_metadata_config(
            model_metadata, model_config
        )

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config
    )

    supports_batching = max_batch_size > 0
    if not supports_batching and FLAGS.batch_size != 1:
        print("ERROR: This model doesn't support batching.")
        sys.exit(1)

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    image_data = []
    for filename in filenames:
        img = Image.open(filename)
        image_data.append(
            preprocess_tch(
                img
            )
        )
        #image_data.append(
        #    preprocess(
        #        img, format, dtype, c, h, w, FLAGS.scaling, FLAGS.protocol.lower()
        #    )
        #)

    requests = []
    responses = []
    result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    user_data = UserData()

    async_requests = []

    sent_count = 0

    if FLAGS.streaming:
        triton_client.start_stream(partial(completion_callback, user_data))

    a = time.time()
    while not last_request:
        input_filenames = []
        repeated_image_data = []

        for idx in range(FLAGS.batch_size):
            input_filenames.append(filenames[image_idx])
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        if supports_batching:
            batched_image_data = np.stack(repeated_image_data, axis=0)
        else:
            batched_image_data = repeated_image_data[0]

        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                batched_image_data, input_name, output_name, dtype, FLAGS
            ):
                sent_count += 1
                if FLAGS.streaming:
                    triton_client.async_stream_infer(
                        FLAGS.model_name,
                        inputs,
                        request_id=str(sent_count),
                        model_version=FLAGS.model_version,
                        outputs=outputs,
                    )
                elif FLAGS.async_set:
                    if FLAGS.protocol.lower() == "grpc":
                        triton_client.async_infer(
                            FLAGS.model_name,
                            inputs,
                            partial(completion_callback, user_data),
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs,
                        )
                    else:
                        async_requests.append(
                            triton_client.async_infer(
                                FLAGS.model_name,
                                inputs,
                                request_id=str(sent_count),
                                model_version=FLAGS.model_version,
                                outputs=outputs,
                            )
                        )
                else:
                    responses.append(
                        triton_client.infer(
                            FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs,
                        )
                    )
        except InferenceServerException as e:
            print("inference failed: " + str(e))
            if FLAGS.streaming:
                triton_client.stop_stream()
            sys.exit(1)

    if FLAGS.streaming:
        triton_client.stop_stream()

    if FLAGS.protocol.lower() == "grpc":
        if FLAGS.streaming or FLAGS.async_set:
            processed_count = 0
            while processed_count < sent_count:
                (results, error) = user_data._completed_requests.get()
                processed_count += 1
                if error is not None:
                    print("inference failed: " + str(error))
                    sys.exit(1)
                responses.append(results)
    else:
        if FLAGS.async_set:
            for async_request in async_requests:
                responses.append(async_request.get_result())

    for response in responses:
        if FLAGS.protocol.lower() == "grpc":
            this_id = response.get_response().id
        else:
            this_id = response.get_response()["id"]
        print("Request {}, batch size {}".format(this_id, FLAGS.batch_size))
        postprocess(response, output_name, FLAGS.batch_size, supports_batching)
    b = time.time()
    print("infer time = %f" % (b - a))

    print("PASS")


"""
python image_client.py -m resnet50_pythononnx -c 1000 -b 1 /data/pics
python image_client.py -m resnet50_tensorrt -c 1000 -b 1 /data/pics
Request 1, batch size 1
    9.538013 (817) = SPORTS CAR
Request 2, batch size 1
    11.358579 (283) = PERSIAN CAT
Request 3, batch size 1
    15.628892 (263) = PEMBROKE
Request 4, batch size 1
    14.287535 (268) = MEXICAN HAIRLESS
Request 5, batch size 1
    12.517859 (782) = SCREEN
Request 6, batch size 1
    11.378448 (665) = MOPED
infer time = 0.068466
PASS

python image_client.py -m resnet50_tensorrt -c 1000 -b 1 -a /data/pics
Request 1, batch size 1
    9.538013 (817) = SPORTS CAR
Request 2, batch size 1
    11.358579 (283) = PERSIAN CAT
Request 3, batch size 1
    15.628892 (263) = PEMBROKE
Request 4, batch size 1
    14.287535 (268) = MEXICAN HAIRLESS
Request 5, batch size 1
    12.517859 (782) = SCREEN
Request 6, batch size 1
    11.378448 (665) = MOPED
infer time = 0.075686
PASS

python image_client.py -m resnet50_tensorrt -c 1000 -b 1 -u localhost:8001 -i gRPC /data/pics
Request 1, batch size 1
    9.538013 (817) = SPORTS CAR
Request 2, batch size 1
    11.358579 (283) = PERSIAN CAT
Request 3, batch size 1
    15.628892 (263) = PEMBROKE
Request 4, batch size 1
    14.287535 (268) = MEXICAN HAIRLESS
Request 5, batch size 1
    12.517859 (782) = SCREEN
Request 6, batch size 1
    11.378448 (665) = MOPED
infer time = 0.042891
PASS

python image_client.py -m resnet50_tensorrt -c 1000 -b 1 -a -u localhost:8001 -i gRPC /data/pics
Request 1, batch size 1
    9.538013 (817) = SPORTS CAR
Request 2, batch size 1
    11.358579 (283) = PERSIAN CAT
Request 3, batch size 1
    15.628892 (263) = PEMBROKE
Request 4, batch size 1
    14.287535 (268) = MEXICAN HAIRLESS
Request 5, batch size 1
    12.517859 (782) = SCREEN
Request 6, batch size 1
    11.378448 (665) = MOPED
infer time = 0.021659
PASS

python image_client.py -m resnet50_tensorrt -c 1000 -b 1 --streaming -u localhost:8001 -i gRPC /data/pics
Request 1, batch size 1
    9.538013 (817) = SPORTS CAR
Request 2, batch size 1
    11.358579 (283) = PERSIAN CAT
Request 3, batch size 1
    15.628892 (263) = PEMBROKE
Request 4, batch size 1
    14.287535 (268) = MEXICAN HAIRLESS
Request 5, batch size 1
    12.517859 (782) = SCREEN
Request 6, batch size 1
    11.378448 (665) = MOPED
infer time = 0.035652
PASS

"""