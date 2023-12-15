import logging as log
import statistics
import sys
from time import perf_counter

import numpy as np
from openvino.runtime import Core, get_version
from openvino.runtime.utils.types import get_dtype


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {get_version()}")
    if len(sys.argv) != 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_model>')
        return 1
    latency = {'PERFORMANCE_HINT': 'LATENCY'}

    core = Core()
    compiled_model = core.compile_model(sys.argv[1], 'CPU', latency)
    ireq = compiled_model.create_infer_request()
    for model_input in compiled_model.inputs:
        fill_tensor_random(ireq.get_tensor(model_input))
    ireq.infer()
    seconds_to_run = 10
    niter = 10
    latencies = []
    start = perf_counter()
    time_point = start
    time_point_to_finish = start + seconds_to_run
    while time_point < time_point_to_finish or len(latencies) < niter:
        ireq.infer()
        iter_end = perf_counter()
        latencies.append((iter_end - time_point) * 1e3)
        time_point = iter_end
    end = time_point
    duration = end - start
    fps = len(latencies) / duration
    log.info(f'Count:          {len(latencies)} iterations')
    log.info(f'Duration:       {duration * 1e3:.2f} ms')
    log.info('Latency:')
    log.info(f'    Median:     {statistics.median(latencies):.2f} ms')
    log.info(f'    Average:    {sum(latencies) / len(latencies):.2f} ms')
    log.info(f'    Min:        {min(latencies):.2f} ms')
    log.info(f'    Max:        {max(latencies):.2f} ms')
    log.info(f'Throughput: {fps:.2f} FPS')


if __name__ == '__main__':
    main()

"""
python samples_benchmark/sync_benchmark.py alexnet.xml
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
[ INFO ] Count:          256 iterations
[ INFO ] Duration:       10019.47 ms
[ INFO ] Latency:
[ INFO ]     Median:     36.47 ms
[ INFO ]     Average:    39.14 ms
[ INFO ]     Min:        29.78 ms
[ INFO ]     Max:        230.23 ms
[ INFO ] Throughput: 25.55 FPS

python samples_benchmark/sync_benchmark.py resnet_fr_torch_onnx-static.xml
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
[ INFO ] Count:          105 iterations
[ INFO ] Duration:       10092.39 ms
[ INFO ] Latency:
[ INFO ]     Median:     92.22 ms
[ INFO ]     Average:    96.12 ms
[ INFO ]     Min:        85.59 ms
[ INFO ]     Max:        133.48 ms
[ INFO ] Throughput: 10.40 FPS

python samples_benchmark/sync_benchmark.py resnet_fr_torch_onnx-fp16-static.xml
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
[ INFO ] Count:          101 iterations
[ INFO ] Duration:       10099.29 ms
[ INFO ] Latency:
[ INFO ]     Median:     94.28 ms
[ INFO ]     Average:    99.99 ms
[ INFO ]     Min:        86.52 ms
[ INFO ]     Max:        210.06 ms
[ INFO ] Throughput: 10.00 FPS

"""