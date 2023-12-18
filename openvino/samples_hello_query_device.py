import logging as log
import sys

from openvino.runtime import Core

def param_to_string(parameters) -> str:
    if isinstance(parameters, (list, tuple)):
        return ', '.join([str(x) for x in parameters])
    else:
        return str(parameters)

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    core = Core()
    log.info('Available devices:')
    for device in core.available_devices:
        log.info(f'{device} :')
        log.info('\tSUPPORTED_PROPERTIES:')
        for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
            if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                try:
                    property_val = core.get_property(device, property_key)
                except TypeError:
                    property_val = 'UNSUPPORTED TYPE'
                log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
        log.info('')

    return 0

if __name__ == '__main__':
    sys.exit(main())

"""
/Volumes/data/envs/openvino/bin/python /Volumes/data/workspace/shouxieai/openvino/hello_query_device.py 
[ INFO ] Available devices:
[ INFO ] CPU :
[ INFO ] 	SUPPORTED_PROPERTIES:
[ INFO ] 		AVAILABLE_DEVICES: 
[ INFO ] 		RANGE_FOR_ASYNC_INFER_REQUESTS: 1, 1, 1
[ INFO ] 		RANGE_FOR_STREAMS: 1, 4
[ INFO ] 		FULL_DEVICE_NAME: Intel(R) Core(TM) i5-4308U CPU @ 2.80GHz
[ INFO ] 		OPTIMIZATION_CAPABILITIES: FP32, FP16, INT8, BIN, EXPORT_IMPORT
[ INFO ] 		CACHING_PROPERTIES: {'FULL_DEVICE_NAME': 'RO'}
[ INFO ] 		NUM_STREAMS: 1
[ INFO ] 		AFFINITY: Affinity.NONE
[ INFO ] 		INFERENCE_NUM_THREADS: 0
[ INFO ] 		PERF_COUNT: False
[ INFO ] 		INFERENCE_PRECISION_HINT: <Type: 'float32'>
[ INFO ] 		PERFORMANCE_HINT: PerformanceMode.LATENCY
[ INFO ] 		EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
[ INFO ] 		PERFORMANCE_HINT_NUM_REQUESTS: 0
[ INFO ] 		ENABLE_CPU_PINNING: True
[ INFO ] 		SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
[ INFO ] 		ENABLE_HYPER_THREADING: True
[ INFO ] 		DEVICE_ID: 
[ INFO ] 

能读出NVIDIA GPU信息，但是执行分类会出错
python samples_hello_classification_wthlabels.py alexnet.xml car.jpeg GPU labels-imagenet.txt
[ INFO ] Creating OpenVINO Runtime Core
[ INFO ] Reading the model:alexnet.xml
61 warnings generated.
70 warnings generated.
Traceback (most recent call last):
  File "/data0/shouxieai/openvino/samples_hello_classification_wthlabels.py", line 120, in <module>
    sys.exit(main())
  File "/data0/shouxieai/openvino/samples_hello_classification_wthlabels.py", line 85, in main
    compiled_model = core.compile_model(model, device_name)
  File "/data0/envs/deepspeed/lib/python3.9/site-packages/openvino/runtime/ie_api.py", line 398, in compile_model
    super().compile_model(model, device_name, {} if config is None else config),
RuntimeError: Check 'false' failed at src/inference/src/core.cpp:114:
Check 'false' failed at src/plugins/intel_gpu/src/plugin/program.cpp:384:
GPU program build failed!
Program build failed(0_part_0): You may enable OCL source dump to see the error log.


"""