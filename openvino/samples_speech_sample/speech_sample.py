"""
mkdir wsj_dnn5b_smbr
#手工下载所有文件到wsj_dnn5b_smbr目录
https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/wsj_dnn5b_smbr/
mo --framework kaldi --input_model wsj_dnn5b_smbr/wsj_dnn5b.nnet --counts wsj_dnn5b_smbr/wsj_dnn5b.counts --remove_output_softmax --output_dir $PWD
"""
import sys
from io import BytesIO
from timeit import default_timer
from typing import Dict

import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, InferRequest, Layout, Type, set_batch

from arg_parser import parse_args
from file_options import read_utterance_file, write_utterance_file
from utils import (GNA_ATOM_FREQUENCY, GNA_CORE_FREQUENCY,
                   calculate_scale_factor, compare_with_reference,
                   get_input_layouts, get_sorted_scale_factors, log,
                   set_scale_factors)


def do_inference(data: Dict[str, np.ndarray], infer_request: InferRequest, cw_l: int = 0, cw_r: int = 0) -> np.ndarray:
    frames_to_infer = {}
    result = {}

    batch_size = infer_request.model_inputs[0].shape[0]
    num_of_frames = next(iter(data.values())).shape[0]

    for output in infer_request.model_outputs:
        result[output.any_name] = np.ndarray((num_of_frames, np.prod(tuple(output.shape)[1:])))

    for i in range(-cw_l, num_of_frames + cw_r, batch_size):
        if i < 0:
            index = 0
        elif i >= num_of_frames:
            index = num_of_frames - 1
        else:
            index = i

        for _input in infer_request.model_inputs:
            frames_to_infer[_input.any_name] = data[_input.any_name][index:index + batch_size]
            num_of_frames_to_infer = len(frames_to_infer[_input.any_name])

            frames_to_infer[_input.any_name] = np.pad(
                frames_to_infer[_input.any_name],
                [(0, batch_size - num_of_frames_to_infer), (0, 0)],
            )

            frames_to_infer[_input.any_name] = frames_to_infer[_input.any_name].reshape(_input.tensor.shape)

        frame_results = infer_request.infer(frames_to_infer)

        if i - cw_r < 0:
            continue

        for output in frame_results.keys():
            vector_result = frame_results[output].reshape((batch_size, result[output.any_name].shape[1]))
            result[output.any_name][i - cw_r:i - cw_r + batch_size] = vector_result[:num_of_frames_to_infer]

    return result


def main():
    args = parse_args()

    core = Core()

    if args.model:
        log.info(f'Reading the model: {args.model}')
        model = core.read_model(args.model)

        model.add_outputs(args.output[0] + args.reference[0])

        if args.layout:
            layouts = get_input_layouts(args.layout, model.inputs)

        ppp = PrePostProcessor(model)

        for i in range(len(model.inputs)):
            ppp.input(i).tensor().set_element_type(Type.f32)

            input_name = model.input(i).get_any_name()

            if args.layout and input_name in layouts.keys():
                ppp.input(i).tensor().set_layout(Layout(layouts[input_name]))
                ppp.input(i).model().set_layout(Layout(layouts[input_name]))

        for i in range(len(model.outputs)):
            ppp.output(i).tensor().set_element_type(Type.f32)

        model = ppp.build()

        if args.batch_size:
            batch_size = args.batch_size if args.context_window_left == args.context_window_right == 0 else 1

            if any((not _input.node.layout.empty for _input in model.inputs)):
                set_batch(model, batch_size)
            else:
                log.warning('Layout is not set for any input, so custom batch size is not set')

    devices = args.device.replace('HETERO:', '').split(',')
    plugin_config = {}

    if 'GNA' in args.device:
        gna_device_mode = devices[0] if '_' in devices[0] else 'GNA_AUTO'
        devices[0] = 'GNA'

        plugin_config['GNA_DEVICE_MODE'] = gna_device_mode
        plugin_config['GNA_PRECISION'] = f'I{args.quantization_bits}'
        plugin_config['GNA_EXEC_TARGET'] = args.exec_target
        plugin_config['GNA_PWL_MAX_ERROR_PERCENT'] = str(args.pwl_me)

        if args.import_gna_model:
            if args.scale_factor[1]:
                log.error(f'Custom scale factor can not be set for imported gna model: {args.import_gna_model}')
                return 1
            else:
                log.info(f'Using scale factor from provided imported gna model: {args.import_gna_model}')
        else:
            if args.scale_factor[1]:
                scale_factors = get_sorted_scale_factors(args.scale_factor, model.inputs)
            else:
                scale_factors = []

                for file_name in args.input[1]:
                    _, utterances = read_utterance_file(file_name)
                    scale_factor = calculate_scale_factor(utterances[0])
                    log.info('Using scale factor(s) calculated from first utterance')
                    scale_factors.append(str(scale_factor))

            set_scale_factors(plugin_config, scale_factors, model.inputs)

        if args.export_embedded_gna_model:
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE'] = args.export_embedded_gna_model
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE_GENERATION'] = args.embedded_gna_configuration

        if args.performance_counter:
            plugin_config['PERF_COUNT'] = 'YES'

    device_str = f'HETERO:{",".join(devices)}' if 'HETERO' in args.device else devices[0]

    if args.model:
        compiled_model = core.compile_model(model, device_str, plugin_config)
    else:
        with open(args.import_gna_model, 'rb') as f:
            buf = BytesIO(f.read())
            compiled_model = core.import_model(buf, device_str, plugin_config)

    if args.export_gna_model:
        log.info(f'Writing GNA Model to {args.export_gna_model}')
        user_stream = compiled_model.export_model()
        with open(args.export_gna_model, 'wb') as f:
            f.write(user_stream)
        return 0

    if args.export_embedded_gna_model:
        log.info(f'Exported GNA embedded model to file {args.export_embedded_gna_model}')
        log.info(f'GNA embedded model export done for GNA generation {args.embedded_gna_configuration}')
        return 0

    input_layer_names = args.input[0] if args.input[0] else [_input.any_name for _input in compiled_model.inputs]
    input_file_names = args.input[1]

    if len(input_layer_names) != len(input_file_names):
        log.error(f'Number of model inputs ({len(compiled_model.inputs)}) is not equal '
                  f'to number of ark files ({len(input_file_names)})')
        return 3

    input_file_data = [read_utterance_file(file_name) for file_name in input_file_names]

    infer_data = [
        {
            input_layer_names[j]: input_file_data[j].utterances[i]
            for j in range(len(input_file_data))
        }
        for i in range(len(input_file_data[0].utterances))
    ]

    output_layer_names = args.output[0] if args.output[0] else [compiled_model.outputs[0].any_name]
    output_file_names = args.output[1]

    reference_layer_names = args.reference[0] if args.reference[0] else [compiled_model.outputs[0].any_name]
    reference_file_names = args.reference[1]

    reference_file_data = [read_utterance_file(file_name) for file_name in reference_file_names]

    references = [
        {
            reference_layer_names[j]: reference_file_data[j].utterances[i]
            for j in range(len(reference_file_data))
        }
        for i in range(len(input_file_data[0].utterances))
    ]

    infer_request = compiled_model.create_infer_request()

    results = []
    total_infer_time = 0

    for i in range(len(infer_data)):
        start_infer_time = default_timer()

        for state in infer_request.query_state():
            state.reset()

        results.append(do_inference(
            infer_data[i],
            infer_request,
            args.context_window_left,
            args.context_window_right,
        ))

        infer_time = default_timer() - start_infer_time
        total_infer_time += infer_time
        num_of_frames = infer_data[i][input_layer_names[0]].shape[0]
        avg_infer_time_per_frame = infer_time / num_of_frames

        log.info('')
        log.info(f'Utterance {i}:')
        log.info(f'Total time in Infer (HW and SW): {infer_time * 1000:.2f}ms')
        log.info(f'Frames in utterance: {num_of_frames}')
        log.info(f'Average Infer time per frame: {avg_infer_time_per_frame * 1000:.2f}ms')

        for name in set(reference_layer_names + output_layer_names):
            log.info('')
            log.info(f'Output layer name: {name}')
            log.info(f'Number scores per frame: {results[i][name].shape[1]}')

            if name in references[i].keys():
                log.info('')
                compare_with_reference(results[i][name], references[i][name])

        if args.performance_counter:
            if 'GNA' in args.device:
                total_cycles = infer_request.profiling_info[0].real_time.total_seconds()
                stall_cycles = infer_request.profiling_info[1].real_time.total_seconds()
                active_cycles = total_cycles - stall_cycles
                frequency = 10**6
                if args.arch == 'CORE':
                    frequency *= GNA_CORE_FREQUENCY
                else:
                    frequency *= GNA_ATOM_FREQUENCY
                total_inference_time = total_cycles / frequency
                active_time = active_cycles / frequency
                stall_time = stall_cycles / frequency
                log.info('')
                log.info('Performance Statistics of GNA Hardware')
                log.info(f'   Total Inference Time: {(total_inference_time * 1000):.4f} ms')
                log.info(f'   Active Time: {(active_time * 1000):.4f} ms')
                log.info(f'   Stall Time:  {(stall_time * 1000):.4f} ms')

    log.info('')
    log.info(f'Total sample time: {total_infer_time * 1000:.2f}ms')

    for i in range(len(output_file_names)):
        log.info(f'Saving results from "{output_layer_names[i]}" layer to {output_file_names[i]}')
        data = [results[j][output_layer_names[i]] for j in range(len(input_file_data[0].utterances))]
        write_utterance_file(output_file_names[i], input_file_data[0].keys, data)

    log.info('This sample is an API example, '
             'for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
python speech_sample.py -d GNA_AUTO -m wsj_dnn5b.xml -i wsj_dnn5b_smbr/dev93_10.ark -r wsj_dnn5b_smbr/dev93_scores_10.ark -o result.npz
Traceback (most recent call last):
  File "/Volumes/data/workspace/shouxieai/openvino/samples_speech_sample/speech_sample.py", line 268, in <module>
    sys.exit(main())
  File "/Volumes/data/workspace/shouxieai/openvino/samples_speech_sample/speech_sample.py", line 145, in main
    compiled_model = core.compile_model(model, device_str, plugin_config)
  File "/Volumes/data/m_openvino_toolkit_macos_10_15_2023.0.0.10926.b4452d56304_x86_64/python/python3.9/openvino/runtime/ie_api.py", line 398, in compile_model
    super().compile_model(model, device_name, {} if config is None else config),
RuntimeError: Check 'false' failed at src/inference/src/core.cpp:117:
Device with "GNA" name is not registered in the OpenVINO Runtime

python speech_sample.py -d CPU -m wsj_dnn5b.xml -i wsj_dnn5b_smbr/dev93_10.ark -r wsj_dnn5b_smbr/dev93_scores_10.ark -o result.npz
[ INFO ] Reading the model: wsj_dnn5b.xml
[ INFO ] 
[ INFO ] Utterance 0:
[ INFO ] Total time in Infer (HW and SW): 14066.57ms
[ INFO ] Frames in utterance: 1294
[ INFO ] Average Infer time per frame: 10.87ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000134
[ INFO ] avg error: 0.0000013
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 1:
[ INFO ] Total time in Infer (HW and SW): 10722.79ms
[ INFO ] Frames in utterance: 1005
[ INFO ] Average Infer time per frame: 10.67ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000191
[ INFO ] avg error: 0.0000013
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 2:
[ INFO ] Total time in Infer (HW and SW): 14133.91ms
[ INFO ] Frames in utterance: 1471
[ INFO ] Average Infer time per frame: 9.61ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000172
[ INFO ] avg error: 0.0000013
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 3:
[ INFO ] Total time in Infer (HW and SW): 7895.24ms
[ INFO ] Frames in utterance: 845
[ INFO ] Average Infer time per frame: 9.34ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000191
[ INFO ] avg error: 0.0000013
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 4:
[ INFO ] Total time in Infer (HW and SW): 8264.09ms
[ INFO ] Frames in utterance: 855
[ INFO ] Average Infer time per frame: 9.67ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000134
[ INFO ] avg error: 0.0000012
[ INFO ] avg rms error: 0.0000016
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 5:
[ INFO ] Total time in Infer (HW and SW): 6560.62ms
[ INFO ] Frames in utterance: 699
[ INFO ] Average Infer time per frame: 9.39ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000134
[ INFO ] avg error: 0.0000012
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 6:
[ INFO ] Total time in Infer (HW and SW): 7368.28ms
[ INFO ] Frames in utterance: 790
[ INFO ] Average Infer time per frame: 9.33ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000153
[ INFO ] avg error: 0.0000013
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 7:
[ INFO ] Total time in Infer (HW and SW): 5821.26ms
[ INFO ] Frames in utterance: 622
[ INFO ] Average Infer time per frame: 9.36ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000134
[ INFO ] avg error: 0.0000013
[ INFO ] avg rms error: 0.0000017
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 8:
[ INFO ] Total time in Infer (HW and SW): 5134.68ms
[ INFO ] Frames in utterance: 548
[ INFO ] Average Infer time per frame: 9.37ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000134
[ INFO ] avg error: 0.0000011
[ INFO ] avg rms error: 0.0000015
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Utterance 9:
[ INFO ] Total time in Infer (HW and SW): 3556.26ms
[ INFO ] Frames in utterance: 368
[ INFO ] Average Infer time per frame: 9.66ms
[ INFO ] 
[ INFO ] Output layer name: affinetransform14:0
[ INFO ] Number scores per frame: 3425
[ INFO ] 
[ INFO ] max error: 0.0000153
[ INFO ] avg error: 0.0000011
[ INFO ] avg rms error: 0.0000016
[ INFO ] stdev error: 0.0000011
[ INFO ] 
[ INFO ] Total sample time: 83523.70ms
[ INFO ] Saving results from "affinetransform14:0" layer to result.npz
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool

"""