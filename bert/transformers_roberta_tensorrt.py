import torch
from transformers.convert_graph_to_onnx import convert
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import onnxruntime as ort
print(ort.__version__)
print(ort.get_device())

import tensorrt as trt

import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

from text_cls_common import GenDataSet, valtrans, label_dict, valtrans_onnx, softmax
from text_cls_common import save_model_path, max_length, num_classes, batch_size, dynamic_onnx_model_path
from trt_common import get_engine, allocate_buffers, do_inference_v2

test_file = 'dataset/test/real/usual_test_labeled.txt'

#python -m transformers.onnx --model=./model onnx/trans/，和用程序转一样，onnxinfer时候报错
#python -m optimum.exporters.onnx --model=./model onnx/trans/optimum.onnx
#RuntimeError: Cannot infer the task from a local directory yet, please specify the task manually.
def trans2onnx(origin_model_path, dynamic_onnx_model_file):
    convert(framework="pt",
            model=origin_model_path,
            output=Path(dynamic_onnx_model_file),
            opset=11,
            pipeline_name="sentiment-analysis")
    print("convert success")

def torch2onnx(device, save_model_path, dynamic_onnx_model_file, batch_size=-1):
    tokenizer = AutoTokenizer.from_pretrained(save_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_model_path, num_labels=num_classes)
    model.to(device)
    tokens = tokenizer(["测试输入随便写1"] * batch_size if batch_size != -1 else ["测试输入随便写1"], padding='max_length', truncation=True, max_length=max_length)
    input_id, types, masks = torch.LongTensor(np.array(tokens['input_ids'])), torch.LongTensor(np.array(tokens['token_type_ids'])), torch.LongTensor(np.array(tokens['attention_mask']))
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    #dynamic_ax = {'input_ids':[1], "input_mask":[1]}
    #dynamic_ax = {'input_ids': symbolic_names, "token_type_ids": symbolic_names, "attention_mask": symbolic_names}
    dynamic_ax = {'input_ids': symbolic_names, "token_type_ids": symbolic_names, "attention_mask": symbolic_names}
    #dynamic_ax = {'input_ids': symbolic_names, "attention_mask": symbolic_names}
    #'token_type_ids',
    with torch.no_grad():
        if batch_size == -1:
            torch.onnx.export(model,
                              (input_id.to(device),
                               types.to(device),
                               masks.to(device)),
                              dynamic_onnx_model_file,
                              verbose=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input_ids',
                                           'token_type_ids',
                                           'attention_mask'],
                              output_names=['output'],
                              dynamic_axes=dynamic_ax
                              )
        else:
            torch.onnx.export(model,
                              (input_id.to(device),
                               types.to(device),
                               masks.to(device)),
                              dynamic_onnx_model_file,
                              verbose=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input_ids',
                                           'token_type_ids',
                                           'attention_mask'],
                              output_names=['output']
                              )
    print("ONNX Model exported to {0}".format(dynamic_onnx_model_file))

"""
mkdir /data0/shouxieai/bert/trt

--shapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 \

#onnx2trt
    #trans2onnx
trtexec \
--onnx=/data0/shouxieai/bert/onnx/trans/robert-wma.onnx \
--saveEngine=/data0/shouxieai/bert/trt/bert_trans.trt \
--memPoolSize=workspace:8192 \
--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
--optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--useCudaGraph \
--device=0

[10/11/2023-09:43:56] [W] [TRT] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.

[10/11/2023-09:44:06] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[10/11/2023-09:44:07] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.

[10/12/2023-17:35:06] [W] [TRT] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.

[10/12/2023-17:35:08] [W] [TRT] Myelin graph with multiple dynamic values may have poor performance if they differ. Dynamic values are: 
[10/12/2023-17:35:08] [W] [TRT]  (# 1 (SHAPE input_ids))
[10/12/2023-17:35:08] [W] [TRT]  (# 0 (SHAPE attention_mask))
[10/12/2023-17:36:11] [I] [TRT] Detected 3 inputs and 2 output network tensors.
[10/12/2023-17:36:11] [W] [TRT] Myelin graph with multiple dynamic values may have poor performance if they differ. Dynamic values are: 
[10/12/2023-17:36:11] [W] [TRT]  (# 1 (SHAPE input_ids))
[10/12/2023-17:36:11] [W] [TRT]  (# 0 (SHAPE attention_mask))


--shapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 \

trtexec \
--onnx=/data0/shouxieai/bert/onnx/trans/robert-wma.onnx \
--saveEngine=/data0/shouxieai/bert/trt/bert_trans_fp16.trt \
--fp16 \
--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
--optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--memPoolSize=workspace:8192 \
--useCudaGraph \
--device=0

部分warning，所有在文件最后
[10/11/2023-09:45:11] [W] [TRT] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.

[10/11/2023-09:46:15] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[10/11/2023-09:46:15] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.

--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
--optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \


    #torch2onnx
trtexec \
--onnx=/data0/shouxieai/bert/onnx/torch/robert-wma.onnx \
--saveEngine=/data0/shouxieai/bert/trt/bert_torch.trt \
--memPoolSize=workspace:8192 \
--useCudaGraph \
--device=0

trtexec \
--onnx=/data0/shouxieai/bert/onnx/torch/robert-wma.onnx \
--saveEngine=/data0/shouxieai/bert/trt/bert_torch.trt \
--memPoolSize=workspace:8192 \
--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
--optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--useCudaGraph \
--device=0

trtexec \
--onnx=/data0/shouxieai/bert/onnx/torch/robert-wma.onnx \
--saveEngine=/data0/shouxieai/bert/trt/bert_torch_fp16.trt \
--fp16 \
--memPoolSize=workspace:8192 \
--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
--optShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--maxShapes=input_ids:64x256,attention_mask:64x256,token_type_ids:64x256 \
--useCudaGraph \
--device=0

"""

def torch_infer(device, model_dir, test_data):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_classes)
    model = model.to(device)
    return valtrans(model, device, test_data)

def onnx_infer(device_no, onnx_model_file, test_data):
    sess = ort.InferenceSession(onnx_model_file, providers=['CUDAExecutionProvider'])
    sess.set_providers(['CUDAExecutionProvider'], [{'device_id': device_no}])
    sess = ort.InferenceSession(onnx_model_file, providers=['TensorrtExecutionProvider'])
    a = [x.name for x in sess.get_inputs()]; print(a)
    #['input_ids', 'attention_mask', 'token_type_ids']
    return valtrans_onnx(sess, test_data)

def valtrans_trt(engine, context, data):
    acc = 0
    infertime = 0
    inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size)
    for (input_id, types, masks, y) in tqdm(data):
        input_id = np.array(input_id).astype(np.int32)
        types = np.array(types).astype(np.int32)
        masks = np.array(masks).astype(np.int32)
        context.active_optimization_profile = 0
        origin_inputshape = context.get_binding_shape(0)
        origin_inputshape[0], origin_inputshape[1] = input_id.shape
        context.set_binding_shape(0, (origin_inputshape))
        context.set_binding_shape(1, (origin_inputshape))
        context.set_binding_shape(2, (origin_inputshape))
        a = time.time()
        inputs[0].host = input_id
        inputs[1].host = types
        inputs[2].host = masks
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        trt_outputs = trt_outputs[0].reshape(origin_inputshape[0], -1)
        b = time.time()
        infertime += b - a
        #pred = np.argmax(softmax(trt_outputs), axis=-1).astype(int)
        pred = np.argmax(trt_outputs, axis=-1).astype(int)
        acc += np.sum(pred == y)
    return acc / len(data.dataset), infertime

#sudo scp dtpct:/data0/trt/samples/python/common.py trt_common.py
def trt_infer(device, trt_engine_file, test_data):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine = get_engine(trt_engine_file, TRT_LOGGER)
    context = engine.create_execution_context()
    names = [_ for _ in engine]
    input_names = list(filter(engine.binding_is_input, names))
    output_names = list(set(names) - set(input_names))
    print('input_names', input_names)
    print('output_names', output_names)
    return valtrans_trt(engine, context, test_data)

if __name__ == '__main__':
    device_no = 0
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(save_model_path)
    dataset = GenDataSet(tokenizer, {"test": test_file}, label_dict, max_length, batch_size)
    test_data_tensor = dataset.get_test_data()
    dynamic_onnx_model_file_trans = dynamic_onnx_model_path + "/trans/robert-wma.onnx"
    dynamic_onnx_model_file_trans = dynamic_onnx_model_path + "/trans/model.onnx"
    #trans2onnx(save_model_path, dynamic_onnx_model_file_trans)
    dynamic_onnx_model_file_torch = dynamic_onnx_model_path + "/torch/robert-wma.onnx"
    torch2onnx(device, save_model_path, dynamic_onnx_model_file_torch, batch_size)
    #torch2onnx(device, save_model_path, dynamic_onnx_model_file_torch)
    acc, infertime = torch_infer(device, save_model_path, test_data_tensor)
    print('%s acc:%.4f, cost: %.4fs' % ('torch_infer', acc, infertime))
    test_data_numpy = dataset.get_test_data(False)
    acc, infertime = onnx_infer(device_no, dynamic_onnx_model_file_torch, test_data_numpy)
    print('%s acc:%.4f, cost: %.4fs' % ('onnx_infer of onnx exported by torch total', acc, infertime))
    #acc, infertime = onnx_infer(device_no, dynamic_onnx_model_file_trans, test_data_numpy)
    #print('%s acc:%.4f, cost: %.4fs' % ('onnx_infer of onnx exported by trans total', acc, infertime))
    #trt_engine_file = 'trt/bert_trans.trt'
    #trt_engine_file = 'trt/bert_trans_fp16.trt'
    trt_engine_file = 'trt/bert_torch.trt'
    #trt_engine_file = 'trt/bert_torch_fp16.trt'
    #acc, infertime = trt_infer(device, trt_engine_file, test_data_numpy)
    #print('%s acc:%.4f, cost: %.4fs' % ('trt_infer of onnx exported by trans total', acc, infertime))

"""
torch_infer acc:0.7834, cost: 0.1435s

#！！！固定/动态batch，onnx infer一样慢，动态batch，trt总是cuMemHostAlloc内存申请失败，size做了abs处理以后又cuMemcpyHtoDAsync失败。
#！！！trt只有改成固定batch，并且batch和总数能够整除，不要传入小于batch行的数据trt才能成功。

onnx_infer of onnx exported by torch total acc:0.2974, cost: 10.7074s
2023-10-12 16:13:52.895829007 [W:onnxruntime:, session_state.cc:1162 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2023-10-12 16:13:52.895848188 [W:onnxruntime:, session_state.cc:1164 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
2023-10-12 16:13:53.600156940 [W:onnxruntime:, session_state.cc:1162 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2023-10-12 16:13:53.600171171 [W:onnxruntime:, session_state.cc:1164 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
2023-10-12 16:13:55.067911497 [W:onnxruntime:Default, tensorrt_execution_provider.h:77 log] [2023-10-12 08:13:55 WARNING] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.

trt_infer of onnx exported by trans total acc:0.2974, cost: 10.6828s
trt_infer of onnx exported by trans total acc:0.2974, cost: 10.6741s #把allocate放在循环外面
/data0/shouxieai/bert/transformers_roberta_tensorrt.py:201: DeprecationWarning: Use get_tensor_mode instead.
  input_names = list(filter(engine.binding_is_input, names))
  0%|          | 0/5000 [00:00<?, ?it/s]/data0/shouxieai/bert/transformers_roberta_tensorrt.py:175: DeprecationWarning: Use set_optimization_profile_async instead.
  context.active_optimization_profile = 0
/data0/shouxieai/bert/transformers_roberta_tensorrt.py:176: DeprecationWarning: Use get_tensor_shape instead.
  origin_inputshape = context.get_binding_shape(0)
/data0/shouxieai/bert/transformers_roberta_tensorrt.py:178: DeprecationWarning: Use set_input_shape instead.
  context.set_binding_shape(0, (origin_inputshape))
/data0/shouxieai/bert/transformers_roberta_tensorrt.py:179: DeprecationWarning: Use set_input_shape instead.
  context.set_binding_shape(1, (origin_inputshape))
/data0/shouxieai/bert/transformers_roberta_tensorrt.py:180: DeprecationWarning: Use set_input_shape instead.
  context.set_binding_shape(2, (origin_inputshape))
"""

"""
&&&& RUNNING TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=/data0/shouxieai/bert/onnx/trans/robert-wma.onnx --saveEngine=/data0/shouxieai/bert/trt/bert_trans_fp16.trt --fp16 --memPoolSize=workspace:8192 --shapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 --useCudaGraph --device=0
[10/11/2023-09:45:10] [I] === Model Options ===
[10/11/2023-09:45:10] [I] Format: ONNX
[10/11/2023-09:45:10] [I] Model: /data0/shouxieai/bert/onnx/trans/robert-wma.onnx
[10/11/2023-09:45:10] [I] Output:
[10/11/2023-09:45:10] [I] === Build Options ===
[10/11/2023-09:45:10] [I] Max batch: explicit batch
[10/11/2023-09:45:10] [I] Memory Pools: workspace: 8192 MiB, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
[10/11/2023-09:45:10] [I] minTiming: 1
[10/11/2023-09:45:10] [I] avgTiming: 8
[10/11/2023-09:45:10] [I] Precision: FP32+FP16
[10/11/2023-09:45:10] [I] LayerPrecisions: 
[10/11/2023-09:45:10] [I] Calibration: 
[10/11/2023-09:45:10] [I] Refit: Disabled
[10/11/2023-09:45:10] [I] Sparsity: Disabled
[10/11/2023-09:45:10] [I] Safe mode: Disabled
[10/11/2023-09:45:10] [I] DirectIO mode: Disabled
[10/11/2023-09:45:10] [I] Restricted mode: Disabled
[10/11/2023-09:45:10] [I] Build only: Disabled
[10/11/2023-09:45:10] [I] Save engine: /data0/shouxieai/bert/trt/bert_trans_fp16.trt
[10/11/2023-09:45:10] [I] Load engine: 
[10/11/2023-09:45:10] [I] Profiling verbosity: 0
[10/11/2023-09:45:10] [I] Tactic sources: Using default tactic sources
[10/11/2023-09:45:10] [I] timingCacheMode: local
[10/11/2023-09:45:10] [I] timingCacheFile: 
[10/11/2023-09:45:10] [I] Input(s)s format: fp32:CHW
[10/11/2023-09:45:10] [I] Output(s)s format: fp32:CHW
[10/11/2023-09:45:10] [I] Input build shape: input_ids=32x128+32x128+32x128
[10/11/2023-09:45:10] [I] Input build shape: attention_mask=32x128+32x128+32x128
[10/11/2023-09:45:10] [I] Input build shape: token_type_ids=32x128+32x128+32x128
[10/11/2023-09:45:10] [I] Input calibration shapes: model
[10/11/2023-09:45:10] [I] === System Options ===
[10/11/2023-09:45:10] [I] Device: 0
[10/11/2023-09:45:10] [I] DLACore: 
[10/11/2023-09:45:10] [I] Plugins:
[10/11/2023-09:45:10] [I] === Inference Options ===
[10/11/2023-09:45:10] [I] Batch: Explicit
[10/11/2023-09:45:10] [I] Input inference shape: token_type_ids=32x128
[10/11/2023-09:45:10] [I] Input inference shape: attention_mask=32x128
[10/11/2023-09:45:10] [I] Input inference shape: input_ids=32x128
[10/11/2023-09:45:10] [I] Iterations: 10
[10/11/2023-09:45:10] [I] Duration: 3s (+ 200ms warm up)
[10/11/2023-09:45:10] [I] Sleep time: 0ms
[10/11/2023-09:45:10] [I] Idle time: 0ms
[10/11/2023-09:45:10] [I] Streams: 1
[10/11/2023-09:45:10] [I] ExposeDMA: Disabled
[10/11/2023-09:45:10] [I] Data transfers: Enabled
[10/11/2023-09:45:10] [I] Spin-wait: Disabled
[10/11/2023-09:45:10] [I] Multithreading: Disabled
[10/11/2023-09:45:10] [I] CUDA Graph: Enabled
[10/11/2023-09:45:10] [I] Separate profiling: Disabled
[10/11/2023-09:45:10] [I] Time Deserialize: Disabled
[10/11/2023-09:45:10] [I] Time Refit: Disabled
[10/11/2023-09:45:10] [I] Inputs:
[10/11/2023-09:45:10] [I] === Reporting Options ===
[10/11/2023-09:45:10] [I] Verbose: Disabled
[10/11/2023-09:45:10] [I] Averages: 10 inferences
[10/11/2023-09:45:10] [I] Percentile: 99
[10/11/2023-09:45:10] [I] Dump refittable layers:Disabled
[10/11/2023-09:45:10] [I] Dump output: Disabled
[10/11/2023-09:45:10] [I] Profile: Disabled
[10/11/2023-09:45:10] [I] Export timing to JSON file: 
[10/11/2023-09:45:10] [I] Export output to JSON file: 
[10/11/2023-09:45:10] [I] Export profile to JSON file: 
[10/11/2023-09:45:10] [I] 
[10/11/2023-09:45:10] [I] === Device Information ===
[10/11/2023-09:45:10] [I] Selected Device: NVIDIA GeForce RTX 3060
[10/11/2023-09:45:10] [I] Compute Capability: 8.6
[10/11/2023-09:45:10] [I] SMs: 28
[10/11/2023-09:45:10] [I] Compute Clock Rate: 1.777 GHz
[10/11/2023-09:45:10] [I] Device Global Memory: 12044 MiB
[10/11/2023-09:45:10] [I] Shared Memory per SM: 100 KiB
[10/11/2023-09:45:10] [I] Memory Bus Width: 192 bits (ECC disabled)
[10/11/2023-09:45:10] [I] Memory Clock Rate: 7.501 GHz
[10/11/2023-09:45:10] [I] 
[10/11/2023-09:45:10] [I] TensorRT version: 8.4.1
[10/11/2023-09:45:10] [I] [TRT] [MemUsageChange] Init CUDA: CPU +333, GPU +0, now: CPU 341, GPU 248 (MiB)
[10/11/2023-09:45:11] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +328, GPU +104, now: CPU 686, GPU 352 (MiB)
[10/11/2023-09:45:11] [I] Start parsing network model
[10/11/2023-09:45:11] [I] [TRT] ----------------------------------------------------------------
[10/11/2023-09:45:11] [I] [TRT] Input filename:   /data0/shouxieai/bert/onnx/trans/robert-wma.onnx
[10/11/2023-09:45:11] [I] [TRT] ONNX IR version:  0.0.6
[10/11/2023-09:45:11] [I] [TRT] Opset version:    11
[10/11/2023-09:45:11] [I] [TRT] Producer name:    pytorch
[10/11/2023-09:45:11] [I] [TRT] Producer version: 1.12.0
[10/11/2023-09:45:11] [I] [TRT] Domain:           
[10/11/2023-09:45:11] [I] [TRT] Model version:    0
[10/11/2023-09:45:11] [I] [TRT] Doc string:       
[10/11/2023-09:45:11] [I] [TRT] ----------------------------------------------------------------
[10/11/2023-09:45:11] [W] [TRT] onnx2trt_utils.cpp:369: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[10/11/2023-09:45:12] [I] Finish parsing network model
[10/11/2023-09:45:13] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +875, GPU +378, now: CPU 1958, GPU 730 (MiB)
[10/11/2023-09:45:13] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +126, GPU +60, now: CPU 2084, GPU 790 (MiB)
[10/11/2023-09:45:13] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.embeddings.position_embeddings.weight] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.embeddings.word_embeddings.weight] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.embeddings.token_type_embeddings.weight] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.embeddings.LayerNorm.bias + (Unnamed Layer* 164) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1613 + (Unnamed Layer* 167) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1614 + (Unnamed Layer* 173) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1617 + (Unnamed Layer* 189) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.0.attention.self.query.bias + (Unnamed Layer* 170) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.0.attention.self.key.bias + (Unnamed Layer* 176) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.0.attention.self.value.bias + (Unnamed Layer* 192) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::Mul_209 + (Unnamed Layer* 118) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Finite FP32 values which would overflow in FP16 detected.  Converting to closest finite FP16 value.
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1623 + (Unnamed Layer* 250) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1624 + (Unnamed Layer* 274) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1625 + (Unnamed Layer* 291) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1626 + (Unnamed Layer* 315) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1627 + (Unnamed Layer* 321) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1630 + (Unnamed Layer* 337) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.1.attention.self.key.bias + (Unnamed Layer* 324) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1636 + (Unnamed Layer* 398) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.1.attention.output.dense.bias + (Unnamed Layer* 401) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.1.attention.output.LayerNorm.bias + (Unnamed Layer* 419) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1637 + (Unnamed Layer* 422) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.1.intermediate.dense.bias + (Unnamed Layer* 425) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1638 + (Unnamed Layer* 439) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1639 + (Unnamed Layer* 463) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1640 + (Unnamed Layer* 469) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1643 + (Unnamed Layer* 485) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.2.attention.self.query.bias + (Unnamed Layer* 466) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.2.attention.self.key.bias + (Unnamed Layer* 472) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.2.attention.self.value.bias + (Unnamed Layer* 488) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1649 + (Unnamed Layer* 546) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1650 + (Unnamed Layer* 570) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.2.intermediate.dense.bias + (Unnamed Layer* 573) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1651 + (Unnamed Layer* 587) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1652 + (Unnamed Layer* 611) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1653 + (Unnamed Layer* 617) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1656 + (Unnamed Layer* 633) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.3.attention.self.key.bias + (Unnamed Layer* 620) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.3.attention.self.value.bias + (Unnamed Layer* 636) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1662 + (Unnamed Layer* 694) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1663 + (Unnamed Layer* 718) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1664 + (Unnamed Layer* 735) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.3.output.dense.bias + (Unnamed Layer* 738) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1665 + (Unnamed Layer* 759) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1666 + (Unnamed Layer* 765) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1669 + (Unnamed Layer* 781) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.4.attention.self.key.bias + (Unnamed Layer* 768) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.4.attention.self.value.bias + (Unnamed Layer* 784) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1675 + (Unnamed Layer* 842) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1676 + (Unnamed Layer* 866) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1677 + (Unnamed Layer* 883) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.4.output.LayerNorm.bias + (Unnamed Layer* 904) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1678 + (Unnamed Layer* 907) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1679 + (Unnamed Layer* 913) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1682 + (Unnamed Layer* 929) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.5.attention.self.key.bias + (Unnamed Layer* 916) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1688 + (Unnamed Layer* 990) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.5.attention.output.dense.bias + (Unnamed Layer* 993) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1689 + (Unnamed Layer* 1014) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.5.intermediate.dense.bias + (Unnamed Layer* 1017) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1690 + (Unnamed Layer* 1031) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.5.output.dense.bias + (Unnamed Layer* 1034) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.5.output.LayerNorm.bias + (Unnamed Layer* 1052) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1691 + (Unnamed Layer* 1055) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1692 + (Unnamed Layer* 1061) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1695 + (Unnamed Layer* 1077) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.6.attention.self.key.bias + (Unnamed Layer* 1064) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.6.attention.self.value.bias + (Unnamed Layer* 1080) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1701 + (Unnamed Layer* 1138) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1702 + (Unnamed Layer* 1162) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1703 + (Unnamed Layer* 1179) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1704 + (Unnamed Layer* 1203) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1705 + (Unnamed Layer* 1209) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1708 + (Unnamed Layer* 1225) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.7.attention.self.query.bias + (Unnamed Layer* 1206) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.7.attention.self.key.bias + (Unnamed Layer* 1212) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.7.attention.self.value.bias + (Unnamed Layer* 1228) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1714 + (Unnamed Layer* 1286) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.7.attention.output.dense.bias + (Unnamed Layer* 1289) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1715 + (Unnamed Layer* 1310) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.7.intermediate.dense.bias + (Unnamed Layer* 1313) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1716 + (Unnamed Layer* 1327) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.7.output.LayerNorm.bias + (Unnamed Layer* 1348) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1717 + (Unnamed Layer* 1351) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1718 + (Unnamed Layer* 1357) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1721 + (Unnamed Layer* 1373) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.8.attention.self.key.bias + (Unnamed Layer* 1360) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.8.attention.self.value.bias + (Unnamed Layer* 1376) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1727 + (Unnamed Layer* 1434) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1728 + (Unnamed Layer* 1458) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.8.intermediate.dense.bias + (Unnamed Layer* 1461) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1729 + (Unnamed Layer* 1475) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1730 + (Unnamed Layer* 1499) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1731 + (Unnamed Layer* 1505) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1734 + (Unnamed Layer* 1521) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.9.attention.self.key.bias + (Unnamed Layer* 1508) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.9.attention.self.value.bias + (Unnamed Layer* 1524) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1740 + (Unnamed Layer* 1582) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.9.attention.output.dense.bias + (Unnamed Layer* 1585) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.9.attention.output.LayerNorm.bias + (Unnamed Layer* 1603) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1741 + (Unnamed Layer* 1606) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.9.intermediate.dense.bias + (Unnamed Layer* 1609) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1742 + (Unnamed Layer* 1623) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1743 + (Unnamed Layer* 1647) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1744 + (Unnamed Layer* 1653) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1747 + (Unnamed Layer* 1669) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.10.attention.self.key.bias + (Unnamed Layer* 1656) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.10.attention.self.value.bias + (Unnamed Layer* 1672) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1753 + (Unnamed Layer* 1730) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.10.attention.output.LayerNorm.bias + (Unnamed Layer* 1751) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1754 + (Unnamed Layer* 1754) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.10.intermediate.dense.bias + (Unnamed Layer* 1757) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1755 + (Unnamed Layer* 1771) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.10.output.dense.bias + (Unnamed Layer* 1774) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.10.output.LayerNorm.bias + (Unnamed Layer* 1792) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1756 + (Unnamed Layer* 1795) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1757 + (Unnamed Layer* 1801) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1760 + (Unnamed Layer* 1817) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.11.attention.self.key.bias + (Unnamed Layer* 1804) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1766 + (Unnamed Layer* 1878) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.encoder.layer.11.attention.output.dense.bias + (Unnamed Layer* 1881) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1767 + (Unnamed Layer* 1902) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=onnx::MatMul_1768 + (Unnamed Layer* 1919) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT]  - Values less than smallest positive FP16 Subnormal value detected. Converting to FP16 minimum subnormalized value. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.pooler.dense.weight] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=bert.pooler.dense.bias + (Unnamed Layer* 1947) [Shuffle]] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:45:22] [W] [TRT] Weights [name=classifier.weight] had the following issues when converted to FP16:
[10/11/2023-09:45:22] [W] [TRT]  - Subnormal FP16 values detected. 
[10/11/2023-09:45:22] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to reduce the magnitude of the weights.
[10/11/2023-09:46:15] [I] [TRT] Detected 3 inputs and 1 output network tensors.
[10/11/2023-09:46:15] [I] [TRT] Total Host Persistent Memory: 32
[10/11/2023-09:46:15] [I] [TRT] Total Device Persistent Memory: 0
[10/11/2023-09:46:15] [I] [TRT] Total Scratch Memory: 56631296
[10/11/2023-09:46:15] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 195 MiB, GPU 547 MiB
[10/11/2023-09:46:15] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 0.005238ms to assign 2 blocks to 2 nodes requiring 56631808 bytes.
[10/11/2023-09:46:15] [I] [TRT] Total Activation Memory: 56631808
[10/11/2023-09:46:15] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +256, now: CPU 0, GPU 256 (MiB)
[10/11/2023-09:46:15] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[10/11/2023-09:46:15] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[10/11/2023-09:46:15] [I] Engine built in 65.2045 sec.
[10/11/2023-09:46:15] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 1557, GPU 672 (MiB)
[10/11/2023-09:46:15] [I] [TRT] Loaded engine size: 196 MiB
[10/11/2023-09:46:16] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +195, now: CPU 0, GPU 195 (MiB)
[10/11/2023-09:46:16] [I] Engine deserialized in 0.0713134 sec.
[10/11/2023-09:46:16] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +54, now: CPU 0, GPU 249 (MiB)
[10/11/2023-09:46:16] [I] Using random values for input input_ids
[10/11/2023-09:46:16] [I] Created input binding for input_ids with dimensions 32x128
[10/11/2023-09:46:16] [I] Using random values for input attention_mask
[10/11/2023-09:46:16] [I] Created input binding for attention_mask with dimensions 32x128
[10/11/2023-09:46:16] [I] Using random values for input token_type_ids
[10/11/2023-09:46:16] [I] Created input binding for token_type_ids with dimensions 32x128
[10/11/2023-09:46:16] [I] Using random values for output output_0
[10/11/2023-09:46:16] [I] Created output binding for output_0 with dimensions 32x6
[10/11/2023-09:46:16] [I] Starting inference
[10/11/2023-09:46:19] [I] Warmup completed 9 queries over 200 ms
[10/11/2023-09:46:19] [I] Timing trace has 137 queries over 3.06444 s
[10/11/2023-09:46:19] [I] 
[10/11/2023-09:46:19] [I] === Trace details ===
[10/11/2023-09:46:19] [I] Trace averages of 10 runs:
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.1407 ms - Host latency: 22.197 ms (enqueue 0.0593445 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.1487 ms - Host latency: 22.1989 ms (enqueue 0.0591888 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.2424 ms - Host latency: 22.292 ms (enqueue 0.0620789 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.5651 ms - Host latency: 22.6141 ms (enqueue 0.0619751 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.2385 ms - Host latency: 22.288 ms (enqueue 0.0661377 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.2065 ms - Host latency: 22.2611 ms (enqueue 0.0619995 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.1307 ms - Host latency: 22.1888 ms (enqueue 0.0616455 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.3188 ms - Host latency: 22.3843 ms (enqueue 0.0856934 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.1994 ms - Host latency: 22.2586 ms (enqueue 0.0879028 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.0793 ms - Host latency: 22.1402 ms (enqueue 0.0851318 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.0281 ms - Host latency: 22.0874 ms (enqueue 0.081665 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.1835 ms - Host latency: 22.2446 ms (enqueue 0.0794189 ms)
[10/11/2023-09:46:19] [I] Average on 10 runs - GPU latency: 22.2541 ms - Host latency: 22.3154 ms (enqueue 0.0823242 ms)
[10/11/2023-09:46:19] [I] 
[10/11/2023-09:46:19] [I] === Performance summary ===
[10/11/2023-09:46:19] [I] Throughput: 44.7064 qps
[10/11/2023-09:46:19] [I] Latency: min = 22.0549 ms, max = 22.9698 ms, mean = 22.264 ms, median = 22.2416 ms, percentile(99%) = 22.8967 ms
[10/11/2023-09:46:19] [I] Enqueue Time: min = 0.0578003 ms, max = 0.11499 ms, mean = 0.0726013 ms, median = 0.079834 ms, percentile(99%) = 0.114868 ms
[10/11/2023-09:46:19] [I] H2D Latency: min = 0.0429688 ms, max = 0.0895996 ms, mean = 0.0534283 ms, median = 0.0537109 ms, percentile(99%) = 0.0881348 ms
[10/11/2023-09:46:19] [I] GPU Compute Time: min = 21.9946 ms, max = 22.9212 ms, mean = 22.2073 ms, median = 22.1881 ms, percentile(99%) = 22.8495 ms
[10/11/2023-09:46:19] [I] D2H Latency: min = 0.00268555 ms, max = 0.0045166 ms, mean = 0.00330659 ms, median = 0.00317383 ms, percentile(99%) = 0.00439453 ms
[10/11/2023-09:46:19] [I] Total Host Walltime: 3.06444 s
[10/11/2023-09:46:19] [I] Total GPU Compute Time: 3.0424 s
[10/11/2023-09:46:19] [I] Explanations of the performance metrics are printed in the verbose logs.
[10/11/2023-09:46:19] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8401] # trtexec --onnx=/data0/shouxieai/bert/onnx/trans/robert-wma.onnx --saveEngine=/data0/shouxieai/bert/trt/bert_trans_fp16.trt --fp16 --memPoolSize=workspace:8192 --shapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 --useCudaGraph --device=0

"""