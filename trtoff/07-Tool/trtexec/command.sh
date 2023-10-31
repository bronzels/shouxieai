#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.cache ./*.lock ./*.json ./*.raw ./*.log ./*.o ./*.d ./*.so

# 00-Create ONNX graphs with Onnx Graphsurgeon
python3 getOnnxModel.py

# 01-Run trtexec from ONNX file without any more option
trtexec \
  --onnx=modelA.onnx \
  > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with more options (see HelpInformation.txt to get more information)
trtexec \
  --onnx=modelA.onnx \
  --saveEngine=model-02.plan \
  --timingCacheFile=model-02.cache \
  --minShapes=tensorX:1x1x28x28 \
  --optShapes=tensorX:4x1x28x28 \
  --maxShapes=tensorX:16x1x28x28 \
  --fp16 \
  --noTF32 \
  --memPoolSize=workspace:1024MiB \
  --builderOptimizationLevel=5 \
  --maxAuxStreams=4 \
  --skipInference \
  --verbose \
  > result-02.log 2>&1

# 03-Load TensorRT engine built above and do inference
trtexec --loadEngine=model-02.plan \
  --shapes=tensorX:4x1x28x28 \
  --noDataTransfers \
  --useSpinWait \
  --useCudaGraph \
  --verbose \
  > result-03.log 2>&1

# 04-Print information of the TensorRT engine (TensorRT>=8.4）
trtexec \
  --onnx=modelA.onnx \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo="./model-04-exportLayerInfo.log" \
  > result-04.log 2>&1

# 05-Print information of profiling
trtexec \
  --loadEngine=./model-02.plan \
  --dumpProfile \
  --exportTimes="./model-02-exportTimes.json" \
  --exportProfile="./model-02-exportProfile.json" \
  > result-05.log 2>&1

# 06-Save data of input/output
trtexec \
  --loadEngine=./model-02.plan \
  --dumpOutput \
  --dumpRawBindingsToFile \
  > result-06.log 2>&1

# 07-Run TensorRT engine with loading input data
trtexec \
  --loadEngine=./model-02.plan \
  --loadInputs=tensorX:tensorX.input.1.1.28.28.Float.raw \
  --dumpOutput \
  > result-07.log 2>&1

# 08-Build and run TensorRT engine with plugins
trtexec \
  --onnx=modelB.onnx \
  --plugins=./AddScalarPlugin.so \
  > result-08.log 2>&1




