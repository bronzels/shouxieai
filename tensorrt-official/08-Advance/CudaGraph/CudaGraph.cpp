#include "cookbookHelper.cuh"

using namespace nvinfer1;

const std::string trtFile {"./model.plan"};
static Logger     gLogger(ILogger::Severity::kERROR);

void run()
{
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder             *builder = createInferBuilder(gLogger);
        INetworkDefinition   *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        IBuilderConfig       *config  = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
        config->addOptimizationProfile(profile);

        IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
        network->markOutput(*identityLayer->getOutput(0));
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        if (engineString == nullptr || engineString->size() == 0)
        {
            std::cout << "Failed building serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr)
        {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        std::ofstream engineFile(trtFile, std::ios::binary);
        if (!engineFile)
        {
            std::cout << "Failed opening file to write" << std::endl;
            return;
        }
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        if (engineFile.fail())
        {
            std::cout << "Failed saving .plan file!" << std::endl;
            return;
        }
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {3, {3, 4, 5}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;
    int nBinding = engine->getNbBindings();
    int nInput   = 0;
    for (int i = 0; i < nBinding; ++i)
    {
        nInput += int(engine->bindingIsInput(i));
    }
    int nOutput = nBinding - nInput;
    for (int i = 0; i < nBinding; ++i)
    {
        std::cout << std::string("Bind[") << i << std::string(i < nInput ? "]:i[" : "]:o[") << (i < nInput ? i : i - nInput) << std::string("]->");
        std::cout << dataTypeToString(engine->getBindingDataType(i)) << std::string(" ");
        std::cout << shapeToString(context->getBindingDimensions(i)) << std::string(" ");
        std::cout << engine->getBindingName(i) << std::endl;
    }

    std::vector<int> vBindingSize(nBinding, 0);
    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    std::vector<void *> vBufferH {nBinding, nullptr};
    std::vector<void *> vBufferD {nBinding, nullptr};
    for (int i = 0; i < nBinding; ++i)
    {
        vBufferH[i] = (void *)new char[vBindingSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vBindingSize[i]));
    }

    float *pData = (float *)vBufferH[0];
    for (int i = 0; i < vBindingSize[0] / dataTypeToSize(engine->getBindingDataType(0)); ++i)
    {
        pData[i] = float(i);
    }

    int  inputSize = 3 * 4 * 5, outputSize = 1;
    Dims outputShape = context->getBindingDimensions(1);
    for (int i = 0; i < outputShape.nbDims; ++i)
    {
        outputSize *= outputShape.d[i];
    }
    std::vector<float>  inputH0(inputSize, 1.0f);
    std::vector<float>  outputH0(outputSize, 0.0f);
    std::vector<void *> binding = {nullptr, nullptr};
    CHECK(cudaMalloc(&binding[0], sizeof(float) * inputSize));
    CHECK(cudaMalloc(&binding[1], sizeof(float) * outputSize));
    for (int i = 0; i < inputSize; ++i)
    {
        inputH0[i] = (float)i;
    }

    // 运行推理和使用 CUDA Graph 要用的流
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 捕获 CUDA Graph 之前要运行一次推理，https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a2f4429652736e8ef6e19f433400108c7
    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);

    for (int i = 0; i < nBinding; ++i)
    {
        printArrayInformation((float *)vBufferH[i], context->getBindingDimensions(i), std::string(engine->getBindingName(i)), true, true);
    }

    cudaGraph_t     graph;
    cudaGraphExec_t graphExec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    //cudaStreamSynchronize(stream); // 不用在 graph 内同步
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // 输入尺寸改变后，也需要首先运行一次推理，然后重新捕获 CUDA Graph，最后再运行推理
    context->setBindingDimensions(0, Dims32 {3, {2, 3, 4}});
    std::cout << std::string("Binding all? ") << std::string(context->allInputDimensionsSpecified() ? "Yes" : "No") << std::endl;

    for (int i = 0; i < nBinding; ++i)
    {
        Dims32 dim  = context->getBindingDimensions(i);
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vBindingSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    // 这里偷懒，因为本次推理绑定的输入输出数据形状不大于上一次推理，所以这里不再重新准备所有 buffer

    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    cudaStreamSynchronize(stream);

    for (int i = 0; i < nBinding; ++i)
    {
        printArrayInformation((float *)vBufferH[i], context->getBindingDimensions(i), std::string(engine->getBindingName(i)), true, true);
    }

    // 再次捕获 CUDA Graph 并运行推理
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferD[i], vBufferH[i], vBindingSize[i], cudaMemcpyHostToDevice, stream));
    }

    context->enqueueV2(vBufferD.data(), stream, nullptr);

    for (int i = nInput; i < nBinding; ++i)
    {
        CHECK(cudaMemcpyAsync(vBufferH[i], vBufferD[i], vBindingSize[i], cudaMemcpyDeviceToHost, stream));
    }
    //cudaStreamSynchronize(stream); // 不用在 graph 内同步
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    for (int i = 0; i < nBinding; ++i)
    {
        delete[] vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    run();
    return 0;
}

"""
Succeeded building serialized engine!
Succeeded building engine!
Succeeded saving .plan file!
Binding all? Yes
Bind[0]:i[0]->FP32  (3, 4, 5) inputT0
Bind[1]:o[0]->FP32  (3, 4, 5) (Unnamed Layer* 0) [Identity]_output

inputT0: (3, 4, 5, )
absSum=1770.0000,mean=29.5000,var=299.9167,max=59.0000,min= 0.0000,diff=59.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
50.00000, 51.00000, 52.00000, 53.00000, 54.00000, 55.00000, 56.00000, 57.00000, 58.00000, 59.00000,
 0.000  1.000  2.000  3.000  4.000
 5.000  6.000  7.000  8.000  9.000
10.000 11.000 12.000 13.000 14.000
15.000 16.000 17.000 18.000 19.000

20.000 21.000 22.000 23.000 24.000
25.000 26.000 27.000 28.000 29.000
30.000 31.000 32.000 33.000 34.000
35.000 36.000 37.000 38.000 39.000

40.000 41.000 42.000 43.000 44.000
45.000 46.000 47.000 48.000 49.000
50.000 51.000 52.000 53.000 54.000
55.000 56.000 57.000 58.000 59.000



(Unnamed Layer* 0) [Identity]_output: (3, 4, 5, )
absSum=1770.0000,mean=29.5000,var=299.9167,max=59.0000,min= 0.0000,diff=59.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
50.00000, 51.00000, 52.00000, 53.00000, 54.00000, 55.00000, 56.00000, 57.00000, 58.00000, 59.00000,
 0.000  1.000  2.000  3.000  4.000
 5.000  6.000  7.000  8.000  9.000
10.000 11.000 12.000 13.000 14.000
15.000 16.000 17.000 18.000 19.000

20.000 21.000 22.000 23.000 24.000
25.000 26.000 27.000 28.000 29.000
30.000 31.000 32.000 33.000 34.000
35.000 36.000 37.000 38.000 39.000

40.000 41.000 42.000 43.000 44.000
45.000 46.000 47.000 48.000 49.000
50.000 51.000 52.000 53.000 54.000
55.000 56.000 57.000 58.000 59.000


Binding all? Yes

inputT0: (2, 3, 4, )
absSum=276.0000,mean=11.5000,var=47.9167,max=23.0000,min= 0.0000,diff=23.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
14.00000, 15.00000, 16.00000, 17.00000, 18.00000, 19.00000, 20.00000, 21.00000, 22.00000, 23.00000,
 0.000  1.000  2.000  3.000
 4.000  5.000  6.000  7.000
 8.000  9.000 10.000 11.000

12.000 13.000 14.000 15.000
16.000 17.000 18.000 19.000
20.000 21.000 22.000 23.000



(Unnamed Layer* 0) [Identity]_output: (2, 3, 4, )
absSum=276.0000,mean=11.5000,var=47.9167,max=23.0000,min= 0.0000,diff=23.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
14.00000, 15.00000, 16.00000, 17.00000, 18.00000, 19.00000, 20.00000, 21.00000, 22.00000, 23.00000,
 0.000  1.000  2.000  3.000
 4.000  5.000  6.000  7.000
 8.000  9.000 10.000 11.000

12.000 13.000 14.000 15.000
16.000 17.000 18.000 19.000
20.000 21.000 22.000 23.000


Succeeded getting serialized engine!
Succeeded loading engine!
Binding all? Yes
Bind[0]:i[0]->FP32  (3, 4, 5) inputT0
Bind[1]:o[0]->FP32  (3, 4, 5) (Unnamed Layer* 0) [Identity]_output

inputT0: (3, 4, 5, )
absSum=1770.0000,mean=29.5000,var=299.9167,max=59.0000,min= 0.0000,diff=59.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
50.00000, 51.00000, 52.00000, 53.00000, 54.00000, 55.00000, 56.00000, 57.00000, 58.00000, 59.00000,
 0.000  1.000  2.000  3.000  4.000
 5.000  6.000  7.000  8.000  9.000
10.000 11.000 12.000 13.000 14.000
15.000 16.000 17.000 18.000 19.000

20.000 21.000 22.000 23.000 24.000
25.000 26.000 27.000 28.000 29.000
30.000 31.000 32.000 33.000 34.000
35.000 36.000 37.000 38.000 39.000

40.000 41.000 42.000 43.000 44.000
45.000 46.000 47.000 48.000 49.000
50.000 51.000 52.000 53.000 54.000
55.000 56.000 57.000 58.000 59.000



(Unnamed Layer* 0) [Identity]_output: (3, 4, 5, )
absSum=1770.0000,mean=29.5000,var=299.9167,max=59.0000,min= 0.0000,diff=59.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
50.00000, 51.00000, 52.00000, 53.00000, 54.00000, 55.00000, 56.00000, 57.00000, 58.00000, 59.00000,
 0.000  1.000  2.000  3.000  4.000
 5.000  6.000  7.000  8.000  9.000
10.000 11.000 12.000 13.000 14.000
15.000 16.000 17.000 18.000 19.000

20.000 21.000 22.000 23.000 24.000
25.000 26.000 27.000 28.000 29.000
30.000 31.000 32.000 33.000 34.000
35.000 36.000 37.000 38.000 39.000

40.000 41.000 42.000 43.000 44.000
45.000 46.000 47.000 48.000 49.000
50.000 51.000 52.000 53.000 54.000
55.000 56.000 57.000 58.000 59.000


Binding all? Yes

inputT0: (2, 3, 4, )
absSum=276.0000,mean=11.5000,var=47.9167,max=23.0000,min= 0.0000,diff=23.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
14.00000, 15.00000, 16.00000, 17.00000, 18.00000, 19.00000, 20.00000, 21.00000, 22.00000, 23.00000,
 0.000  1.000  2.000  3.000
 4.000  5.000  6.000  7.000
 8.000  9.000 10.000 11.000

12.000 13.000 14.000 15.000
16.000 17.000 18.000 19.000
20.000 21.000 22.000 23.000



(Unnamed Layer* 0) [Identity]_output: (2, 3, 4, )
absSum=276.0000,mean=11.5000,var=47.9167,max=23.0000,min= 0.0000,diff=23.0000,
 0.00000,  1.00000,  2.00000,  3.00000,  4.00000,  5.00000,  6.00000,  7.00000,  8.00000,  9.00000,
14.00000, 15.00000, 16.00000, 17.00000, 18.00000, 19.00000, 20.00000, 21.00000, 22.00000, 23.00000,
 0.000  1.000  2.000  3.000
 4.000  5.000  6.000  7.000
 8.000  9.000 10.000 11.000

12.000 13.000 14.000 15.000
16.000 17.000 18.000 19.000
20.000 21.000 22.000 23.000



"""