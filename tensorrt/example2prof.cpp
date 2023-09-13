#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

class GoblinProfiler: public nvinfer1::IProfiler {
public:
    virtual void reportLayerTime(const char *layerName, float ms) noexcept {
        using namespace std;
        cout << "PROF : " << layerName << ":" << ms << endl;
    }
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        using namespace std;
        string s;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                s = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                s = "ERROR";
                break;
            case Severity::kWARNING:
                s = "WARNING";
                break;
            case Severity::kINFO:
                s = "INFO";
                break;
            case Severity::kVERBOSE:
                s = "VERBOSE";
                break;
        }
        cerr << s << ": " << msg << endl;
    }
};

template <typename T>
struct Destroy {
    void operator()(T *t) const {
        t->destroy();
    }
};

std::string printDim(const nvinfer1::Dims & d) {
    using namespace std;
    ostringstream oss;
    for (int j = 0; j < d.nbDims; ++j) {
        oss << d.d[j];
        if(j < d.nbDims - 1)
            oss << "x";
    }
    return oss.str();
}

void printNetwork(const nvinfer1::INetworkDefinition &net) {
    using namespace std;
    using namespace nvinfer1;
    cout << "\n=============\nNetwork info :" << endl;

    cout << "\nInputs : " << endl;
    for (int i = 0; i < net.getNbInputs(); ++i) {
        ITensor * inp = net.getInput(i);
        cout << "input" << i << " , dtype=" << (int)inp->getType() << " , dims=" << printDim(inp->getDimensions()) << endl;
    }

    cout << "\nLayers : " << endl;
    cout << "Number of layers : " << net.getNbLayers() << endl;
    for (int i = 0; i < net.getNbLayers(); ++i) {
        ILayer *l = net.getLayer(i);
        cout << "layer" << i << " , name=" << l->getName() << ", type=" << (int)l->getType() << " , IN ";
        for (int j = 0; j < l->getNbInputs(); ++j)
            cout << printDim(l->getInput(j)->getDimensions()) << " ";
        cout << ", OUT ";
        for (int j = 0; j < l->getNbOutputs(); ++j)
            cout << printDim(l->getOutput(j)->getDimensions()) << " ";
        cout << endl;
    }

    cout << "\nOutputs : " << endl;
    for (int i = 0; i < net.getNbOutputs(); ++i) {
        ITensor * outp = net.getOutput(i);
        cout << "output" << i << " , dtype=" << (int)outp->getType() << " , dims=" << printDim(outp->getDimensions()) << endl;
    }

    cout << "=============\n" << endl;
}

nvinfer1::ICudaEngine *createCudaEngine(const std::string &onnxFileName, nvinfer1::ILogger &logger, int batchSize) {
    using namespace std;
    using namespace nvinfer1;

    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
        builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
        nvonnxparser::createParser(*network, logger)};

    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

    printNetwork(*network);

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims2{batchSize, 3});
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims2{batchSize, 3});
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims2{batchSize, 3});

    // Modern version with config
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config( builder->createBuilderConfig());
    //config->setMaxWorkspaceSize(64*1024*1024)
    config->addOptimizationProfile(profile);
    return builder->buildEngineWithConfig(*network, *config);
}

void launchInference(nvinfer1::IExecutionContext *context, cudaStream_t stream, std::vector<float> const &inputTensor,
                     std::vector<float> &outputTensor, void **bindings, int batchSize) {
    int inputId = 0, outputId = 1;
    using namespace std;
    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    context->enqueueV2(bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[outputId], outputTensor.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}

int main() {
    using namespace std;
    using namespace nvinfer1;

    Logger logger;
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT（almost）minimal example2 !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    int batchSize = 2;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine("model2.onnx", logger, batchSize));

    if (!engine)
        throw runtime_error("Engine creation failed !");

    cout << "=============\nbindings :\n";
    int n = engine->getNbBindings();
    for ( int i = 0; i < n; ++i) {
        Dims d = engine->getBindingDimensions(i);
        cout << i << " : " << engine->getBindingName(i) << " : dims=" << printDim(d);
        cout << " , dtype=" << (int) engine->getBindingDataType(i) << " ";
        cout << (engine->bindingIsInput(i) ? "IN" : "OUT") << endl;
    }
    cout << "=============\n\n";

    logger.log(ILogger::Severity::kINFO, "Creating context ...");
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context(engine->createExecutionContext());
    context->setBindingDimensions(0, Dims2(batchSize, 3));

    GoblinProfiler profiler;
    context->setProfiler(&profiler);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    vector<float> inputTensor{0.5, -0.5, 1.0, 0.0, 0.0, 0.0};
    vector<float> outputTensor(2 * batchSize, -4.9);
    void *bindings[2]{0};
    size_t sizes[] = {inputTensor.size(), outputTensor.size()};
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        cudaMalloc(&bindings[i], sizes[i] * sizeof(float));
    }

    cout << "Running the inference !" << endl;
    launchInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);
    cudaStreamSynchronize(stream);
    cout << "y = [";
    for (int i = 0; i < batchSize; ++i) {
        cout << " [" << outputTensor.at(2 * i) << ", " << outputTensor.at(2 * i + 1) << "]";
        if(i < batchSize - 1)
            cout << ", ";
    }
    cout << " ]" << endl;

    cudaStreamDestroy(stream);
    cudaFree(bindings[0]);
    cudaFree(bindings[1]);
    return 0;
}