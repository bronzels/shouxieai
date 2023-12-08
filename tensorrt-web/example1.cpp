#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

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

nvinfer1::ICudaEngine *createCudaEngine(const std::string &onnxFileName, nvinfer1::ILogger &logger) {
    using namespace std;
    using namespace nvinfer1;

    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
        builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
        nvonnxparser::createParser(*network, logger)};

    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

    // Modern version with config
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config( builder->createBuilderConfig());
    //config->setMaxWorkspaceSize(64*1024*1024)
    return builder->buildEngineWithConfig(*network, *config);
}

void launchInference(nvinfer1::IExecutionContext *context, cudaStream_t stream, std::vector<float> const &inputTensor,
                     std::vector<float> &outputTensor, void **bindings, int batchSize) {
    int inputId = 0, outputId = 1;
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
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT（almost）minimal example1 !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine("model1.onnx", logger));

    if (!engine)
        throw runtime_error("Engine creation failed !");

    cout << "=============\nbindings :\n";
    int n = engine->getNbBindings();
    for ( int i = 0; i < n; ++i) {
        Dims d = engine->getBindingDimensions(i);
        cout << i << " : " << engine->getBindingName(i) << ": dims=";
        for (int j = 0; j < d.nbDims; ++j) {
            cout << d.d[j];
            if(j < d.nbDims - 1)
                cout << "x";
        }
        cout << " , dtype=" << (int) engine->getBindingDataType(i) << " ";
        cout << (engine->bindingIsInput(i) ? "IN" : "OUT") << endl;
    }
    cout << "=============\n\n";

    logger.log(ILogger::Severity::kINFO, "Creating context ...");
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context(engine->createExecutionContext());

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    vector<float> inputTensor{0.5, -05, 1.0};
    vector<float> outputTensor(2, -4.9);
    void *bindings[2]{0};
    int batchSize = 1;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        Dims dims{engine->getBindingDimensions(i)};
        size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
        cudaMalloc(&bindings[i], size * sizeof(float));
    }

    cout << "Running the inference !" << endl;
    launchInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);
    cudaStreamSynchronize(stream);
    cout << "y = [" << outputTensor[0] << ", " << outputTensor[1] << "]" << endl;

    cudaStreamDestroy(stream);
    cudaFree(bindings[0]);
    cudaFree(bindings[1]);
    return 0;
}