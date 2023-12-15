#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#define MAXK 1e6
//#include <format>

#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

int tmain(int argc, tchar* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;

        if (argc != 5) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name> <labels_name>" << slog::endl;
            return EXIT_FAILURE;
        }

        const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = TSTRING2STRING(argv[1]);
        const std::string image_path = TSTRING2STRING(argv[2]);
        const std::string device_name = TSTRING2STRING(argv[3]);
        const std::string labels_name = TSTRING2STRING(argv[4]);

        std::string line;
        std::ifstream ifs_labels(labels_name);
        std::vector<std::string> labels;
        while(getline(ifs_labels, line)) {
            labels.push_back(line);
        }
        ifs_labels.close();

        ov::Core core;

        slog::info << "Loading model files: " << model_path << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        printInputAndOutputsInfo(*model);

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        ov::preprocess::PrePostProcessor ppp(model);

        clock_t start = clock();
        FormatReader::ReaderPtr reader(image_path.c_str());
        if (reader.get() == nullptr) {
            std::stringstream ss;
            ss << "Image " + image_path + " cannot be read!";
            throw std::logic_error(ss.str());
        }

        ov::element::Type input_type = ov::element::u8;
        ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
        std::shared_ptr<unsigned char> input_data = reader->getData();

        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

        const ov::Layout tensor_layout{"NHWC"};

        ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
        ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);

        model = ppp.build();

        ov::CompiledModel compiled_model = core.compile_model(model, device_name);

        ov::InferRequest infer_request = compiled_model.create_infer_request();

        infer_request.set_input_tensor(input_tensor);

        clock_t start2 = clock();
        infer_request.infer();
        clock_t stop = clock();
        long duacounts = stop - start;
        long duacounts2 = stop - start2;
        double duration = ((double)duacounts)/CLOCKS_PER_SEC;
        double duration2 = ((double)duacounts2)/CLOCKS_PER_SEC;

        //std::string duaresult = std::format("duration: {.4f}", duration);
        char format_str2[16] = { 0 };
        snprintf(format_str2, sizeof(format_str2) - 1, "duration: %.8f", duration2);
        slog::info << format_str2 << slog::endl;
        char format_str[256] = { 0 };
        snprintf(format_str, sizeof(format_str) - 1, "duration preprocess(recompile every time for dynamic batch) + infer time: %.8f", duration);
        slog::info << format_str << slog::endl;

        const ov::Tensor& output_tensor = infer_request.get_output_tensor();

        ClassificationResult classification_result(output_tensor, {image_path}, 1, 10, labels);
        classification_result.show();

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
./samples_hello_classification_wthlabels alexnet.xml car.jpeg CPU labels-alexnet.txt
[ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
[ INFO ]
[ INFO ] Loading model files: alexnet.xml
[ INFO ] model name: AlexNet
[ INFO ]     inputs
[ INFO ]         input name: data
[ INFO ]         input type: f32
[ INFO ]         input shape: [1,3,227,227]
[ INFO ]     outputs
[ INFO ]         output name: prob
[ INFO ]         output type: f32
[ INFO ]         output shape: [1,1000]
[ INFO ] duration: 0.07
[ INFO ] duration preprocess(recompile every time for dynamic batch) + infer time: 0.30000000

Top 10 results:

Image car.jpeg

classid probability label
------- ----------- -----
450     0.2675963   bobsled
817     0.1104873   sports car
405     0.0519568   airship
684     0.0464151   ocarina
744     0.0321938   projectile
891     0.0316675   waffle iron
859     0.0302557   toaster
902     0.0292995   whistle
593     0.0292969   harmonica
511     0.0278494   convertible

 */