#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

#include "samples/classification_results.h"
#include "format_reader_ptr.h"

long long get_nanoseconds() {
    // 定义一个timespec结构体，用于存储时间信息
    struct timespec ts;
    // 调用clock_gettime函数，获取当前时间，存储在ts中
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // 返回ts中的秒数乘以10的9次方，加上纳秒数
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void openvino_classifier(std::vector<std::string> image_file_names)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream ifs_labels("labels-imagenet.txt");
    while(getline(ifs_labels, line)) {
        labels.push_back(line);
    }
    ifs_labels.close();

    size_t input_batch_size = 1;
    //size_t input_batch_size = image_file_names.size();

    size_t input_channel_size = 3;
    ov::Core core;
    //auto model = core.compile_model("classifier.onnx");
    //auto model = core.compile_model("resnet50.onnx");
    auto model = core.compile_model("resnet50-dynamic.onnx");
    auto iq = model.create_infer_request();
    auto output = iq.get_output_tensor(0);
    //clock_t start1, start2, stop;
    long long ns_start1, ns_start2, ns_stop;
    //start1 = clock();
    ns_start1 = get_nanoseconds();
    //std::cout << "start1：" << start1 << std::endl;
    /*
    size_t input_height_size = 224;
    size_t input_width_size = 224;
    */
    /*
    auto image1 = cv::imread(image_file_names[0]);
    size_t input_height_size = image1.rows;
    size_t input_width_size = image1.cols;
    std::cout << "image1.rows: " << image1.rows << std::endl;
    std::cout << "image1.cols: " << image1.cols << std::endl;
    */
    /*
    input.set_shape({input_batch_size, input_channel_size, input_width_size, input_height_size});
    float* input_data_host = input.data<float>();
    */

    for (size_t i = 0; i < image_file_names.size(); ++i) {
        const auto& image_file_name = image_file_names[i];
        auto image = cv::imread(image_file_name);
        size_t input_height_size = image.rows;
        size_t input_width_size = image.cols;
        std::cout << "image1.rows: " << image.rows << std::endl;
        std::cout << "image1.cols: " << image.cols << std::endl;
        auto input = iq.get_input_tensor(0);
        input.set_shape({input_batch_size, input_channel_size, input_width_size, input_height_size});
        float* input_data_host = input.data<float>();
        float mean[] = {0.406, 0.456, 0.485};
        float std[] = {0.225, 0.224, 0.229};

        //cv::resize(image, image, cv::Size(input_width_size, input_height_size));
        int image_area = image.cols * image.rows;
        unsigned char* pimage = image.data;
        float* phost_b = input_data_host + image_area * 0;
        float* phost_g = input_data_host + image_area * 1;
        float* phost_r = input_data_host + image_area * 2;
        /*
        float* phost_b = input_data_host + image_area * 3 * i + image_area * 0;
        float* phost_g = input_data_host + image_area * 3 * i + image_area * 1;
        float* phost_r = input_data_host + image_area * 3 * i + image_area * 2;
        */
        for(int i = 0; i < image_area; ++i, pimage += 3){
            *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
            *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
            *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
        }

      //start2 = clock();
      ns_start2 = get_nanoseconds();
      iq.infer();
      ns_stop = get_nanoseconds();
      //stop = clock();
      //std::cout << "start2：" << start2 << std::endl;
      const int num_classes = 1000;
      //ClassificationResult classification_result(output, image_file_names, input_batch_size, 10, labels);
      ClassificationResult classification_result(output, {image_file_name}, input_batch_size, 10, labels);
      /*
      float* prob = output.data<float>();
      int predict_label = std::max_element(prob, prob + num_classes) - prob;

      float confidence = prob[predict_label];
      printf("confidence = %f, label_index = %d, label=%s\n", confidence, predict_label, labels[predict_label].c_str());
      */
      classification_result.show();
      //std::cout << "infer时间：" << ((double)(stop - start2)) / CLOCKS_PER_SEC << std::endl;
      //std::cout << "total时间：" << ((double)(stop - start1)) / CLOCKS_PER_SEC << std::endl;
      std::cout << "infer时间ns：" << ((double)(ns_stop - ns_start2)) / 1000000000.0 << std::endl;
      std::cout << "total时间ns：" << ((double)(ns_stop - ns_start1)) / 1000000000.0 << std::endl;

    }
}

int main(int argc, char *argv[]){
    if (argc != 2) {
        std::cout << "argc not 2: " << argc << std::endl;
        exit(-1);
    }

    std::vector<std::string> labels;
    boost::split(labels, std::string(argv[1]), boost::is_any_of(","));

    openvino_classifier(labels);
    return 0;
}

/*
classifier.onnx
./misc_classification_onnx_wthlabels car.jpeg
confidence = 0.228403, label_index = 817, label=sports car
infer时间ns：0.0497815
total时间ns：0.0505142

./misc_classification_onnx_wthlabels car.jpeg

Top 10 results:

Image car.jpeg

classid probability label
------- ----------- -----
817     0.2284026   sports car
472     0.1582138   canoe
784     0.1385197   screwdriver
693     0.0460029   paddle
590     0.0356732   hand-held computer
772     0.0245135   safety pin
867     0.0225798   trailer truck
511     0.0214936   convertible
473     0.0198144998   can opener
621     0.0198011   lawn mower

infer时间ns：0.05315462
total时间ns：0.05392876

./misc_classification_onnx_wthlabels car.jpeg,dog.jpeg,laptop.jpeg

Top 10 results:

Image car.jpeg

classid probability label
------- ----------- -----
817     0.2284025   sports car
472     0.1582143   canoe
784     0.1385197   screwdriver
693     0.0460029   paddle
590     0.0356731   hand-held computer
772     0.0245135   safety pin
867     0.0225798   trailer truck
511     0.0214935   convertible
473     0.0198998   can opener
621     0.0198011   lawn mower

Image dog.jpeg

classid probability label
------- ----------- -----
263     0.7865472   Pembroke
264     0.2117914   Cardigan
273     0.0001946   dingo
253     0.0001484   basenji
151     0.0001455   Chihuahua
231     0.0001416   collie
157     0.0001268   papillon
227     0.0000871   kelpie
232     0.0000750   Border collie
186     0.0000735   Norwich terrier

Image laptop.jpeg

classid probability label
------- ----------- -----
681     0.4685615   notebook
620     0.2356814   laptop
527     0.1043595   desktop computer
782     0.0676399   screen
810     0.0550855   space bar
590     0.0280587   hand-held computer
673     0.0074940   mouse
664     0.0066947   monitor
526     0.0042089   desk
916     0.0038441   web site

infer时间ns：0.0847067
total时间ns：0.08624184


resnet50.onnx
./misc_classification_onnx_wthlabels car.jpeg,dog.jpeg,laptop.jpeg

Top 10 resul1441ts:

Image car.jpeg

classid probability label
------- ----------- -----
817     9.1131859   sports car
472     8.7460270   canoe
784     8.6130886   screwdriver
693     7.5107803   paddle
590     7.2564735   hand-held computer
772     6.8812995   safety pin
867     6.7991319   trailer truck
511     6.7498288   convertible
473     6.6727839   can opener
621     6.6678138   lawn mower

Image dog.jpeg

classid probability label
------- ----------- -----
263     16.7579727  Pembroke
264     15.4459219  Cardigan
273     8.4533911   dingo
253     8.1822958   basenji
151     8.1629667   Chihuahua
231     8.1357012   collie
157     8.0254145   papillon
227     7.6495843   kelpie
232     7.4995999   Border collie
186     7.4799147   Norwich terrier

Image laptop.jpeg

classid probability label
------- ----------- -----
681     12.9896173  notebook
620     12.3024311  laptop
527     11.4877920  desktop computer
782     11.0541477  screen
810     10.8488359  space bar
590     10.1742477  hand-held computer
673     8.8540468   mouse
664     8.7412729   monitor
526     8.2771416   desk
916     8.1865005   web site

infer时间ns：0.06713036
total时间ns：0.06909315


resnet50-dynamic.onnx, 在3的batch_size里，把每个图片resize到第一个图片大小，预测结果都是错的
./misc_classification_onnx_wthlabels car.jpeg,dog.jpeg,laptop.jpeg

Top 10 results:

Image car.jpeg

classid probability label
------- ----------- -----
893     0.9895977   wallet
709     0.0049139   pencil box
446     0.0042856   binder
748     0.0005869   purse
593     0.0002521   harmonica
636     0.0000924   mailbag
696     0.0000720   paintbrush
592     0.0000248   hard disc
464     0.0000248   buckle
481     0.0000188   cassette

Image dog.jpeg

classid probability label
------- ----------- -----
814     0.7233258   speedboat
913     0.1111398   wreck
977     0.0739769   sandbar
144     0.0721574   pelican
978     0.0066975   seashore
472     0.0051869   canoe
693     0.0012136   paddle
975     0.0008178   lakeside
132     0.0004274   American egret
137     0.0004071   American coot

Image laptop.jpeg

classid probability label
------- ----------- -----
579     0.5943111   grand piano
881     0.4056732   upright
687     0.0000072   organ
553     0.0000017   file
526     0.0000012   desk
742     0.0000010   printer
401     0.0000010   accordion
446     0.0000008   binder
709     0.0000006   pencil box
770     0.0000004   running shoe

infer时间ns：0.05720701
total时间ns：0.05833848


resnet50-dynamic.onnx, 在1的batch_size里，不resize图片
image1.rows: 118
image1.cols: 122

Top 10 results:

Image car.jpeg

classid probability label
------- ----------- -----
893     0.9895977   wallet
709     0.0049139   pencil box
446     0.0042856   binder
748     0.0005869   purse
593     0.0002521   harmonica
636     0.0000924   mailbag
696     0.0000720   paintbrush
592     0.0000248   hard disc
464     0.0000248   buckle
481     0.0000188   cassette

infer时间ns：0.04185731
total时间ns：0.04237826
image1.rows: 118
image1.cols: 117

Top 10 results:

Image dog.jpeg

classid probability label
------- ----------- -----
169     0.7699625   borzoi
176     0.1981849   Saluki
172     0.0287884   whippet
231     0.0020972   collie
173     0.0003778   Ibizan hound
353     0.0002628   gazelle
690     0.0000573   oxcart
230     0.0000558   Shetland sheepdog
171     0.0000524   Italian greyhound
263     0.0000297   Pembroke

infer时间ns：0.01821995
total时间ns：0.06103708
image1.rows: 110
image1.cols: 120

Top 10 results:

Image laptop.jpeg

classid probability label
------- ----------- -----
791     0.5324421   shopping cart
790     0.3385024   shopping basket
746     0.0250895   puck
404     0.0180643   airliner
593     0.0174136   harmonica
628     0.0167843   liner
878     0.0155679   typewriter keyboard
754     0.0067585   radio
753     0.0051971   radiator
677     0.0041148   nail

infer时间ns：0.01679809
total时间ns：0.07846425


omz_downloader --name resnet-50-pytorch
omz_converter --name resnet-50-pytorch
#其实还是2步，先把pytorch转成onnx，onnx再转IR，导出onnx指定了input-shape

 */