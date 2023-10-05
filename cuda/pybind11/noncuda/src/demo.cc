#include <pybind11/functional.h>
#include <opencv2/opencv.hpp>
#include "demo.h"

void NumpyUint83CToCvMat(py::array_t<unsigned char> &array) {
    if(array.ndim() != 3) {
        //throw std::runtime_error("3-channel image must be 3 dims");
        std::cout << "3-channel image must be 3 dims" << std::endl;
        return;
    }
    py::buffer_info buf = array.request();
    cv::Mat img_mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);
    cv::imwrite("img2.jpg", img_mat);
}

py::array_t<unsigned char> CvMatUint83CToNumpy() {
    cv::Mat img_mat = cv::imread("img2.jpg");
    py::array_t<unsigned char> array = py::array_t<unsigned char>({img_mat.rows, img_mat.cols, 3}, img_mat.data);
    return array;
}

PYBIND11_MODULE(demo, m) {
    m.doc() = "Pybind11 demo";
    m.def("NumpyUint83CToCvMat", &NumpyUint83CToCvMat);
    m.def("CvMatUint83CToNumpy", &CvMatUint83CToNumpy, py::return_value_policy::move);
}