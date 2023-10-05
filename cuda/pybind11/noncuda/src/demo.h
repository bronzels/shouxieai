#ifndef DEMO_PYBIND11_SRC_DEMO_H_
#define DEMO_PYBIND11_SRC_DEMO_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void NumpyUint83CToCvMat(py::array_t<unsigned char> &array);

py::array_t<unsigned char> CvMatUint83CToNumpy();

#endif //DEMO_PYBIND11_SRC_DEMO_H_