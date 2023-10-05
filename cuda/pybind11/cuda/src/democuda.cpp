#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

void run_kernel
(double *vec, double scalar, unsigned int num_elements);

void multiply_with_scalar(pybind11::array_t<double> vec, double scalar) {
    unsigned int num_elements = vec.size();
    auto ha = vec.request();
    if (ha.ndim != 1) {
        std::stringstream strstr;
        strstr << "ha.ndim != !" << std::endl;
        strstr << "ha.ndim = " << ha.ndim << std::endl;
        throw std::runtime_error(strstr.str());
    }
    double *ptr = reinterpret_cast<double*>(ha.ptr);
    run_kernel(ptr, scalar, num_elements);
}

PYBIND11_MODULE(democuda, m)
{
    m.def("multiply_with_scalar", &multiply_with_scalar, "A function which multiplies a vec with a scalar");
}