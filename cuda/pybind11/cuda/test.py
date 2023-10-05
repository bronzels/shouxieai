import sys

import democuda
import numpy

vec = numpy.linspace(0, 100, 1 << 10)

print("before vec[:10]: ", vec[:10])
print("before vec[-10:]: ", vec[-10:])
democuda.multiply_with_scalar(vec, 10.)
print("after vec[:10]: ", vec[:10])
print("after vec[-10:]: ", vec[-10:])

