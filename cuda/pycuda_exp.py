import pycuda.autoinit
import pycuda.gpuarray as ga
import pycuda.cumath as gm
import numpy as np
import sys
import timeit

if len(sys.argv) == 3 and sys.argv[1] == '-l':
    length = int(sys.argv[2])
else:
    length = 10000000

np.random.seed(1)
array = np.random.randn(length).astype(np.float32)
array_gpu = ga.to_gpu(array)

def npexp():
    np.exp(array)

def gmexp():
    gm.exp(array_gpu)

if __name__ == '__main__':
    n = 1000
    t1 = timeit.timeit('npexp()', setup='from __main__ import npexp', number=n)
    print(t1)
    t2 = timeit.timeit('gmexp()', setup='from __main__ import gmexp', number=n)
    print(t2)

"""
9.962783711001975
1.2841396420008095
"""