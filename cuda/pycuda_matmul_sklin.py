import numpy
import time
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import skcuda.linalg as sklin

TPB = 16
A = numpy.full((TPB*100, TPB*100), 3, numpy.float32)
B = numpy.full((TPB*100, TPB*100), 3, numpy.float32)

start_gpu = time.time()
a_gpu = gpuarray.to_gpu(A)
b_gpu = gpuarray.to_gpu(B)

sklin.init()
multi_gpu = sklin.dot(a_gpu, b_gpu)

end_gpu = time.time()
time_gpu = end_gpu - start_gpu
print("GPU time: "+str(time_gpu))

"""
skcuda这个库太老了，引用的numpy版本过低

/data0/envs/deepspeed/lib/python3.9/site-packages/skcuda/cublas.py:284: UserWarning: creating CUBLAS context to get version number
  warnings.warn('creating CUBLAS context to get version number')
Traceback (most recent call last):
  File "/data0/shouxieai/cuda/pycuda_matmul.py", line 5, in <module>
    import skcuda.linalg as sklin
  File "/data0/envs/deepspeed/lib/python3.9/site-packages/skcuda/linalg.py", line 25, in <module>
    from . import misc
  File "/data0/envs/deepspeed/lib/python3.9/site-packages/skcuda/misc.py", line 637, in <module>
    num_types = [np.typeDict[t] for t in \
  File "/data0/envs/deepspeed/lib/python3.9/site-packages/skcuda/misc.py", line 637, in <listcomp>
    num_types = [np.typeDict[t] for t in \
  File "/data0/envs/deepspeed/lib/python3.9/site-packages/numpy/__init__.py", line 320, in __getattr__
    raise AttributeError("module {!r} has no attribute "
AttributeError: module 'numpy' has no attribute 'typeDict'
"""