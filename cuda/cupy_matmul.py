#pip install cupy-cuda116
import cupy as cp
import numpy as np
import time

m, n, p = 3072*8, 1024*8, 2048*8

a_cp = cp.random.randn(m, n).astype(cp.float32)
b_cp = cp.random.randn(n, p).astype(cp.float32)

a_np = cp.asnumpy(a_cp)
b_np = cp.asnumpy(b_cp)

c_np = a_np.dot(b_np)

#start_gpu = time.time()
start_gpu = time.perf_counter()
c_cp = a_cp.dot(b_cp)
#end_gpu = time.time()
end_gpu = time.perf_counter()
time_gpu = end_gpu - start_gpu
method_name = "GPU(shared memory padded by c func wrapped)"
print(method_name + " time: " + str(time_gpu))

printa = True

if printa:
    print("c_np[:2,:10]:\n", c_np[:2, :10])
    print("c_np[-2:,-10:]:\n", c_np[-2:, -10:])

if printa:
    print("c_cp[:2,:10]:\n", c_cp[:2, :10])
    print("c_cp[-2:,-10:]:\n", c_cp[-2:, -10:])

# 检查结果是否正确
c_np_2cp = cp.asarray(c_np)
assert cp.allclose(c_cp, c_np_2cp, atol=1e-3, rtol=1e-3)

"""
GPU(shared memory padded by c func wrapped) time: 0.4961280160023307
c_np[:2,:10]:
 [[  90.53057   -30.507149   55.858    -108.45674  -153.14172   119.95956
    73.01925   113.89256    43.8785     24.981855]
 [ 116.67839    31.564081  104.930084   19.467527  160.91046   151.58539
   -43.997765  -37.29757    88.34987  -167.44849 ]]
c_np[-2:,-10:]:
 [[ 154.72816   -59.924484   28.312683  120.08865   -38.526287  -18.111359
  -136.94104    83.12688   -26.195492   -9.385181]
 [  85.7623   -108.828575 -102.63193   219.8967     40.608376 -101.70859
   140.1974    -47.623177  186.74599    65.073456]]
c_cp[:2,:10]:
 [[  90.53055   -30.506788   55.85797  -108.45674  -153.14186   119.95943
    73.01934   113.89276    43.878445   24.981888]
 [ 116.678215   31.563978  104.93003    19.467463  160.9104    151.5853
   -43.997723  -37.29747    88.34989  -167.44844 ]]
c_cp[-2:,-10:]:
 [[ 154.72818   -59.924603   28.312784  120.08879   -38.526184  -18.111517
  -136.94086    83.12679   -26.195532   -9.385256]
 [  85.762215 -108.82859  -102.63189   219.89584    40.608387 -101.709
   140.19708   -47.62336   186.74615    65.07353 ]]
"""