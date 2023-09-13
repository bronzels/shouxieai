import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


# 定义矩阵乘法的GPU内核
TPB = 32
TPB_plus1 = 33
mod = SourceModule("""__global__ void matmul_smem_pad(float * mat_a, float * mat_b, float * mat_out, int m, int n, int k)
{
    __shared__ float sA[BDIM][BDIM_PLUS1];
    __shared__ float sB[BDIM][BDIM_PLUS1];

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < m && y < n)
    {
        float tmp = 0.;
        for (int i = 0; i < k/BDIM; i ++)
        {
            sA[tx][ty] = *(mat_a + k * x + ty + i * BDIM);
            sB[tx][ty] = *(mat_b + n * (tx + i * BDIM) + y);
            __syncthreads();
            for (int j = 0; j < BDIM; j ++)
                tmp += sA[tx][j] * sB[j][ty];
            __syncthreads();
        }
        *(mat_out + n * x + y) = tmp;
    }
}
""".replace('BDIM_PLUS1',str(TPB+1)).replace('BDIM',str(TPB)))

# 定义两个输入矩阵和输出矩阵
m, n, p = 3072*8, 1024*8, 2048*8
a = np.random.randn(m, n).astype(np.float32)
b = np.random.randn(n, p).astype(np.float32)
c = np.zeros((m, p), dtype=np.float32)

# 将矩阵传输到GPU上
a_gpu = drv.mem_alloc(a.nbytes)
b_gpu = drv.mem_alloc(b.nbytes)
c_gpu = drv.mem_alloc(c.nbytes)

drv.memcpy_htod(a_gpu, a)
drv.memcpy_htod(b_gpu, b)

# 执行GPU内核并将结果传回CPU
block_size = (32, 32, 1)
grid_size = ((m + block_size[0] - 1) // block_size[0],
             (p + block_size[1] - 1) // block_size[1],
             1)
matrix_multiply = mod.get_function("matmul_smem_pad")
#start_gpu = time.time()
#0.00034880638122558594明显不对
start = drv.Event()
end=drv.Event()
start.record()
matrix_multiply(a_gpu, b_gpu, c_gpu, np.int32(m), np.int32(p), np.int32(n),
                block=block_size, grid=grid_size)
end.record()
end.synchronize()
time_gpu = start.time_till(end)*1e-3
#end_gpu = time.time()
#time_gpu = end_gpu - start_gpu
method_name = "GPU(shared memory padded by c func wrapped)"
print(method_name + " time: " + str(time_gpu))

printa=True

a_cpu = a.dot(b)
if printa:
    print("a_cpu[:2,:10]:\n", a_cpu[:2, :10])
    print("a_cpu[-2:,-10:]:\n", a_cpu[-2:, -10:])

drv.memcpy_dtoh(c, c_gpu)
if printa:
    print("c[:2,:10]:\n", c[:2, :10])
    print("c[-2:,-10:]:\n", c[-2:, -10:])

# 检查结果是否正确
assert np.allclose(c, a_cpu, atol=1e-3, rtol=1e-3)

"""
GPU(shared memory padded by c func wrapped) time: 13.312400390625001
a_cpu[:2,:10]:
 [[ 120.30779   -39.71905  -176.4715   -183.55315   -22.251783   22.74745
    16.433672 -213.28099    61.74389   154.98674 ]
 [ -72.54647    89.8768     -9.922535    9.130657   31.756996  -12.121211
    83.88107    25.910006  -25.337194  -61.936424]]
a_cpu[-2:,-10:]:
 [[  11.963551   -35.26315   -213.7908      51.71185     65.769485
  -175.84863     -1.2551985   25.339666   -16.503616    92.22289  ]
 [  24.758532    44.571995  -211.3156     -38.7902     118.83846
  -120.27321    -55.2911     -47.390274    94.52349     45.375504 ]]
c[:2,:10]:
 [[ 120.30783   -39.71905  -176.47156  -183.55328   -22.25179    22.747438
    16.433565 -213.28098    61.743847  154.98663 ]
 [ -72.54645    89.87686    -9.922443    9.130603   31.756834  -12.121168
    83.88103    25.909813  -25.33713   -61.936333]]
c[-2:,-10:]:
 [[  11.963518   -35.263107  -213.79047     51.71164     65.76975
  -175.84822     -1.2552042   25.339626   -16.503641    92.22292  ]
 [  24.758692    44.571903  -211.3153     -38.790314   118.83828
  -120.273254   -55.291016   -47.390503    94.523705    45.375504 ]]
"""