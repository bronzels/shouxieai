from cuda import cuda, nvrtc
import numpy as np
import time

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

# 定义矩阵乘法的GPU内核
TPB = 32
TPB_plus1 = 33
matmul_smem_pad = """\
extern "C" __global__ 
void matmul_smem_pad(float * mat_a, float * mat_b, float * mat_out, size_t m, size_t n, size_t k)
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
""".replace('BDIM_PLUS1',str(TPB+1)).replace('BDIM',str(TPB))

# Create program
err, prog = nvrtc.nvrtcCreateProgram(str.encode(matmul_smem_pad), b"matmul_smem_pad.cu", 0, [], [])

# Compile program
opts = [b"--fmad=false", b"--gpu-architecture=compute_86"]
err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)

# Get PTX from compilation
err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
ptx = b" " * ptxSize
err, = nvrtc.nvrtcGetPrtx(prog, ptx)

# Initialize CUDA Driver API
err, = cuda.cuInit(0)

# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)

# Create context
err, context = cuda.cuCtxCreate(0, cuDevice)

# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
# Note: Incompatible --gpu_architdcture would be detected here
err, module = cuda.cudModuleLoadData(ptx.ctypes.data)
ASSERT_DRV(err)
err, kernel = cuda.cuModuleGetFunction(module, b"matmul_smem_pad")
ASSERT_DRV(err)

# 定义两个输入矩阵和输出矩阵
m, n, p = 3072*8, 1024*8, 2048*8
a = np.random.randn(m, n).astype(np.float32)
b = np.random.randn(n, p).astype(np.float32)
c = np.zeros((m, p), dtype=np.float32)

# 将矩阵传输到GPU上
a_gpu = cuda.cuMemAlloc(a.nbytes)
b_gpu = cuda.cuMemAlloc(b.nbytes)
c_gpu = cuda.cuMemAlloc(c.nbytes)

err, stream = cuda.cuStreamCreate(0)

err, = cuda.cuMemcpyHtoDAsync(a_gpu, a.ctypes.data, a.nbytes, stream)
err, = cuda.cuMemcpyHtoDAsync(b_gpu, b.ctypes.data, b.nbytes, stream)

block_size = (32, 32, 1)
grid_size = ((m + block_size[0] - 1) // block_size[0],
             (p + block_size[1] - 1) // block_size[1],
             1)

args = [a_gpu, b_gpu, c_gpu, np.array(m, dtype=np.uint32), np.array(p, dtype=np.uint32), np.array(n, dtype=np.uint32)]
args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

start = cuda.CUevent()
end = cuda.CUevent()
cuda.cuEventRecordWithFlags(start, stream, cuda.cuda.CUevent_wait_flags(0))
start_gpu = time.time()
# 执行GPU内核并将结果传回CPU
err, = cuda.cuLaunchKernel(
    kernel,
    grid_size[0],
    grid_size[1],
    grid_size[2],
    block_size[0],
    block_size[1],
    block_size[2],
    0,
    stream,
    args.ctypes.data,
    0,
)
ASSERT_DRV(err)
cuda.cuEventRecordWithFlags(end, stream, cuda.cuda.CUevent_wait_flags(0))
cuda.cuStreamSynchronize(stream)
end_gpu = time.time()
time_gpu = end_gpu - start_gpu
method_name = "GPU(shared memory padded by c func wrapped)"
print(method_name + " time: " + str(time_gpu))

printa=True

a_cpu = a.dot(b)
if printa:
    print("a_cpu[:2,:10]:\n", a_cpu[:2, :10])
    print("a_cpu[-2:,-10:]:\n", a_cpu[-2:, -10:])

err, = cuda.cuMemcpyDtoHAsync(c.ctypes.data, c_gpu, c.nbytes, stream)
if printa:
    print("c[:2,:10]:\n", c[:2, :10])
    print("c[-2:,-10:]:\n", c[-2:, -10:])

# 检查结果是否正确
assert np.allclose(c, a_cpu, atol=1e-3, rtol=1e-3)

"""
GPU(shared memory padded by c func wrapped) time: 13.282296875
a_cpu[:2,:10]:
 [[ -57.660015    127.483505      6.783642     94.10514     -70.92239
  -169.08203      55.700172    -58.500206    -75.752625     22.159966  ]
 [ -33.580822   -119.04451      -0.84903526  -16.444998     11.244322
   -71.28312      29.219736     25.338657    -75.94582    -112.15874   ]]
a_cpu[-2:,-10:]:
 [[-127.67889     76.08202     32.1596       1.4845405   79.91324
    95.30013     56.031006    24.943783    -9.5245075  114.24239  ]
 [ 196.51324   -198.12738    -32.292614   -82.63124     48.575916
  -111.82152     -2.192215  -118.80109    -61.37823     56.134453 ]]
c[:2,:10]:
 [[ -57.660114    127.48322       6.783642     94.10511     -70.92228
  -169.08194      55.700157    -58.50033     -75.75276      22.159983  ]
 [ -33.580784   -119.04455      -0.84907025  -16.444979     11.244301
   -71.283195     29.219744     25.338707    -75.94578    -112.15875   ]]
c[-2:,-10:]:
 [[-127.67891     76.08209     32.159515     1.4846147   79.91322
    95.30012     56.031055    24.94377     -9.524599   114.242424 ]
 [ 196.51306   -198.1273     -32.29248    -82.63133     48.57592
  -111.821335    -2.1922727 -118.8012     -61.377895    56.134247 ]]
"""