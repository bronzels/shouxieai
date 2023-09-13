from numba import guvectorize, jit, cuda, float32
import numba
import numpy
import math
import time
#pycuda集成copy到host全部都是nan，单独没问题。

def compare_matrixes(A, B):
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            #if A[x,y] != B[x,y]:
            #if abs(A[x,y] - B[x,y]) > 1e-6: global memory
            if abs(A[x,y] - B[x,y]) > 1e-4: #shared memory
                print("Matrixes are different");
                print("(%d,%d) - %f | %f " %(x, y, A[x,y], B[x,y]))
                return
    print("Matrixes are same")
TPB = 32
TPB_plus1 = 33

def matmul_cpu(A, B):
    return numpy.matmul(A, B)

"""
@numba.jit(nopython=True)
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0
            for k in range(A.shape[1]):
                tmp += A[x,k]*B[k,y]
            C[x,y] = tmp

@guvectorize('float32[:,:], float32[:,:], float32[:,:]', '(m,n),(n,p)->(m,p)', target='cuda', nopython=True)
@jit(target_backend="cuda", nopython=True)
def matmul_gpu2(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0
            for k in range(A.shape[1]):
                tmp += A[x,k]*B[k,y]
            C[x,y] = tmp
和cpu一样非常慢
"""

@cuda.jit
def matmul_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit
def matmul_shared_mem(A, B, C):
    sA = cuda.shared.array(shape=(TPB,TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB,TPB), dtype=float32)

    x,y=cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(int(A.shape[1]/TPB)):
        sA[tx, ty] = A[x, ty+i*TPB]
        sB[tx, ty] = B[tx+i*TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx,j]*sB[j,ty]
        cuda.syncthreads()
    C[x,y] = tmp

@cuda.jit
def matmul_shared_mem_pad(A, B, C):
    #直接加1报错，必须用常量
    sA = cuda.shared.array(shape=(TPB,TPB_plus1), dtype=float32)
    sB = cuda.shared.array(shape=(TPB,TPB_plus1), dtype=float32)

    x,y=cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(int(A.shape[1]/TPB)):
        sA[tx, ty] = A[x, ty+i*TPB]
        sB[tx, ty] = B[tx+i*TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx,j]*sB[j,ty]
        cuda.syncthreads()
    C[x,y] = tmp

"""
m = TPB*3
k = TPB*1
n = TPB*2
printa = False
numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
"""
m = 3072*8
k = 1024*8
n = 2048*8
printa = True

A = numpy.random.random((m,k))#numpy.full((m, k), 3, numpy.float32)
B = numpy.random.random((k,n))#numpy.full((k, n), 3, numpy.float32)
C_cpu = numpy.zeros((m, n), dtype=numpy.float32)
print("A.shape:", A.shape)
print("B.shape:", B.shape)
print("C_cpu.shape:", C_cpu.shape)

print("Start processing in CPU")
start_cpu = time.time()
#matmul_cpu(A,B,C_cpu)
C_cpu = matmul_cpu(A,B)
end_cpu = time.time()
time_cpu = ( end_cpu - start_cpu)
print("CPU time： "+str(time_cpu))
if printa:
    print("C_cpu[:2,:10]:\n", C_cpu[:2,:10])
    print("C_cpu[-2:,-10:]:\n", C_cpu[-2:,-10:])


threadsperblock = (TPB, TPB, 1)
blockspergrid_x = int(math.ceil(A.shape[0]/threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1]/threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y, 1)

#gpu_a_a = []
for i in range(3,4,1):
    if i == 0:
        method_name = "GPU(global memory)"
        kernel = matmul_gpu
    elif i == 1:
        method_name = "GPU(shared memory)"
        kernel = matmul_shared_mem
    else:
        method_name = "GPU(shared memory padded)"
        kernel = matmul_shared_mem_pad

    print("Start processing in ", method_name)
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_mem = cuda.device_array((A.shape[0], B.shape[1]))
    start_gpu = time.time()
    kernel[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_mem)
    cuda.synchronize()
    end_gpu = time.time()
    C_gpu = C_mem.copy_to_host()
    time_gpu = end_gpu - start_gpu
    print(method_name + " time: " + str(time_gpu))

    print("Compare CPU with " + method_name)
    compare_matrixes(C_gpu, C_cpu)
    #gpu_a_a.append(C_gpu)
    if printa:
        print("C_gpu[:2,:10]:\n", C_gpu[:2, :10])
        print("C_gpu[-2:,-10:]:\n", C_gpu[-2:, -10:])

#compare_matrixes(gpu_a_a[0], gpu_a_a[1])
#compare_matrixes(gpu_a_a[1], gpu_a_a[2])

#print("Compare GPU global with GPU smem")
#compare_matrixes(C_global_gpu, C_shared_gpu)

"""
matrixes: 3072 * 1024, 1024 * 2048; block: 16 * 16 
Start processing in CPU
CPU(manua) time： 23.659889698028564
Start processing in  GPU(global memory)
GPU(global memory) time: 0.24089908599853516
Compare CPU with GPU(global memory)
Matrixes are same
Start processing in  GPU(shared memory)
GPU(shared memory) time: 0.21659564971923828
Compare CPU with GPU(shared memory)
Matrixes are same
Start processing in  GPU(shared memory padded)
GPU(shared memory padded) time: 0.21481561660766602
Compare CPU with GPU(shared memory padded)
Matrixes are same

matrixes: 3072*8 * 1024*8 , 1024*8 * 2048*8; block: 16 * 16 
Start processing in  GPU(global memory)
GPU(global memory) time: 66.88991641998291
Start processing in  GPU(shared memory)
GPU(shared memory) time: 75.89680361747742
Start processing in  GPU(shared memory padded)
GPU(shared memory padded) time: 75.41626691818237
Matrixes are same
Matrixes are same

matrixes: 3072*8 * 1024*8 , 1024*8 * 2048*8; block: 32 * 32 
Start processing in CPU
CPU(numpy) time： 46.0459623336792
Start processing in  GPU(global memory)
GPU(global memory) time: 68.0925920009613
Start processing in  GPU(shared memory)
GPU(shared memory) time: 88.5304069519043
Start processing in  GPU(shared memory padded)
GPU(shared memory padded) time: 78.57171964645386
Matrixes are same
Matrixes are same


"""