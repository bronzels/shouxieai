import cv2
import numpy as np
from numba import cuda
import time
import math

@cuda.jit
def process_gpu(img, channels):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for c in range(channels):
        color = img[tx,ty][c] * 2.0 + 30
        if color > 255:
            img[tx,ty][c] = 255
        elif color < 0:
            img[tx,ty][c] = 0
        else:
            img[tx,ty][c] = color


def process_cpu(img, dst):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = img[i, j][c] * 2.0 + 30
                if color > 255:
                    dst[i,j][c] = 255
                elif color < 0:
                    dst[i,j][c] = 0
                else:
                    dst[i, j][c] = color

if __name__ == "__main__":
    img = cv2.imread('test-nvidia.jpg')
    rows, cols, channels = img.shape
    dst_cpu = img.copy()
    dst_gpu = img.copy()

    start_cpu = time.time()
    process_cpu(img, dst_cpu)
    end_cpu = time.time()
    time_cpu = end_cpu - start_cpu
    print("CPU process time: " + str(time_cpu))

    dImg = cuda.to_device(img)
    threadsperblock = (16, 16)
    blockspergrid_x = int(math.ceil(rows/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols/threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda.synchronize()
    start_gpu = time.time()
    process_gpu[blockspergrid,threadsperblock](dImg, channels)
    cuda.synchronize()
    dst_gpu = dImg.copy_to_host()
    end_gpu = time.time()
    time_gpu = end_gpu - start_gpu
    print("CPU process time: " + str(time_gpu))

    cv2.imwrite("result_cpu.jpg", dst_cpu)
    cv2.imwrite("result_gpu.jpg", dst_gpu)
    print("Done.")

"""
CPU process time: 1.2900402545928955
CPU process time: 0.12803125381469727
"""





