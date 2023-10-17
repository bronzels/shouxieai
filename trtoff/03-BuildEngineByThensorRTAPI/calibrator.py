import os
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibrationDataPath, nCalibration, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.imageList = glob(calibrationDataPath + "*.jpg")[:100]
        self.nCalibration = nCalibration
        self.shape = inputShape # (N,C,H,W)
        self.bufferSize = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile = cacheFile
        _, self.dIn = cudart.cudaMalloc(self.bufferSize)
        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.nCalibration):
            print(" > calibration %d" % i)
            subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            res[i, 0] = cv2.imread(imageList[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return res

    def get_batch_size(self):  # necessary API
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        try:
            data = next(self.oneBatch)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.bufferSize, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # necessary API
        if os.path.exists(self.cacheFile):
            print("Succeed finding cache file: %s" % self.cacheFile)
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache: %s" % self.cacheFile)
            return

    def write_calibration_cache(self, cache):  # necessary API
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache")
        return

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator("../00-MNISTData/test/", 5, (1, 1, 28, 28), "./int8.cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")

