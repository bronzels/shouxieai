import os
from datetime import datetime as dt
from glob import glob
import time

import calibrator
import cv2
import numpy as np
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable

np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
paraFile = "./para.npz"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# for FP16 mode
bUseFP16Mode = False
# for INT8 mode
bUseINT8Mode = True
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

os.system("rm -rf ./*.npz ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = t.nn.Conv2d(1, 32, (5, 5), padding=(2,2), bias=True)
        self.conv2 = t.nn.Conv2d(32, 64, (5, 5), padding=(2,2), bias=True)
        self.fc1 = t.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.fc2 = t.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        z = F.softmax(y, dim=1)
        z = t.argmax(z, dim=1)
        return y, z

class MyData(t.utils.data.Dataset):

    def __init__(self, isTrain=True):
        if isTrain:
            self.data = trainFileList
        else:
            self.data = testFileList

    def __getitem__(self, index):
        imageName = self.data[index]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        index = int(imageName[-7])
        label[index] = 1
        return t.from_numpy(data.reshape(1, nHeight, nWidth).astype(np.float32)), t.from_numpy(label)

    def __len__(self):
        return len(self.data)

model = Net().cuda()
ceLoss = t.nn.CrossEntropyLoss()
opt = t.optim.Adam(model.parameters(), lr=0.001)
trainDataset = MyData(True)
testDataset = MyData(False)
trainLoader = t.utils.data.DataLoader(dataset=trainDataset, batch_size=nTrainBatchSize, shuffle=True)
testLoader = t.utils.data.DataLoader(dataset=testDataset, batch_size=nTrainBatchSize, shuffle=True)

for epoch in range(10):
    for xTrain, yTrain in trainLoader:
        xTrain = Variable(xTrain).cuda()
        yTrain = Variable(yTrain).cuda()
        opt.zero_grad()
        y_, z = model(xTrain)
        loss = ceLoss(y_, yTrain)
        loss.backward()
        opt.step()

    data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth)

    with t.no_grad():
        acc = 0
        n = 0
        cnt = 0
        for xTest, yTest in testLoader:
            xTest = Variable(xTest).cuda()
            yTest = Variable(yTest).cuda()
            a = time.time()
            y_, z = model(xTest)
            b = time.time()
            acc += t.sum(z == t.matmul(yTest, t.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda:0")))
            n += xTest.shape[0]
            cnt += 1
        print("%s, epoch %2d, loss = %f, test acc = %f" % (dt.now(), epoch + 1, loss.data, acc / n))
        xInfer = t.tensor(np.array(data)).cuda()
        a = time.time()
        y_, z = model(xInfer)
        b = time.time()
        print("pyTorch eval time = %f" % (b - a))
        print("z:", z)

    para = {}
    for name, parameter in model.named_parameters():
        para[name] = parameter.detach().cpu().numpy()
    np.savez(paraFile, **para)

    del para
    print("Succeed building model in pyTorch!")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if bUseINT8Mode:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

    inputTensor = network.add_input("inputT0", trt.float32, [-1, 1, nHeight, nWidth])
    profile.set_shape(inputTensor.name, [1, 1, nHeight, nWidth], [4, 1, nHeight, nWidth], [8, 1, nHeight, nWidth])
    config.add_optimization_profile(profile)

    para = np.load(paraFile)

    w = np.ascontiguousarray(para["conv1.weight"])
    b = np.ascontiguousarray(para["conv1.bias"])
    _0 = network.add_convolution_nd(inputTensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
    _0.padding_nd = [2, 2]
    _1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
    _2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
    _2.stride_nd = [2, 2]

    w = np.ascontiguousarray(para["conv2.weight"])
    b = np.ascontiguousarray(para["conv2.bias"])
    _3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
    _3.padding_nd = [2, 2]
    _4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
    _5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
    _5.stride_nd = [2, 2]

    _6 = network.add_shuffle(_5.get_output(0))
    _6.reshape_dims = (-1, 64 * 7 * 7)

    w = np.ascontiguousarray(para["fc1.weight"].transpose())
    b = np.ascontiguousarray(para["fc1.bias"].reshape(1, -1))
    _7 = network.add_constant(w.shape, trt.Weights(w))
    _8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
    _9 = network.add_constant(b.shape, trt.Weights(b))
    _10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
    _11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["fc2.weight"].transpose())
    b = np.ascontiguousarray(para["fc2.bias"].reshape(1, -1))
    _12 = network.add_constant(w.shape, trt.Weights(w))
    _13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
    _14 = network.add_constant(b.shape, trt.Weights(b))
    _15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

    _16 = network.add_softmax(_15.get_output(0))
    _16.axes = 1 << 1

    _17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

    network.mark_output(_17.get_output(1))

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    else:
        print("Succeeded building engine!")
    with open(trtFile, "wb") as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [1, 1, nHeight, nWidth])
    for i in range(nIO):
        print("[%2d]%s->" %(i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    a = time.time()
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], bufferD[i])

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    b = time.time()
    print("tensorRT eval time = %f" % (b - a))

    for i in range(nIO):
        print(lTensorName[i])
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)

    print("Succeeded running model in TensorRT!")

"""
TF32
2023-10-17 09:39:53.411248, epoch 10, loss = 0.010111, test acc = 0.950000
pyTorch eval time = 0.000389
tensorRT eval time = 0.000279

FP16
2023-10-17 09:42:43.263172, epoch 10, loss = 0.010111, test acc = 0.950000
pyTorch eval time = 0.000341
tensorRT eval time = 0.000219

INT8
2023-10-17 09:45:33.546946, epoch 10, loss = 0.010111, test acc = 0.950000
pyTorch eval time = 0.000356
tensorRT eval time = 0.000206
2023-10-17 09:47:23.938286, epoch 10, loss = 0.010111, test acc = 0.950000
pyTorch eval time = 0.000335
tensorRT eval time = 0.000205

C++
TP32
tensorRT eval time: 0.00050525
tensorRT eval time: 0.00049910

FP16
tensorRT eval time: 0.00046509
tensorRT eval time: 0.00047197

INT8
tensorRT eval time: 0.00042170
tensorRT eval time: 0.00045043

"""