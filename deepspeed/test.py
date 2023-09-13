import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import torch.nn as nn
import torch.nn.functional as F

"""
deepspeed test.py --deepspeed_config config.json
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  test.py --deepspeed_config=config.json

deepspeed train.py --deepspeed_config=ds_config.json -p 1 --steps=200
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  train.py -p 2 --steps=200  --deepspeed_config=ds_config.json
"""
def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


deepspeed.init_distributed()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

args = add_argument()
net = Net()
parameters = filter(lambda p: p.requires_grad, net.parameters())
print(len(trainset))
#model_engine, optimizer, trainloader, __ = deepspeed.initialize(
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters)
    #args=args, model=net, model_parameters=parameters, training_data=trainset)

criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if i % args.log_interval == (args.log_interval - 1):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.log_interval))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images.to(model_engine.local_rank))

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))