from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test_deepspeed import evaluate

import warnings
warnings.filterwarnings("ignore")

from terminaltables import AsciiTable

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import deepspeed

"""
deepspeed test.py --deepspeed_config config.json
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  test.py --deepspeed_config=config.json

deepspeed train.py --deepspeed_config=ds_config.json -p 1 --steps=200
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  train.py -p 2 --steps=200  --deepspeed_config=ds_config.json
"""

"""
deepspeed train-deepspeed.py --deepspeed_config config.json
deepspeed train-deepspeed.py --deepspeed_config zero-stage2.json
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  train-deepspeed.py --deepspeed_config=config.json
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  train-deepspeed.py --deepspeed_config=zero-stage2.json --master_port=29600
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  train-deepspeed.py --deepspeed_config=zero-stage3.json --master_port=29600
deepspeed --hostfile=/data0/shouxieai/deepspeed/hostfile  --include="dtpct:0@mdubu:0"  train-deepspeed.py --deepspeed_config=config-stage3.json

1个epoch

single card, wn deepspeed, 
    1.2hrs(estimated, error due to memory, never finish)
cpu,  
    13hrs(estimated, didn't finish)
2 cards on 2 hosts, wth deepspeed(wth zero3 offload), 
    1days 3hrs(estimated, run 1 day, manually stop, didn't finish)
2 cards on 2 hosts, wth deepspeed(wn zero3 offload), 
    1days 2.3hrs(estimated, run 1 day, manually stop, didn't finish)
single card, wth deepspeed(wth zero3 offload), 
    2.6hrs(finished)
single card, wth deepspeed(wn zero3 offload), 
    1.7hrs(estimated, manually stop)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    #parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    #parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/darknet53.conv.74", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    #parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    """
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
    parser.add_argument('--master_port',
                        type=int,
                        default=29500)
    """
    parser = deepspeed.add_config_arguments(parser)
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    deepspeed.init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        #pin_memory=False,
        collate_fn=dataset.collate_fn,
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    #model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        #args = opt, model = model, model_parameters = parameters, training_data = dataset)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=opt, model=model, model_parameters=parameters)

    #optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
        #for batch_i, (_, imgs, targets) in enumerate(trainloader):
            batches_done = len(dataloader) * epoch + batch_i

            #imgs = Variable(imgs.to(device))
            #targets = Variable(targets.to(device), requires_grad=False)
            imgs, targets = Variable(imgs.to(model_engine.local_rank)), Variable(targets.to(
                model_engine.local_rank))
            #print ('imgs',imgs.shape)
            #print('targets', targets.shape)
            #loss, outputs = model(imgs, targets)
            loss, outputs = model_engine(imgs, targets)
            #loss.backward()
            model_engine.backward(loss)

            if batches_done % opt.gradient_accumulations:
                #optimizer.step()
                model_engine.step()
                optimizer.zero_grad()


            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                #iou_thres=0.5,
                iou_thres=0.1,
                conf_thres=0.5,
                #conf_thres=0.8,
                nms_thres=0.5,
                #nms_thres=0.4,
                img_size=opt.img_size,
                batch_size=2,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


