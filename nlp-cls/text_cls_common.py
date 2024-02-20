from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
import torch.nn as nn
import numpy as np
import torch
import json
import os
from tqdm import tqdm
import time

from dataloader import MyDataset, MyDataloader

label_dict = {
    'fear': 0,
    'neutral': 1,
    'sad': 2,
    'surprise': 3,
    'angry': 4,
    'happy': 5
}

save_model_path = './model'
#max_length = 128
max_length = 128
num_classes = 6
#batch_size = 32
#batch_size = 32
batch_size = 100
onnx_model_path = './onnx'

import random


class Mydataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.len

class GenDataSet():
    def __init__(self, tokenizer, file_dict, label_dict, max_length=128, batch_size=32):
        self.train_file = file_dict.get("train", None)
        self.val_file = file_dict.get("val", None)
        self.test_file = file_dict.get("test", None)
        self.max_length = max_length
        self.batch_size = batch_size
        self.label_dict = label_dict
        self.tokenizer = tokenizer

    def gen_data(self, file, tensor_ele=True):
        print('os.getcwd():', os.getcwd())
        print('file:', file)
        if not os.path.exists(file):
            raise Exception("数据集文件不存在")
        input_ids = []
        input_types = []
        input_masks = []
        labels = []
        with open(file, encoding='utf8') as f:
            data = json.load(f)
        if not data:
            raise Exception("数据集数据不存在")

        for index, item in enumerate(data):
            text = item['content']
            label = item['label']
            tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length)
            input_id, types, masks = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
            y_ = self.label_dict[label]
            input_ids.append(input_id)
            input_types.append(types)
            input_masks.append(masks)
            labels.append(y_)

            if index % 1000 == 0:
                print('处理', index, '条数据')

        if tensor_ele:
            data_gen = TensorDataset(torch.LongTensor(np.array(input_ids)),
                                     torch.LongTensor(np.array(input_types)),
                                     torch.LongTensor(np.array(input_masks)),
                                     torch.LongTensor(np.array(labels)))
            sampler = RandomSampler(data_gen)
            return DataLoader(data_gen, sampler=sampler, batch_size=self.batch_size)
        else:
            data_gen = MyDataset(np.array(input_ids).astype(np.int32),
                                     np.array(input_types).astype(np.int32),
                                     np.array(input_masks).astype(np.int32),
                                     np.array(labels).astype(np.int32))
            return MyDataloader(data_gen, self.batch_size)

    def get_train_data(self, tensor_ele=True):
        return self.gen_data(self.train_file, tensor_ele) if self.train_file is not None else None

    def get_val_data(self, tensor_ele=True):
        return self.gen_data(self.val_file, tensor_ele) if self.val_file is not None else None

    def get_test_data(self, tensor_ele=True):
        return self.gen_data(self.test_file, tensor_ele) if self.test_file is not None else None

def valtrans(model, device, data):
    model.eval()
    test_loss = 0.0
    acc = 0
    infertime = 0
    for (input_id, types, masks, y) in tqdm(data):
        input_id, types, masks, y = input_id.to(device), types.to(device), masks.to(device), y.to(device)
        with torch.no_grad():
            a = time.time()
            y_ = model(input_id, token_type_ids=types, attention_mask=masks)
            b = time.time()
            infertime += b - a
            logits = y_['logits']
        test_loss += nn.functional.cross_entropy(logits, y.squeeze())
        pred = logits.max(-1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(data)
    return acc / len(data.dataset), infertime

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def valtrans_onnx(sess, data):
    acc = 0
    infertime = 0
    for (input_id, types, masks, y) in tqdm(data):
        #print("input_id:\n", input_id)
        #print("types:\n", types)
        #print("masks:\n", masks)
        #print("y:\n", y)
        input_feed = {
            'input_ids': input_id,
            'token_type_ids': types,
            'attention_mask': masks,
        }
        #logits = sess.run(output_names=['logits'], input_feed=input_feed)
        a = time.time()
        logits = sess.run(output_names=None, input_feed=input_feed)
        b = time.time()
        infertime += b - a
        #pred = np.argmax(logits[0], axis=-1).astype(int)
        pred = np.argmax(softmax(logits[0]), axis=-1).astype(int)
        acc += np.sum(pred == y)
    return acc / len(data.dataset), infertime

