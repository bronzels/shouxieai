from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import numpy as np
import torch
import json
import os
import time

class GenDataSet():
    def __init__(self, tokenizer, train_file, val_file, label_dict, max_length=128, batch_size=10):
        self.train_file = train_file
        self.val_file = val_file
        self.max_length = max_length
        self.batch_size = batch_size
        self.label_dict = label_dict
        self.tokenizer = tokenizer

    def gen_data(self, file):
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
            input_ids.append(input_id)
            input_types.append(types)
            input_masks.append(masks)
            y_ = self.label_dict[label]
            labels.append(y_)

            if index % 1000 == 0:
                print('处理', index, '条数据')

        data_gen = TensorDataset(torch.LongTensor(np.array(input_ids)),
                                 torch.LongTensor(np.array(input_types)),
                                 torch.LongTensor(np.array(input_masks)),
                                 torch.LongTensor(np.array(labels)))

        sampler = RandomSampler(data_gen)
        return DataLoader(data_gen, sampler=sampler, batch_size=self.batch_size)

    def get_train_data(self):
        return self.gen_data(self.train_file)

    def get_val_data(self):
        return self.gen_data(self.val_file)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from tqdm import tqdm

label_dict = {
    'fear': 0,
    'neutral': 1,
    'sad': 2,
    'surprise': 3,
    'angry': 4,
    'happy': 5
}

model_dir = 'chinese-roberta-wwm-ext'
train_file = 'dataset/train/usual_train.txt'
val_file = 'dataset/eval/virus_eval_labeled.txt'
save_model_path = './model/'
max_length = 128
num_classes = 6
batch_size = 10
epoch = 10

def val(model, device, data):
    model.eval()
    test_loss = 0.0
    acc = 0
    for (input_id, types, masks, y) in tqdm(data):
        input_id, types, masks, y = input_id.to(device), types.to(device), masks.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(input_id, token_type_ids=types, attention_mask=masks)
            logits = y_['logits']
        test_loss += nn.functional.cross_entropy(logits, y.squeeze())
        pred = logits.max(-1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(data)
    return acc / len(data.dataset)

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_classes)
    dataset = GenDataSet(tokenizer, train_file, val_file, label_dict, max_length, batch_size)
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    from torch.cuda.amp import autocast
    args_amp = True
    scaler = torch.cuda.amp.GradScaler() if args_amp else None
    """
    wn amp: 
        Train Epoch: 0 , cost 426.712070 second
    wth amp: 
        Train Epoch: 0 , cost 263.857049 second
    """

    best_acc = 0.0
    for epoch_index in range(epoch):
        batch_epoch = 0
        start_time = time.time()
        for (input_id, types, masks, y) in tqdm(train_data):
            input_id, types, masks, y = input_id.to(device), types.to(device), masks.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=scaler is not None):
                outputs = model(input_id, token_type_ids=types, attention_mask=masks, labels=y)
                loss = outputs.loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_epoch += 1
            if batch_epoch % 10 == 0:
                print('Train Epoch:', epoch_index, ', batch_epoch: ', batch_epoch, ', loss = ', loss.item())
        end_time = time.time()  # 记录程序结束运行时间
        print('Train Epoch:', epoch_index, ', cost %f second' % (end_time - start_time))

        acc = val(model, device, val_data)
        print('Train Epoch:', epoch_index, ' val acc = ', acc)
        if best_acc < acc:
            model.save_pretrained(save_model_path)
            tokenizer.save_pretrained(save_model_path)
            best_acc = acc

if __name__ == '__main__':
    main()




