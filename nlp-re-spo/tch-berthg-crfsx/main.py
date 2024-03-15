import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as tud
from transformers import AdamW, get_linear_schedule_with_warmup

import predict
from model import BertForRE
from utils import NERDataset, tag2idx, relation2idx
from config import parsers

def train(opt, train_loader, dev_loader, optimizer, scheduler):
    dev_loss_max = float("inf")
    for epoch in range(opt.epochs):
        model.train()
        flag = False
        epoch_loss = []
        pbar = tqdm(train_loader)
        pbar.set_description("[Train Epoch {}]".format(epoch))
        for batch_idx, batch_data in enumerate(pbar):
            input_ids = batch_data['input_ids'].to(device)
            tag_ids = batch_data['tag_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            sub_mask = batch_data['sub_mask'].to(device)
            obj_mask = batch_data['obj_mask'].to(device)
            labels = batch_data['labels'].to(device)
            real_lens = batch_data['real_lens']

            model.zero_grad()
            loss = model.compute_loss(input_ids, attention_mask, tag_ids, sub_mask, obj_mask, labels, real_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

        train_mean = np.mean(epoch_loss)
        dev_loss_mean = dev(dev_loader, epoch)

        if dev_loss_mean < dev_loss_max:
            dev_loss_max = dev_loss_mean
            model.save_pretrained(f'{opt.save_pretrained_best}')
            torch.save(model.state_dict(), opt.save_model_best)
            flag = True

        print('[Train Epoch {}/{}] train_loss: {:.4f} dev_loss: {:.4f} save_best_model: {}'.format(epoch, opt.epochs,
                                                                                        train_mean, dev_loss_mean,
                                                                                        "*" if flag else ""))
        text = "葛淑珍离开赵本山18年，走出自己的路。"
        predict.pred(text)
        text = "除演艺事业外，李冰冰热心公益，发起并亲自参与多项环保慈善活动，积极投身其中，身体力行担起了回馈社会的责任于02年出演《少年包青天》，进入大家视线"
        predict.pred(text)

def dev(dev_loader, epoch):
    model.eval()
    dev_loss = []
    pbar = tqdm(dev_loader)
    pbar.set_description("[Dev Epoch {}]".format(epoch))
    for batch_idx, batch_data in enumerate(pbar):
        input_ids = batch_data['input_ids'].to(device)
        tag_ids = batch_data['tag_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        sub_mask = batch_data['sub_mask'].to(device)
        obj_mask = batch_data['obj_mask'].to(device)
        labels = batch_data['labels'].to(device)
        real_lens = batch_data['real_lens']

        with torch.no_grad():
            loss = model.compute_loss(input_ids, attention_mask, tag_ids, sub_mask, obj_mask, labels, real_lens)
            dev_loss.append(loss.item())

    return np.mean(dev_loss)

if __name__ == '__main__':
    opt = parsers()

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    train_dataset = NERDataset(opt.train_file, opt.bert_file, tag2idx, relation2idx)
    train_dataloader = tud.DataLoader(train_dataset, opt.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_dataset = NERDataset(opt.dev_file, opt.bert_file, tag2idx, relation2idx)
    dev_dataloader = tud.DataLoader(dev_dataset, opt.batch_size, shuffle=True, collate_fn=dev_dataset.collate_fn)

    test_dataset = NERDataset(opt.test_file, opt.bert_file, tag2idx, relation2idx)
    test_dataloader = tud.DataLoader(test_dataset, opt.batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

    model = BertForRE.from_pretrained(opt.bert_file)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    total_steps = len(train_dataloader) * opt.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learn_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    train(opt, train_dataloader, dev_dataloader, optimizer, scheduler)
    test_loss_mean = dev(test_dataloader, opt.epochs - 1)
    print("last epoch test loss:", test_loss_mean)

"""
[Dev Epoch 8]: 100%|██████████████████████████████████████████████████████| 625/625 [01:39<00:00,  6.27it/s]
[Train Epoch 8/10] train_loss: 4.2139 dev_loss: 4.6490 save_best_model: 
Subject包含：{'赵本山', '葛淑珍'}
Object包含：set()
运行时间：0.003706216812133789s
Subject包含：set()
Object包含：{'李冰冰'}
运行时间：0.006755828857421875s
[Train Epoch 9]: 100%|██████████████████████████████████████████████████| 1875/1875 [15:48<00:00,  1.98it/s]
[Dev Epoch 9]: 100%|██████████████████████████████████████████████████████| 625/625 [01:39<00:00,  6.28it/s]
[Train Epoch 9/10] train_loss: 4.1425 dev_loss: 4.7163 save_best_model: 
Subject包含：{'赵本山', '葛淑珍'}
Object包含：set()
运行时间：0.0037178993225097656s
Subject包含：set()
Object包含：{'李冰冰'}
运行时间：0.00676417350769043s
[Dev Epoch 9]: 100%|██████████████████████████████████████████████████████| 625/625 [01:39<00:00,  6.29it/s]
last epoch test loss: 4.716859968566895

"""