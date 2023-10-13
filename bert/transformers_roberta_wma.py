
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from text_cls_common import GenDataSet, valtrans, label_dict
from text_cls_common import save_model_path, max_length, num_classes, batch_size

model_dir = 'chinese-roberta-wwm-ext'
train_file = 'dataset/train/usual_train.txt'
val_file = 'dataset/eval/virus_eval_labeled.txt'
epoch = 1

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_classes)
    dataset = GenDataSet(tokenizer, {"train": train_file, "val": val_file}, label_dict, max_length, batch_size)
    train_data = dataset.get_train_data()
    val_data = dataset.get_val_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    #lr = 1e-5
    #decayRate = 0.96
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=decayRate)
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
            #my_lr_scheduler.step()
            batch_epoch += 1
            if batch_epoch % 10 == 0:
                print('Train Epoch:', epoch_index, ', batch_epoch: ', batch_epoch, ', loss = ', loss.item())
        end_time = time.time()  # 记录程序结束运行时间
        print('Train Epoch:', epoch_index, ', cost %f second' % (end_time - start_time))

        acc, _ = valtrans(model, device, val_data)
        print('Train Epoch:', epoch_index, ' val acc = ', acc)
        if best_acc < acc:
            model.save_pretrained(save_model_path)
            tokenizer.save_pretrained(save_model_path)
            best_acc = acc

if __name__ == '__main__':
    main()




