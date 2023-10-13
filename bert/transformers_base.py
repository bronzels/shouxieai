import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import transformers as tfs
import math
import warnings
import time

warnings.filterwarnings('ignore')

from torch.cuda.amp import autocast
args_amp=True
scaler = torch.cuda.amp.GradScaler() if args_amp else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
save_model_path = './bert-base-uncased'
pth_saved_path = save_model_path + '/tuned_2cls_model.pth'

class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences):
        with autocast(enabled=scaler is not None):
            batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                               max_length=66, padding=True)
            input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
            attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
            bert_cls_hidden_state = bert_output[0][:,0,:]
            linear_output = self.dense(bert_cls_hidden_state)
        return linear_output

def get_data():
    #https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv
    train_df = pd.read_csv('train.tsv', delimiter='\t', header=None)

    train_set = train_df[:3000]
    print("Train set shape:", train_set.shape)
    train_set[1].value_counts()

    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(sentences, targets)
    return train_inputs, test_inputs, train_targets, test_targets

def valbase(model, test_inputs, test_targets):
    total = len(test_inputs)
    hit = 0
    infertime = 0
    with torch.no_grad():
        a = time.time()
        outputs = model(test_inputs)
        b = time.time()
        infertime += b - a
        _, pred = torch.max(outputs, 1)
        hit += np.sum(np.array(pred.cpu().numpy().tolist()) == np.array(test_targets))
    return hit / total * 100, infertime
"""        
        for i in range(total):
            outputs = model([test_inputs[i]])
            _, predicted = torch.max(outputs, 1)
            if predicted == test_targets[i]:
                hit += 1
"""

def main():
    train_inputs, test_inputs, train_targets, test_targets = get_data()

    batch_count = int(len(train_inputs) / batch_size)
    batch_train_inputs, batch_train_targets = [], []
    for i in range(batch_count):
        batch_train_inputs.append(train_inputs[i*batch_size:(i+1)*batch_size])
        batch_train_targets.append(train_targets[i*batch_size:(i+1)*batch_size])

    #device = torch.device("cuda")
    #device = torch.device("cpu")

    #epochs = 3
    epochs = 15
    #lr = 0.01
    lr = 0.001
    print_every_batch = 5
    bert_classifier_model = BertClassificationModel().to(device)
    optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr, momentum=0.7)
    criterion = nn.CrossEntropyLoss()
    """
    3， 0.001， 0.9
    Batch: 35, Loss: 0.2281
    cost 34.051260 second
    Accuracy: 88.93%
    
    3， 0.001， 0.7
    Batch: 35, Loss: 0.2959
    cost 34.135480 second
    Accuracy: 88.53%
    
    15， 0.001， 0.9
    Batch: 35, Loss: 0.0009
    cost 170.172588 second
    Accuracy: 90.00%
    
    Batch: 35, Loss: 0.0023
    cost 81.291749 second
    Accuracy: 87.07%
    """
    start_time = time.time()
    for epoch in range(epochs):
        print_avg_loss = 0
        for i in range(batch_count):
            inputs = batch_train_inputs[i]
            labels = torch.tensor(batch_train_targets[i]).to(device)
            optimizer.zero_grad()
            with autocast(enabled=scaler is not None):
                outputs = bert_classifier_model(inputs)
                loss = criterion(outputs, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            print_avg_loss += loss.item()
            if i % print_every_batch == (print_every_batch-1):
                print("Batch: %d, Loss: %.4f" % ((i+1), print_avg_loss/print_every_batch))
                print_avg_loss = 0
    end_time = time.time()  # 记录程序结束运行时间
    print('cost %f second' % (end_time - start_time))

    print("Accuracy: %.2f%%" % valbase(bert_classifier_model, test_inputs, test_targets))

    torch.save(bert_classifier_model, pth_saved_path)
    model = torch.load(pth_saved_path)
    print("Accuracy saved/loaded : %.2f%%" % valbase(model, test_inputs, test_targets))

if __name__ == '__main__':
    main()

