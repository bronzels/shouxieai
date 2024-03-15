from transformers import BertTokenizer

import os
import pickle
import torch

from common import BERT_PATH
from common import ClsDataset

#读取数据
def readIMDB(path, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
       files = os.listdir(os.path.join(path, seg, label))
       for file in files:
          with open (os.path.join (path, seg, label, file), 'r', encoding='utf8') as rf:
              review = rf.read().replace('\n', '')
              if label == 'pos':
                 data.append([review, 1])
              elif label == 'neg':
                 data.append([review, 0])
    return data

if __name__ == '__main__':
    train_data = readIMDB('aclImdb')
    test_data = readIMDB('aclImdb', 'test')
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_max_len = 512
    train_dataset = ClsDataset(train_data, tokenizer, train_max_len)
    train_set, check_set = torch.utils.data.random_split(train_dataset, [20000, 5000])
    test_set = ClsDataset(test_data, tokenizer, train_max_len)

    with open('./data/train_dataset_save.pkl', 'wb') as f:
        pickle.dump(train_set, f)
    with open('./data/check_set_save.pkl', 'wb') as f:
        pickle.dump(check_set, f)
    with open('./data/test_set_save.pkl', 'wb') as f:
        pickle.dump(test_set, f)