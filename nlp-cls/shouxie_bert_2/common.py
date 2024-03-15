BERT_PATH = 'bert_base_uncased'

import random
import torch
from torch.utils.data import Dataset

random.seed(1)
torch.manual_seed(1)

class ClsDataset(Dataset):
    #def __init__(self, path, tokenizer, train_max_len):
    def __init__(self, in_data, tokenizer, train_max_len):
        #self.path = path
        self.in_data = in_data
        self.tokenizer = tokenizer
        self.train_max_len = train_max_len
        self.feeatures = self.get_features()
        self.nums = len(self.feeatures)

    def get_features(self):
        features = []
        for text, label in self.in_data:
            inputs = self.tokenizer.encode_plus(
                text=text,
                max_length=self.train_max_len,
                padding="max_length",
                truncation="only_first",
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            features.append(
                (
                    inputs['input_ids'],
                    inputs['token_type_ids'],
                    inputs['attention_mask'],
                    int(label),
                )
            )
        return features

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            "token_ids": torch.tensor(self.feeatures[item][0]).long(),
            "token_type_ids": torch.tensor(self.feeatures[item][1]).long(),
            "attention_masks": torch.tensor(self.feeatures[item][2]).long(),
            "labels": torch.tensor(self.feeatures[item][3]).long(),
        }
        return data
'''
    def get_features(self):
        features = []
        with open(self.path, 'r') as fp:
            lines = fp.read().strip().split('\n')
            for i, line in enumerate(lines):
                line = line.split('\t')
                text = line[0]
                label = line[1]
                inputs = self.tokenizer.encode_plus(
                    text=text,
                    max_length=self.train_max_len,
                    padding="max_length",
                    truncation="only_first",
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                if i < 3:
                    print("input_ids:", str(inputs['input_ids']))
                    print("token_type_ids:", str(inputs['token_type_ids']))
                    print("attention_mask:", str(inputs['attention_mask']))
                    print("label:", label)
                features.append(
                    (
                        inputs['input_ids'],
                        inputs['token_type_ids'],
                        inputs['attention_mask'],
                        int(label),
                    )
                )
        return features
'''
