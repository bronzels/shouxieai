# !/usr/bin/env Python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re
import random
import numpy as np
import math

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)
sentence = re.sub("[,.!?\\-]", "", text.lower()).split("\n")
vocab = " ".join([i for i in sentence])
vocab = list(set([i for i in vocab.split(" ")]))
word2idx = {'MASK': 0, 'CLS': 1, 'SEQ': 2, 'PAD': 3}
for i in range(len(vocab)):
    word2idx[vocab[i]] = i + 4
idx2word = {i: j for i, j in enumerate(word2idx)}
vocab_size = len(idx2word)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


token_list = []
for i in range(len(sentence)):
    token_list.append([word2idx[j] for j in sentence[i].split(" ")])

max_len = 30
num_pred = 5
batch_size = 6
n_layers = 6
embedding_size = 768
segments_len = 2
embed_size = 768
dim = 64
num_heads = 12
d_ff = 64
dropout = 0.5

class my_dataset(Dataset):
    def __init__(self, input_ids, segment_ids, masked_pos, masked_tokens, isNext):
        super().__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_pos = masked_pos
        self.masked_tokens = masked_tokens
        self.isNext = isNext

    def __getitem__(self, index):
        return self.input_ids[index], self.segment_ids[index], self.masked_pos[index], self.masked_tokens[index], \
               self.isNext[index]

    def __len__(self):
        return self.input_ids.size(0)

def make_data(seq_data):
    batch = []
    left_cnt = right_cnt = 0
    while left_cnt < batch_size / 2 or right_cnt < batch_size / 2:
        rand_a_idx = rand_b_idx = 0
        sen_a_idx = random.randrange(len(seq_data))
        sen_b_idx = random.randrange(len(seq_data))
        tokens_a = seq_data[sen_a_idx]
        tokens_b = seq_data[sen_b_idx]
        input_ids = [word2idx['CLS']] + tokens_a + [word2idx['SEQ']] + tokens_b + [word2idx['SEQ']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        n_pred = min(num_pred, int(len(input_ids) * 0.15))
        test_input_ids = [i for i, j in enumerate(input_ids) if idx2word[j] != 'CLS' and idx2word[j] != 'SEQ']
        random.shuffle(test_input_ids)
        masked_tokens, masked_pos = [], []
        for word in range(n_pred):
            cand_rep_idx = test_input_ids[word]
            masked_pos.append(cand_rep_idx)
            masked_tokens.append(input_ids[cand_rep_idx])
            p = random.random()
            #if p > 0.8:
            if p > 0.2:
                input_ids[cand_rep_idx] = word2idx['MASK']
            elif p > 0.1:
                other_idx = random.randrange(len(input_ids))
                input_ids[cand_rep_idx] = input_ids[other_idx]
            #else:
            #    input_ids[cand_rep_idx] = input_ids[word]

        n_pad = max_len - len(input_ids)
        #input_ids.extend(n_pad * [0])
        input_ids.extend(n_pad * [word2idx['PAD']])
        segment_ids.extend(n_pad * [0])

        if num_pred > n_pred:
            n_pad = num_pred - n_pred
            masked_pos.extend(n_pad * [0])
            masked_tokens.extend(n_pad * [0])

        if sen_a_idx + 1 != sen_b_idx and left_cnt < batch_size / 2:
            isNext = False
            left_cnt = left_cnt + 1
            batch.append([input_ids, segment_ids, masked_pos, masked_tokens, isNext])
        elif sen_a_idx + 1 == sen_b_idx and right_cnt < batch_size / 2:
            isNext = True
            right_cnt = right_cnt + 1
            batch.append([input_ids, segment_ids, masked_pos, masked_tokens, isNext])
    return batch

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(dim)
        scores.masked_fill_(attn_mask, 0)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(embed_size, dim * num_heads, bias=False)
        self.W_K = nn.Linear(embed_size, dim * num_heads, bias=False)
        self.W_V = nn.Linear(embed_size, dim * num_heads, bias=False)
        self.projection = torch.nn.Linear(num_heads * dim, embed_size)

    def forward(self, input_Q, input_K, input_V, atten_mask):
        torch.backends.cudnn.enabled = False
        residual = input_Q
        _, len_q, embeding_size = input_Q.size()
        _, len_k, _ = input_K.size()
        Batch_size, atten_len_k, atten_len_v = atten_mask.size()
        Q = self.W_Q(input_Q).view(Batch_size, num_heads, len_q, dim)
        K = self.W_K(input_K).view(Batch_size, num_heads, len_k, dim)
        V = self.W_V(input_V).view(Batch_size, num_heads, len_k, dim)

        atten_mask = atten_mask.unsqueeze(1)
        atten_mask = atten_mask.repeat(1, num_heads, 1, 1)
        atten = ScaledDotProductAttention()(Q, K, V, atten_mask)
        atten = atten.transpose(1, 2)

        atten = atten.reshape(Batch_size, atten_len_k, -1)
        atten = self.projection(atten).to(device)
        atten_ret = (residual + torch.softmax(atten, dim=1))
        atten_ret = nn.LayerNorm(embed_size).to(device)(atten_ret)
        return atten_ret

class Feed_forward(nn.Module):

    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(embed_size, d_ff).to(device)
        self.W2 = nn.Linear(d_ff, embed_size).to(device)
        self.b1 = torch.rand(d_ff).to(device)
        self.b2 = torch.rand(embed_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_inputs):
        fc1 = self.W1(enc_inputs) + self.b1
        fc1 = self.relu(fc1)
        fc2 = self.W2(fc1) + self.b2
        output = fc2
        residual = enc_inputs
        #Add_And_Norm = nn.LayerNorm(embed_size).cuda()(output + residual)
        Add_And_Norm = nn.LayerNorm(embed_size).to(device)(output + residual)
        return Add_And_Norm

class Encoder_layer(nn.Module):

    def __init__(self):
        super().__init__()
        self.multi_head_attention = Multi_Head_Attention()
        self.feed_forward = Feed_forward()

    def forward(self, enc_inputs, enc_atten_mask):
        atten_output = self.multi_head_attention(enc_inputs, enc_inputs, enc_inputs, enc_atten_mask)
        output = self.feed_forward(atten_output).to(device)
        return output, atten_output

def get_attn_pad_mask(seq_q, seq_k):
    Batch_size, len_q = seq_q.size()
    Batch_size, len_k = seq_k.size()
    atten_mask = seq_k.eq(0).unsqueeze(1)
    atten_mask = atten_mask.expand(Batch_size, len_q, len_k)
    return atten_mask

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocab_size, embedding_size).to(device)
        self.pos_embed = torch.nn.Embedding(max_len, embedding_size).to(device)
        self.seg_embed = torch.nn.Embedding(segments_len, embedding_size).to(device)
        self.layers = nn.ModuleList(Encoder_layer() for _ in range(n_layers))
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(embedding_size, 2)
        self.fc2 = nn.Linear(embedding_size, vocab_size)
        self.linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, input_token, segments_, masked_pos):
        Batch_size, seq_len = input_token.size()
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(Batch_size, 1).to(device)
        input_token_embed = self.token_embed(input_token) + self.seg_embed(segments_) + self.pos_embed(pos)
        enc_atten_mask = get_attn_pad_mask(input_token, input_token)
        output = input_token_embed
        for layer in self.layers:
            output, _ = layer(output, enc_atten_mask)
        _, seq_len, _ = output.size()
        nsp_output = output
        nsp_output = self.fc1(nsp_output)
        nsp_output = self.classifier(nsp_output)
        nsp_output = torch.sum(nsp_output.transpose(2,1), dim=-1)
        nsp_output = torch.softmax(nsp_output, dim=-1)

        masked_pos = masked_pos.unsqueeze(-1).repeat(1, 1, embedding_size)
        nlm_output = torch.gather(output, 1, masked_pos)
        nlm_output = gelu(self.linear(nlm_output))
        nlm_output = self.fc2(nlm_output)
        return nsp_output, nlm_output

def train():
    batch = make_data(token_list)
    input_ids, segment_ids, masked_pos, masked_tokens, isNext = zip(*batch)
    input_ids, segment_ids, masked_pos, masked_tokens, isNext = torch.LongTensor(input_ids), torch.LongTensor(
        segment_ids), torch.LongTensor(masked_pos), torch.LongTensor(masked_tokens), torch.LongTensor(isNext)
    train_data = my_dataset(input_ids, segment_ids, masked_pos, masked_tokens, isNext)
    train_iter = DataLoader(train_data, batch_size, shuffle=True)
    crition = torch.nn.CrossEntropyLoss()
    model = BERT().train()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for step in range(1000):
        for input_ids_i, segment_ids_i, masked_pos_i, masked_tokens_i, isNext_i in train_iter:
            input_ids_i, segment_ids_i, masked_pos_i, masked_tokens_i, isNext_i = input_ids_i.to(device), \
                                                                                  segment_ids_i.to(device), \
                                                                                  masked_pos_i.to(device), \
                                                                                  masked_tokens_i.to(device), \
                                                                                  isNext_i.to(device)
            optimizer.zero_grad()
            nsp_out, nlm_out = model(input_ids_i, segment_ids_i, masked_pos_i)
            classify_loss = crition(nsp_out, isNext_i)
            masked_tokens_i = masked_tokens_i.view(-1)
            nlm_out = nlm_out.view(-1, vocab_size)
            nlm_out = nlm_out
            nlm_loss = 0
            nlm_loss = crition(nlm_out, masked_tokens_i)
            nlm_loss = nlm_loss.mean()
            loss = nlm_loss + classify_loss
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("step {0} loss {1} loss {2}".format(step, nlm_loss, classify_loss))

if __name__ == '__main__':
    train()



