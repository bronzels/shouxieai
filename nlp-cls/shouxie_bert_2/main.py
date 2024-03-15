import torch
import torch.nn as nn
from transformers import BertTokenizer
import math

import os

import pickle
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, classification_report

from common import BERT_PATH
from common import ClsDataset

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

## 定义BerEmbeddings模块
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, device, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.device = device
        self.max_len = max_len
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(2, hidden_size)
        self.pos_embeddings = nn.Embedding(self.max_len, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)
    def forward(self, input_token, segment_ids):
        token_embeddings = self.token_embeddings(input_token)
        type_embeddings = self.type_embeddings(segment_ids)
        pos_ids = []
        input_count = list(input_token.size())[0]
        max_len = list(input_token.size())[1]
        for i in range(input_count):
            tmp = [x for x in range(max_len)]
            pos_ids.append(tmp)
        pos_ids = torch.tensor(pos_ids).to(self.device)
        postion_embeddings = self.pos_embeddings(pos_ids)
        embedding_x = token_embeddings + type_embeddings + postion_embeddings
        embedding_x = self.emb_normalization(embedding_x)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x



## 定义Encoder模块
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.feedforward_act = GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, attention_x):
        attention_x = self.dense1(attention_x)
        attention_x = self.feedforward_act(attention_x)
        attention_x = self.dense2(attention_x)
        attention_x = self.dropout(attention_x)
        return attention_x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 dropout_prob=0.1
                 ):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size
        # 申明网络
        self.q_dense = nn.Linear(self.out_dim, self.out_dim)
        self.k_dense = nn.Linear(self.out_dim, self.out_dim)
        self.v_dense = nn.Linear(self.out_dim, self.out_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)
    def forward(self, x, attention_mask):
        qx = x
        kx = x
        vx = x
        q = self.q_dense(qx)
        k = self.k_dense(kx)
        v = self.v_dense(vx)
        # 先将batch_size*seq_len*embedding_size变成batch_size*seq_len*head*head_size
        # 再将batch_size*seq_len*head*head_size转置成batch_size*head*seq_len*head_size
        shape = list(x.size())
        batch_size = shape[0]
        seq_len = shape[1]
        q = q.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        q = q.transpose(1, 2)
        k = k.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        k = k.transpose(1, 2)
        v = v.view([batch_size, seq_len, self.attention_head_num, self.attention_head_size])
        v = v.transpose(1, 2)
        # q与k的转置相乘得到：[batch_size, head, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(2, 3))
        # 因为q、k相乘，结果变大，因此对结果除以根号512
        attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))
        # 防止padding补全的0经过softmax后影响结果，对每个0值都加一个很大的负数，这样softmax后也会约等于0
        # attention_mask的shape为：[batch_size, seq_len, seq_len]
        if attention_mask is not None:
            add_mask = (1.0 - attention_mask.float()) * 1e5
            add_mask = add_mask[:, None, :, :]
            attention_scores -= add_mask
        attention_scores = self.softmax(attention_scores)
        attention_scores = self.dropout(attention_scores)
        attention_scores = torch.matmul(attention_scores, v)
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        attention_scores = attention_scores.view([batch_size, seq_len, self.out_dim])
        attention_scores = self.o_dense(attention_scores)
        return attention_scores

class Transformer(nn.Module):
    def __init__(self,
                 hidden_size,
                 attention_head_num,
                 attention_head_size,
                 intermediate_size,
                 dropout_prob=0.1
                 ):
        super(Transformer, self).__init__()
        self.multi_attention = MultiHeadSelfAttention(
            attention_head_num=attention_head_num,
            attention_head_size=attention_head_size)
        self.attention_layernorm = nn.LayerNorm(hidden_size)
        self.feedforward = FeedForward(
                hidden_size,
                intermediate_size,
                dropout_prob)
        self.feedforward_layernorm = nn.LayerNorm(hidden_size)
    def forward(self, x, attention_mask):
        attention_x = self.multi_attention(x, attention_mask)
        attention_x = x + attention_x
        attention_x = self.attention_layernorm(attention_x)
        feedforward_x = self.feedforward(attention_x)
        feedforward_x = attention_x + feedforward_x
        feedforward_x = self.feedforward_layernorm(feedforward_x)
        return feedforward_x

## 定义BertPooler模块
class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Config1:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.hidden_size = 768
        self.attention_head_num = 12
        self.attention_head_size = self.hidden_size // self.attention_head_num
        self.intermediate_size = 3072
        self.num_hidden_layers = 12
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.AttentionMask = True
        self.max_len = 512
        self.layer_norm_eps = 1e-12
        self.hidden_act = "gelu"

class MyBertModel(nn.Module):
    def __init__(self, config):
        super(MyBertModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.attention_head_num = config.attention_head_num
        self.attention_head_size = config.attention_head_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.device = config.device
        self.AttentionMask = config.AttentionMask
        self.max_len = config.max_len
        # 申明网络
        self.bert_emd = BertEmbeddings(vocab_size=self.vocab_size, max_len=self.max_len,
                                             hidden_size=self.hidden_size, device=self.device)
        self.bert_encoder = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(self.device)
            for _ in range(self.num_hidden_layers)
        )
        self.pooler = BertPooler(self.hidden_size)

    def gen_attention_masks(self, attention_mask):
        size = list(attention_mask.size())
        batch = size[0]
        max_len = size[1]
        process_attention_mask = torch.zeros(batch, max_len, max_len, requires_grad=False)
        true_len = torch.sum(attention_mask, dim=1)
        for i in range(batch):
            process_attention_mask[i, :true_len[i], :true_len[i]] = 1
        return process_attention_mask

    def load_local2target(self):
        local2target_emb = {
            'bert_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'bert_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
            'bert_emd.pos_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'bert_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.gamma',
            'bert_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.beta'
        }
        local2target_encoder = {
            'bert_encoder.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
            'bert_encoder.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
            'bert_encoder.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
            'bert_encoder.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
            'bert_encoder.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
            'bert_encoder.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
            'bert_encoder.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
            'bert_encoder.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
            'bert_encoder.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.gamma',
            'bert_encoder.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.beta',
            'bert_encoder.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
            'bert_encoder.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
            'bert_encoder.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
            'bert_encoder.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
            'bert_encoder.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.gamma',
            'bert_encoder.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.beta',
        }
        local2target_pooler = {
            "pooler.dense.weight": "bert.pooler.dense.weight",
            "pooler.dense.bias": "bert.pooler.dense.bias",
        }
        return local2target_emb, local2target_encoder, local2target_pooler

    def load_pretrain(self, sen_length, path):
        local2target_emb, local2target_encoder, local2target_pooler = self.load_local2target()
        pretrain_model_dict = torch.load(path)
        if sen_length == 512:
            finetune_model_dict = self.state_dict()
            new_parameter_dict = {}
            # 加载embedding层参数
            for key in local2target_emb:
                local = key
                target = local2target_emb[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            # 加载encoder层参数
            for i in range(self.num_hidden_layers):
                for key in local2target_encoder:
                    local = key % i
                    target = local2target_encoder[key] % i
                    new_parameter_dict[local] = pretrain_model_dict[target]
            # 加载pooler层参数
            for key in local2target_pooler:
                local = key
                target = local2target_pooler[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            finetune_model_dict.update(new_parameter_dict)
            self.load_state_dict(finetune_model_dict)
        else:
            raise Exception('输入预训练模型的长度错误')

    def forward(self, input_token, segment_ids, attention_mask):
        embedding_x = self.bert_emd(input_token, segment_ids)
        if self.AttentionMask:
            attention_mask = self.gen_attention_masks(attention_mask).to(self.device)
        else:
            attention_mask = None
        feedforward_x = None
        for i in range(self.num_hidden_layers):
            if i == 0:
                feedforward_x = self.bert_encoder[i](embedding_x, attention_mask)
            else:
                feedforward_x = self.bert_encoder[i](feedforward_x, attention_mask)
        sequence_output = feedforward_x
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


def main1():
    #tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    tokenizer = BertTokenizer.from_pretrained('./bert_base_uncased/')
    config = Config1(tokenizer.vocab_size)
    model = MyBertModel(config)
    model.load_pretrain(512, os.path.join(BERT_PATH, 'pytorch_model.bin'))
    model.eval()
    model.to(config.device)
    for name, param in model.named_parameters():
        print(name)


class Config2:
    def __init__(self, vocab_size, bert_dir, hidden_dropout_prob, lr):
        self.vocab_size = vocab_size
        self.hidden_size = 768
        self.attention_head_num = 12
        self.attention_head_size = self.hidden_size // self.attention_head_num
        self.intermediate_size = 3072
        self.num_hidden_layers = 12
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.AttentionMask = True
        self.max_len = 512
        self.layer_norm_eps = 1e-12
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = hidden_dropout_prob
        # 以下的是训练的一些参数
        self.bert_dir = bert_dir
        self.train_max_len = 512  # 32
        self.batch_size = 4
        self.train_epochs = 5
        self.lr = lr
        self.num_labels = 2
        self.namefig = 'dropout%flr%f' % (hidden_dropout_prob, lr)
        self.output_dir = './checkpoints'  + "/" + self.namefig + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.use_pretrained = True

class BertModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(BertModelForSequenceClassification, self).__init__()
        self.bert = MyBertModel(config)
        # 加载预训练参数
        if config.use_pretrained:
            self.bert.load_pretrain(config.max_len, os.path.join(config.bert_dir, 'pytorch_model.bin'))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                ):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        self.device = args.device
        self.model = BertModelForSequenceClassification(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self):
        accumulation_steps = 8
        best_dev_micro_f1 = 0.0
        loss_list = []
        dev_list = []
        pth_temp = self.args.namefig + ".txt"
        f = open(pth_temp, 'w+')
        stopa = 0
        stopb = 0
        for epoch in range(self.args.train_epochs):
            # if stopa == 1:
            #     break
            for batch_idx, train_data in enumerate(self.train_loader):
                # f.flush()
                # stopb = stopb + 1
                # if stopb > 20:
                #     stopa = 1
                #     break
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                loss = loss / accumulation_steps  # 梯度积累
                loss.backward()
                if ((batch_idx + 1) % accumulation_steps) == 0:
                    # 每 8 次更新一下网络中的参数
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if ((batch_idx + 1) % accumulation_steps) == 1:
                    loss_list.append([loss.item(), batch_idx])
                    msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                        epoch + 1, batch_idx, len(self.train_loader), 100. *
                        batch_idx / len(self.train_loader), loss.item()
                    )
                    print(msg)
                    print(msg, file=f)
                if batch_idx == len(self.train_loader) - 1:
                    # 在每个 Epoch 的最后输出一下结果
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, precision, recall, f1 = self.get_metrics(dev_outputs, dev_targets)
                    dev_list.append([accuracy, precision, recall, f1])
                    msg = "loss：{:.4f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} f1：{:.4f}".format(
                        dev_loss, accuracy, precision, recall, f1)
                    print(msg)
                    print(msg, file=f)
                    if f1 > best_dev_micro_f1:
                        msg = "------------>保存当前最好的模型"
                        print(msg)
                        print(msg, file=f)
                        best_dev_micro_f1 = f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        torch.save(self.model.state_dict(), checkpoint_path)
        pth_temp = self.args.namefig + "loss_list.pkl"
        with open(pth_temp, 'wb') as f:
            pickle.dump(loss_list, f)
        pth_temp = self.args.namefig + "dev_list.pkl"
        with open(pth_temp, 'wb') as f:
            pickle.dump(dev_list, f)
        f.close()
        checkpoint_path = os.path.join(self.args.output_dir, 'bestresult.pt')
        torch.save(self.model.state_dict(), checkpoint_path)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, args):
        model = self.model
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.train_max_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
            if len(outputs) != 0:
                return outputs[0]
            else:
                return '未识别'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs, average='micro')
        recall = precision_score(targets, outputs, average='micro')
        micro_f1 = f1_score(targets, outputs, average='micro')
        return accuracy, precision, recall, micro_f1

    def get_classification_report(self, outputs, targets):
        report = classification_report(targets, outputs)
        return report


def main2():
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    # 加载字典、词集合
    # with open('./data/train_set_save.pkl', 'rb') as f:
    #     train_dataset = pickle.load(f)
    with open('./data/check_set_save.pkl', 'rb') as f:
        check_dataset = pickle.load(f)
    with open('./data/test_set_save.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    with open('./data/train_dataset_save.pkl', 'rb') as f:
        train_dataset_temp = pickle.load(f)
    # 构建数据集和数据迭代器，设定 batch_size 大小为 4
    train_loader = DataLoader(dataset=train_dataset_temp,
                              batch_size=4,
                              shuffle=True)
    eval_loader = DataLoader(dataset=check_dataset,
                             batch_size=4,
                             shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=4,
                             shuffle=False)
    ## 调试超参数
    # test_parameter = [[0, 2e-5], [0.1, 2e-5], [0.5, 2e-5], [0.1, 2e-4], [0.1, 2e-6]]
    # for i in range(len(test_parameter)):
    #     hidden_dropout_prob, lr = test_parameter[i]
    #     config = Config(tokenizer.vocab_size, BERT_PATH, hidden_dropout_prob, lr)
    #     trainer = Trainer(config, train_loader, eval_loader, test_loader)
    #     trainer.train()
    ## 使用最优参数在Trainset上重新训练模型
    result_parameter = [[0.1, 5e-6]]
    hidden_dropout_prob, lr = result_parameter[0]
    config = Config2(tokenizer.vocab_size, BERT_PATH, hidden_dropout_prob, lr)
    trainer = Trainer(config, train_loader, eval_loader, test_loader)
    trainer.train()
    # 在测试集上测试模型性能
    checkpoint_path = os.path.join(config.output_dir, 'best.pt')
    #checkpoint_path = './result/bestresult.pt'
    total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    report = trainer.get_classification_report(test_targets, test_outputs)
    print(report)

if __name__ == '__main__':
  main1()
  main2()

'''
loss：367.0629 accuracy：0.9180 precision：0.9180 recall：0.9180 f1：0.9180
              precision    recall  f1-score   support

           0       0.91      0.95      0.93     12028
           1       0.95      0.92      0.93     12972

    accuracy                           0.93     25000
   macro avg       0.93      0.93      0.93     25000
weighted avg       0.93      0.93      0.93     25000

'''