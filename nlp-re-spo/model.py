import re
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from utils import START_TAG, STOP_TAG, tag2idx, idx2tag, relation2idx, idx2relation

device = "cuda" if torch.cuda.is_available() else 'cpu'

def log_sum_exp(smat):
    vmax = smat.max(dim=1, keepdim=True).values
    return (smat - vmax).exp().sum(axis=1, keepdim=True).log() + vmax

def entity_average(hidden_output, e_mask):
    e_mask_unsqueeze = e_mask.unsqueeze(1)
    length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
    sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
    avg_vector = sum_vector.float() / length_tensor.float()
    return avg_vector

class BertForRE(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForRE, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_dim = config.hidden_size

        self.tag2idx = tag2idx
        self.tag_size = len(tag2idx)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.transitions.data[tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, tag2idx[STOP_TAG]] = -10000

        self.target_size = len(relation2idx)
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.drop = nn.Dropout(0.1)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.hidden_dim * 3)
        self.hidden2label = nn.Linear(self.hidden_dim * 3, self.target_size)
        self.criterion = nn.CrossEntropyLoss()

    def get_features(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output
        return sequence_output, pooled_output

    def get_total_scores(self, frames, real_lengths):
        alpha = torch.full((frames.shape[0], self.tag_size), -10000.0).to(device)
        alpha[:, self.tag2idx[START_TAG]] = 0.
        alpha_ = torch.zeros((frames.shape[0], self.tag_size)).to(device)
        frames = frames.transpose(0, 1)
        index = 0
        for frame in frames:
            index += 1
            alpha = log_sum_exp(alpha.unsqueeze(-1) + frame.unsqueeze(1) + self.transitions.T).squeeze(1)
            for idx, length in enumerate(real_lengths):
                if length == index:
                    alpha_[idx] = alpha[idx]
        alpha_ = log_sum_exp(alpha_.unsqueeze(-1) + 0 + self.transitions[[self.tag2idx[STOP_TAG]], :].T).flatten()
        return alpha_

    def get_golden_scores(self, frames, tag_ids_batch, real_lengths):
        score = torch.zeros(tag_ids_batch.shape[0]).to(device)
        score_ = torch.zeros(tag_ids_batch.shape[0]).to(device)
        tags = torch.cat([torch.full([tag_ids_batch.shape[0], 1], self.tag2idx[START_TAG], dtype=torch.long).to(device),
                          tag_ids_batch], dim=1)
        index = 0
        for i in range(frames.shape[1]):
            index += 1
            frame = frames[:, i, :]
            score += self.transitions[tags[:, i + 1], tags[:, i]] + frame[
                range(frame.shape[0]), tags[:, i + 1]]

            for idx, length in enumerate(real_lengths):
                if length == index:
                    score_[idx] = score[idx]
        score_ = score_ + self.transitions[self.tag2idx[STOP_TAG], tags[:, -1]]
        return score_

    def viterbi_decode(self, frames):
        backtrace = []
        alpha = torch.full((1, self.tag_size), -10000.).to(device)
        alpha[0][self.tag2idx[START_TAG]] = 0
        for frame in frames:
            smat = alpha.T + frame.unsqueeze(0) + self.transitions.T
            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)
        smat = alpha.T + 0 + self.transitions[[self.tag2idx[STOP_TAG]], :].T
        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()
        #best_tag_id = idx.detach()

        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1]

    def get_relation_logit(self, pooled_output, hidden_output, sub_mask, obj_mask):
        sub_h = entity_average(hidden_output, sub_mask)
        obj_h = entity_average(hidden_output, obj_mask)
        sub_h = self.activation(self.dense(sub_h))
        obj_h = self.activation(self.dense(obj_h))
        concat_h = torch.cat([pooled_output, sub_h, obj_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2label(self.drop(concat_h))
        return logits

    def compute_loss(self, input_ids, attention_mask, tag_ids, sub_mask, obj_mask, labels, real_lengths):
        hidden_output, pooled_output = self.get_features(input_ids, attention_mask)
        feats = self.hidden2tag(hidden_output)
        total_scores = self.get_total_scores(feats, real_lengths)
        gold_score = self.get_golden_scores(feats, tag_ids, real_lengths)
        ner_loss = torch.mean(total_scores - gold_score)
        relation_logits = self.get_relation_logit(pooled_output, hidden_output, sub_mask, obj_mask)
        relation_loss = self.criterion(relation_logits, labels)
        return ner_loss + relation_loss

    @staticmethod
    def extract(chars, tags):
        result = []
        pre = ''
        w = []
        subjects = set()
        objects = set()
        for idx, tag in enumerate(tags):
            if not pre:
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
                else:
                    if tag == f'I-{pre}':
                        w.append(chars[idx])
                    else:
                        result.append([w, pre])
                        w = []
                        pre = ''
                        if tag.startswith('B'):
                            pre = tag.split('-')[1]
                            w.append(chars[idx])
        for res in result:
            if res[1] in {'SUB', 'BOTH'}:
                subjects.add(''.join(res[0]))
            if res[1] in {'OBJ', 'BOTH'}:
                objects.add(''.join(res[0]))
        return subjects, objects

    def predict(self, text, chars, input_ids, attention_mask):
        hidden_output, pooled_output = self.get_features(input_ids, attention_mask)
        feats = self.hidden2tag(hidden_output)
        feats = feats.squeeze(0)
        res_ner = self.viterbi_decode(feats)
        pred_tags = [idx2tag[ix] for ix in res_ner[1]]
        subjects, objects = self.extract(chars, pred_tags)
        print(f'Subject包含：{subjects}')
        print(f'Object包含：{objects}')

        length = len(text)
        for sub in subjects:
            for obj in objects:
                if obj != sub:
                    try:
                        sub_mask = [0] * length
                        start_sub, end_sub = re.search(sub, text).span()
                        sub_mask[start_sub:end_sub] = [1] * (end_sub - start_sub)
                        sub_mask = [0] + sub_mask + [0]
                        sub_mask = torch.tensor(sub_mask, dtype=torch.long).unsqueeze(0).to(device)

                        obj_mask = [0] * length
                        start_obj, end_obj = re.search(obj, text).span()
                        obj_mask[start_obj:end_obj] = [1] * (end_obj - start_obj)
                        obj_mask = [0] + obj_mask + [0]
                        obj_mask = torch.tensor(obj_mask, dtype=torch.long).unsqueeze(0).to(device)

                        logits = self.get_relation_logit(pooled_output, hidden_output, sub_mask, obj_mask)

                        relation = idx2relation[logits.argmax().item()]
                        print(f'{sub} - {relation} - {obj}')
                    except:
                        print(f'{sub} - {obj}: 向量提取异常')

