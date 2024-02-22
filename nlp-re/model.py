import torch
import torch.nn as nn
from transformers import BertModel
from utils import get_idx2tag

class SentenceRE(nn.Module):
    def __init__(self, hparams):
        super(SentenceRE, self).__init__()
        self.bert_model = BertModel.from_pretrained(hparams.bert_path)

        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.dense = nn.Linear(hparams.embedding_dim, hparams.embedding_dim)
        self.drop = nn.Dropout(hparams.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(hparams.embedding_dim * 3)
        self.hidden2tag = nn.Linear(hparams.embedding_dim * 3, len(get_idx2tag(hparams.target_file)))

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask, return_dict=False)

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector

