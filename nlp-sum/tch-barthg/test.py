import logging

import jieba
import lawrouge
import numpy as np
import datasets
import torch
from datasets import load_dataset, Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from torch.utils.data import dataloader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, BertTokenizer, \
    BartForConditionalGeneration

import mypickle

logger = logging.getLogger("bart-large-chinese")
logging.basicConfig(level=logging.INFO)
dataset = mypickle.read_pickle_data("./", "dataset_test")
TokenModel = "fnlp"
tokenizer = BertTokenizer.from_pretrained(TokenModel)
model = BartForConditionalGeneration.from_pretrained(r'model')

def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }

max_input_length = 1024
max_target_length = 128
model_inputs = tokenizer(dataset[0]["document"], max_length=max_input_length, padding="max_length", truncation=True)



def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dd = datasets.DatasetDict({"test": dataset})
test_batch_size = 1
tokenized_datasets = dd.map(preprocess_function, batched=True)
args = Seq2SeqTrainingArguments(
    fp16=True,
    output_dir=r'./',
    do_eval=True,
    per_device_eval_batch_size=test_batch_size,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["test"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
)

eval_dataloader = trainer.get_eval_dataloader()

model.to("cuda:0")

rouge = Rouge()
rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0

num_return_sequences = 4

for i, batch in enumerate(tqdm(eval_dataloader)):
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids=batch["input_ids"].to("cuda:0"),
            attention_mask=batch["attention_mask"].to("cuda:0"),
            early_stopping=True,
            num_beams=num_return_sequences,
            length_penalty=1.,
            # no_repeat_ngram_size=0,
            # diversity_penalty=0.,
            num_return_sequences=1,
            num_beam_groups=1,
            max_length=64,
        )
    #
    lsp = []
    for s in range(test_batch_size):
        lsp.append(int(s * num_return_sequences))  # num_return_sequences
    outputs = output[lsp, :]
    labels = batch["labels"]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    decoded_preds = [" ".join((pred.replace(" ", ""))) for pred in decoded_preds]
    decoded_labels = [" ".join((label.replace(" ", ""))) for label in decoded_labels]
    for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
        scores = rouge.get_scores(hyps=decoded_pred, refs=decoded_label)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        bleu += sentence_bleu(
            references=[decoded_label.split(' ')],
            hypothesis=decoded_pred.split(' '),
            smoothing_function=SmoothingFunction().method1
        )
bleu /= len(dataset)
rouge_1 /= len(dataset)
rouge_2 /= len(dataset)
rouge_l /= len(dataset)
