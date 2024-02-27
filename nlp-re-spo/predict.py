import torch
from transformers import BertTokenizer
from model import BertForRE
import time
from config import parsers

def pred(text):
    opt = parsers()

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model = BertForRE.from_pretrained(opt.save_pretrained_best)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(opt.bert_file)
    startTime = time.time()

    chars = [char for char in text]

    chars = ['[CLS]'] + chars + ['[SEP]']
    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
    attention_mask = [1] * len(input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        model.predict(text, chars, input_ids, attention_mask)

    endTime = time.time()
    print(f"运行时间：{endTime-startTime}s")

if __name__ == '__main__':
    text = "葛淑珍离开赵本山18年，走出自己的路。"
    pred(text)
    text = "除演艺事业外，李冰冰热心公益，发起并亲自参与多项环保慈善活动，积极投身其中，身体力行担起了回馈社会的责任于02年出演《少年包青天》，进入大家视线"
    pred(text)