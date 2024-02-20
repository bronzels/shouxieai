import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

label_dict = {
    0: '恐惧',
    1: '无情绪',
    2: '悲伤',
    3: '惊奇',
    4: '愤怒',
    5: '积极'
}
model_dir = "./model"
num_classes = 6
max_length = 128

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_classes)

    while True:
        text = input("请输入内容： \n ")
        if not text or text == "":
            continue
        if text == "q":
            break

        encoded_input = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)

        input_ids = torch.tensor([encoded_input['input_ids']])
        token_type_ids = torch.tensor([encoded_input['token_type_ids']])
        attention_mask = torch.tensor([encoded_input['attention_mask']])
        y_ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = y_['logits'][0]
        pred = output.max(-1, keepdim=True)[1][0].item()
        print('预测结果：', label_dict[pred])

if __name__ == '__main__':
    main()
