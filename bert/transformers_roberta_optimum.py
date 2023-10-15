import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

import time

from text_cls_common import num_classes, save_model_path, onnx_model_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(save_model_path, num_labels=num_classes)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(save_model_path)
    vanilla_clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    a = time.time()
    output = vanilla_clf("这是什么鬼地方，服务太差了。")
    b = time.time()
    print(output)
    print('vanilla_clf latency:%.4f' % (b-a))

    ort_model = ORTModelForSequenceClassification.from_pretrained(save_model_path, from_transformers=True)
    onnx_path = onnx_model_path + '/optimum'
    ort_model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    optimizer = ORTOptimizer.from_pretrained(ort_model)
    optimization_config = OptimizationConfig(optimization_level=99)
    optimizer.optimize(
        save_dir=onnx_path,
        optimization_config=optimization_config
    )
    optimized_model = ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model_optimized.onnx")
    optimized_model = optimized_model.to(device)
    optimized_clf = pipeline("text-classification", model=optimized_model, tokenizer=tokenizer, device=device)
    a = time.time()
    output = optimized_clf("这个酒店太棒了，我喜欢。")
    b = time.time()
    print(output)
    print('optimized_clf latency:%.4f' % (b-a))

if __name__ == '__main__':
    main()

"""
[{'label': 'LABEL_4', 'score': 0.9592370986938477}]
vanilla_clf latency:0.4620
[{'label': 'LABEL_5', 'score': 0.9591822624206543}]
optimized_clf latency:0.0101

[{'label': 'LABEL_4', 'score': 0.9592370986938477}]
vanilla_clf latency:0.4625
[{'label': 'LABEL_5', 'score': 0.9591822624206543}]
optimized_clf latency:0.0037

[{'label': 'LABEL_4', 'score': 0.9592370986938477}]
vanilla_clf latency:0.4639
{'label': 'LABEL_5', 'score': 0.9591822624206543}]
optimized_clf latency:0.0037

--minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
--optShapes=input_ids:100x128,attention_mask:100x128,token_type_ids:100x128 \
--maxShapes=input_ids:100x128,attention_mask:100x128,token_type_ids:100x128 \

trtexec \
--onnx=/data0/shouxieai/bert/onnx/optimum/model.onnx \
--saveEngine=/data0/shouxieai/bert/trt/bert_optimum.trt \
--shapes=input_ids:100x128,attention_mask:100x128,token_type_ids:100x128 \
--memPoolSize=workspace:8192 \
--useCudaGraph \
--device=0

"""
