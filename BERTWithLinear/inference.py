# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     inference
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/1
   Software:      PyCharm
'''
import argparse

import torch
from transformers import BertTokenizer

from BERTWithLinear.model import BERTForPunctuator

'''
单句推理模式
    1. 提取单句特征
    2. 构造模型
    3. 加载checkpoint
    4. 更改模型模式
    5. 零梯度预测
    6. argmax
    7. 读取字典
    8. 输出结果
'''
parser = argparse.ArgumentParser()

parser.add_argument("--ckp", default="./checkpoint_v2/epoch2.pt", help="where the model saved")
parser.add_argument("--label-vocab", default="../dataset/processed_for_bert/label.dict.tsv")
parser.add_argument("--bert-path", default="./pretrained_bert", help="where the pretrained bert model saved")
parser.add_argument("--device", default="cpu", help="whether use cpu or gpu")

args = parser.parse_args()

if __name__ == '__main__':
    sentence = "yes it must be confessed"
    sentence.lower()
    sentence = sentence.split(" ")

    device = torch.device(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    ids = tokenizer.convert_tokens_to_ids(sentence)

    input = torch.tensor(ids, dtype=torch.long)

    checkpoint = torch.load(args.ckp, map_location=device)
    # print(checkpoint)
    model = BERTForPunctuator(5, device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(input.unsqueeze(0))
    pred_ids = torch.argmax(pred.squeeze(0), 1).numpy().tolist()

    # load vocab
    with open(args.label_vocab, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    label_vocab = {int(e.split("\t")[1]): e.split("\t")[0][0] for e in lines}

    pred_label = [label_vocab.get(id) for id in pred_ids]

    res = []
    for word, label in zip(sentence, pred_label):
        res.append(f"{word}{label if label != '_' else ''}")

    print(" ".join(res))

