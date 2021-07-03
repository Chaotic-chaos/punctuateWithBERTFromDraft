# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     model
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/6/24
   Software:      PyCharm
'''
import torch
from transformers import BertModel, BertTokenizer

'''
模型结构文件
    1. BERT from huggingface: bert-base-uncased
        1) input: {
            "input_ids": 转为id的输入序列 / bsz * sequence_length
            "attention_mask": mask矩阵 / bsz * sequence_length
        }
        2) last_hidden_state: bsz * sequence_length * 768
    2. Linear Layer from nn.Linear
        1) dimension: 768 * output_classes
        2) input: bert.last_hidden_state
'''

from torch import nn
import torch.nn.functional as F

class BERTForPunctuator(nn.Module):
    def __init__(self, label_size, device):
        super(BERTForPunctuator, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("./pretrained_bert")
        self.linear = nn.Linear(768, label_size)
        self.device = device

    def forward(self, sentences):
        attention_mask = torch.sign(sentences)
        attention_mask = attention_mask.to(self.device)
        input = {
            "input_ids": sentences,
            "attention_mask": attention_mask
        }

        x = self.bert(**input)
        x = self.linear(x.last_hidden_state)

        pred = F.log_softmax(x, dim=-1)

        return pred


if __name__ == '__main__':
    sentence = "hello i am a student"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokens = tokenizer.tokenize(sentence)

    ids = tokenizer.convert_tokens_to_ids(tokens)

    ids.append(0)
    # print(ids)
    input_sen = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    model = BERTForPunctuator(5)

    pred = model(input_sen)

    label = torch.tensor([1, 1, 1, 1, 4, 0], dtype=torch.long)

    loss_func = nn.CrossEntropyLoss()

    loss = loss_func(pred.squeeze(), label)

    print(loss)

    loss.backward()



    # print(111)
