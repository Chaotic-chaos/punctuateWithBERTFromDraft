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
    1. Embedding
        1) torch.nn.Embedding(30522<bert词表长度>, 768<bert词嵌入维度>)
    2. bLSTM
        1) torch.nn.LSTM()
    3. Linear
        1) torch.nn.Linear()
'''

from torch import nn
import torch.nn.functional as F

class LSTMForPunctuator(nn.Module):
    def __init__(self, label_size, device):
        super(LSTMForPunctuator, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.embedding = nn.Embedding(30522, 768)
        self.lstm = nn.LSTM(input_size=768, hidden_size=384, num_layers=10, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(768, label_size)
        self.device = device

    def forward(self, sentences):
        x = self.embedding(sentences)
        x = self.lstm(x)
        x = self.linear(x[0])

        x = F.log_softmax(x, dim=2)

        return x


if __name__ == '__main__':
    sentence = "hello i am a student"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokens = tokenizer.tokenize(sentence)

    ids = tokenizer.convert_tokens_to_ids(tokens)

    ids.append(0)
    # print(ids)
    input_sen = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    model = LSTMForPunctuator(5, torch.device("cpu"))

    pred = model(input_sen)

    label = torch.tensor([1, 1, 1, 1, 4, 0], dtype=torch.long)

    loss_func = nn.CrossEntropyLoss(ignore_index=0)

    loss = loss_func(pred.squeeze(), label)

    print(loss)

    loss.backward()



    # print(111)
