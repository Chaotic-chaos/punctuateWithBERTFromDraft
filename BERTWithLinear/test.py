# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     test
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/6
   Software:      PyCharm
'''
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics

from model import BERTForPunctuator
from dataLoader import PunctuationDataset, collate_fn

'''
模型测试
    1. 加载dataset
    2. 构造dataloader
    3. 构造模型
    4. 加载ckp
    5. 预测batch
    6. argmax得分
    7. 获取每个batch的预测值，标签值，剔除padding元素
    8. 使用sklearn工具包计算评价指标
'''

parser = argparse.ArgumentParser()

parser.add_argument("--test-set", default="../dataset/processed_for_bert/test.tsv", help="test dataset path")
parser.add_argument("--label-vocab", default="../dataset/processed_for_bert/label.dict.tsv", help="label vocabulary path")
parser.add_argument("--label-size", default=5, help="label dimension")
parser.add_argument("--batch-size", default=20, help="batch size")
parser.add_argument("--device", default="cpu", help="whether use gpu or not")
parser.add_argument("--ckp", default="./checkpoint/epoch20.pt", help="where the model saved")
# test mode: 1-with _SPACE 2-without _SPACE
parser.add_argument("--mode", default=1, help="evaluate the model with _SPACE or not")

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device(args.device)

    print("Loading Data...")
    test_dataset = PunctuationDataset(input_path=args.test_set, label_vocab_path=args.label_vocab)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)


    # build model
    print("Building Model...")
    model = BERTForPunctuator(args.label_size, device)

    # load ckp
    checkpoint = torch.load(args.ckp, map_location=device)
    model.load_state_dict(checkpoint["model"])

    # move to device
    model = model.to(device)

    # set eval mode
    model.eval()

    print(model)

    labels_all = []
    preds_all = []
    for sentences, labels in tqdm(test_dataloader, desc="[Testing]"):
        # move data to device
        sentences = sentences.to(device)
        # labels = labels.to(device)

        # forward
        with torch.no_grad():
            preds = model(sentences)

        # make deduce
        scores = torch.argmax(preds, dim=2)
        scores = scores.cpu().numpy()
        labels = labels.numpy()
        nonzero_elem = np.nonzero(labels)

        for row, column in zip(nonzero_elem[0], nonzero_elem[1]):
            # print(f"pred: {scores[row, column]}")
            # print(f"ground_truth: {labels[row, column]}")
            preds_all.append(scores[row, column])
            labels_all.append(labels[row, column])

        # for debug
        break

    assert len(preds_all) == len(labels_all), "Predict labels don't match the Ground truth"
    labels_name = [",COMMA", ".PERIOD", "?QUESTIONMARK", "_SPACE"]

    if args.mode == 2:
        # clean the _SPACE: 4
        clean_res = list(filter(lambda x: 4 not in x, [(pred, label) for pred, label in zip(preds_all, labels_all)]))
        preds_all = list(zip(*clean_res))[0]
        labels_all = list(zip(*clean_res))[1]
        assert len(preds_all) == len(labels_all), "Predict labels don't match the Ground truth"
        labels_name = [",COMMA", ".PERIOD", "?QUESTIONMARK"]
        # print(1)

    print(metrics.classification_report(y_true=labels_all, y_pred=preds_all, target_names=labels_name))
