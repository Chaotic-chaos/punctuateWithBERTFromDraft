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
from numpy import nan
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
    8. 转为包含标点标签的句子
    9. 调用baseline错误率计算方法进行计算
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


def compute_error_stream(ground_truth, predictions):
    SPACE = "_SPACE"
    MAPPING = {}

    PUNCTUATION_VOCABULARY = {",COMMA", ".PERIOD", "?QUESTIONMARK", SPACE}
    PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ",COMMA", ";SEMICOLON": ",COMMA", "-DASH": ""}

    counter = 0
    total_correct = 0

    correct = 0.
    substitutions = 0.
    deletions = 0.
    insertions = 0.

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    # for target_path, predicted_path in zip(target_paths, predicted_paths):

    target_punctuation = " "
    predicted_punctuation = " "

    t_i = 0
    p_i = 0

    # with open(target_path, 'r', encoding='utf-8') as target, open(predicted_path, 'r',
    #                                                               encoding='utf-8') as predicted:

    target_stream = ground_truth
    predicted_stream = predictions

    while True:

        if PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
            while PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[
                t_i]) in PUNCTUATION_VOCABULARY:  # skip multiple consecutive punctuations
                target_punctuation = PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                target_punctuation = MAPPING.get(target_punctuation, target_punctuation)
                t_i += 1
        else:
            target_punctuation = " "

        if predicted_stream[p_i] in PUNCTUATION_VOCABULARY:
            predicted_punctuation = MAPPING.get(predicted_stream[p_i], predicted_stream[p_i])
            p_i += 1
        else:
            predicted_punctuation = " "

        is_correct = target_punctuation == predicted_punctuation

        counter += 1
        total_correct += is_correct

        if predicted_punctuation == " " and target_punctuation != " ":
            deletions += 1
        elif predicted_punctuation != " " and target_punctuation == " ":
            insertions += 1
        elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation == target_punctuation:
            correct += 1
        elif predicted_punctuation != " " and target_punctuation != " " and predicted_punctuation != target_punctuation:
            substitutions += 1

        true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + float(is_correct)
        false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + float(
            not is_correct)
        false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + float(
            not is_correct)

        assert target_stream[t_i] == predicted_stream[p_i] or predicted_stream[p_i] == "<unk>", \
            ("File: %s \n" + \
             "Error: %s (%s) != %s (%s) \n" + \
             "Target context: %s \n" + \
             "Predicted context: %s") % \
            (target_path,
             target_stream[t_i], t_i, predicted_stream[p_i], p_i,
             " ".join(target_stream[t_i - 2:t_i + 2]),
             " ".join(predicted_stream[p_i - 2:p_i + 2]))

        t_i += 1
        p_i += 1

        if t_i >= len(target_stream) - 1 and p_i >= len(predicted_stream) - 1:
            break

    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print("-" * 46)
    print("{:<16} {:<9} {:<9} {:<9}".format('PUNCTUATION', 'PRECISION', 'RECALL', 'F-SCORE'))
    for p in PUNCTUATION_VOCABULARY:

        if p == SPACE:
            continue

        overall_tp += true_positives.get(p, 0.)
        overall_fp += false_positives.get(p, 0.)
        overall_fn += false_negatives.get(p, 0.)

        punctuation = p
        precision = (true_positives.get(p, 0.) / (
                    true_positives.get(p, 0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p, 0.) / (
                    true_positives.get(p, 0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan
        print(u"{:<16} {:<9} {:<9} {:<9}".format(punctuation, round(precision, 3) * 100, round(recall, 3) * 100,
                                                 round(f_score, 3) * 100).encode('utf-8'))
    print("-" * 46)
    pre = overall_tp / (overall_tp + overall_fp) if overall_fp else nan
    rec = overall_tp / (overall_tp + overall_fn) if overall_fn else nan
    f1 = (2. * pre * rec) / (pre + rec) if (pre + rec) else nan
    print("{:<16} {:<9} {:<9} {:<9}".format("Overall", round(pre, 3) * 100, round(rec, 3) * 100, round(f1, 3) * 100))
    print("Err: %s%%" % round((100.0 - float(total_correct) / float(counter - 1) * 100.0), 2))
    print(
        "SER: %s%%" % round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1))

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
        # break

    assert len(preds_all) == len(labels_all), "Predict labels don't match the Ground truth"

    ################################################## 旧版计算方法，暂时废弃 ###################################################################
    # labels_name = [",COMMA", ".PERIOD", "?QUESTIONMARK", "_SPACE"]

    # if args.mode == 2:
    #     # clean the _SPACE: 4
    #     clean_res = list(filter(lambda x: 4 not in x, [(pred, label) for pred, label in zip(preds_all, labels_all)]))
    #     preds_all = list(zip(*clean_res))[0]
    #     labels_all = list(zip(*clean_res))[1]
    #     assert len(preds_all) == len(labels_all), "Predict labels don't match the Ground truth"
    #     labels_name = [",COMMA", ".PERIOD", "?QUESTIONMARK"]
    #     # print(1)
    #
    # print(metrics.classification_report(y_true=labels_all, y_pred=preds_all, target_names=labels_name))
    ################################################## 旧版计算方法，暂时废弃 ###################################################################

    # read ground_truth
    with open(args.test_set, "r", encoding="utf-8") as t:
        lines = [line.strip().split("\t")[0] for line in t.readlines()]
    # read vocab
    with open(args.label_vocab, "r", encoding="utf-8") as d:
        dicts = [line.strip() for line in d.readlines()]
    vocab = {int(e.split("\t")[1]): e.split("\t")[0] for e in dicts}

    words = []
    for line in lines:
        for word in line.split():
            words.append(word)

    ground_truth_stream = []
    predictions_stream = []
    for index, word in enumerate(tqdm(words, desc="[Building stream]")):

        # for debug
        # if index == len(preds_all):
        #     break

        ground_truth_stream.append(word)
        if labels_all[index] != 4:
            ground_truth_stream.append(vocab.get(labels_all[index]))

        predictions_stream.append(word)
        if preds_all[index] != 4:
            predictions_stream.append(vocab.get(preds_all[index]))

    # calculate error rate
    compute_error_stream(ground_truth_stream, predictions_stream)
