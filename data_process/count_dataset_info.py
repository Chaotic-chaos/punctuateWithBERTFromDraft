# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     count_dataset_info
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/10/15
   Software:      PyCharm
'''
import argparse

'''为论文中的实验部分提供数据集中的一些数据信息'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", default="../dataset/LibriTTS/processed_for_new/dev-clean.tsv")

    args = parser.parse_args()

    total_puncs = 0

    with open(args.dataset, "r", encoding="utf-8") as d:
        line = d.readline()
        while line:
            # print(line)
            # line = d.readline()
            tags = line.split("\t")[2]
            for tag in tags.split(" "):
                if tag != "_SPACE":
                    total_puncs += 1
            line = d.readline()

    print(f"Current dataset has {total_puncs} punctuations")