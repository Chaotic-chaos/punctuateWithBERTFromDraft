# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     preprocess_for_libritts
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/14
   Software:      PyCharm
'''
import argparse
import os
import re

import icecream as ic
from tqdm import tqdm

'''
对LibriTTS数据集进行预处理
    1. 模式①：baseline 模式②：bert/bert+wav2vec
    2. 读取所有*trans*.txt文件
    3. 以\t为分隔符，取0, 2（即音频名称以及正则化后的文本）
    4. 剔除特殊句法
        1) 's -> is
        2) 're -> are
        3) 剔除 --
        4) 剔除连字符 -
    5. 标点映射
        1) ' -> ,
        2) " -> ,
        3) ! -> .
        4) : -> ,
        5) ; -> ,
    6. 将所有字符转换为小写
    7. 抽取平行句对/转换标点标签
        ([squence, split with space], [label, split with space])
'''

parser = argparse.ArgumentParser()

parser.add_argument("--src-path", default=r"H:\Datasets\Opensource\LibriTTS", help="Path of the datasets")
parser.add_argument("--out-path", default=r"../dataset/LibriTTS/processed_for_baseline", help="Path of the outputs")
parser.add_argument("--mode", default=1, help="Process the data into which mode")

args = parser.parse_args()

if args.mode == 2:
    punc_map = {
        "'": ",",
        "\"": ",",
        "!": ".",
        ";": ",",
        ":": ",",
        "-": "",
    }

    label_map = {
        ",": ",COMMA",
        ".": ".PERIOD",
        "?": "?QUESTIONMARK",
    }

    # list all directories
    dirs = os.listdir(args.src_path)

    # scan all splits
    for dir in dirs:
        # filter all files in this split, find all the trans file
        trans_files = list(filter(lambda name: "trans" in name, sum([file for _, _, file in os.walk(os.path.join(args.src_path, dir))], [])))

        # scan all trans file
        res = []
        for file in trans_files:
            with open(os.path.join(args.src_path, dir, dir, file.split(".")[0].split("_")[0], file.split(".")[0].split("_")[1], file), "r", encoding="utf-8") as f:
                lines = [e.strip() for e in f.readlines()]

            for line in tqdm(lines, desc=f"[Processing {file}]"):
                line = line.split("\t")

                # split audio name & normalized text
                audio_name = line[0]
                transcript = line[-1]

                # wash the transcripts
                res_line = []
                res_label = []
                for word in transcript.split(" "):
                    # 清洗句子
                    word = re.sub(pattern=".*'s", string=word, repl=f"{word[:-2]} is")
                    word = re.sub(pattern=".*--.*", string=word, repl="")
                    word = re.sub(pattern=".*'re", string=word, repl=f"{word[:-3]} are")
                    word = re.sub(pattern=".*'ve", string=word, repl=f"{word[:-3]} have")
                    word = re.sub(pattern=".*'t", string=word, repl=f"{word[:-2]} not")
                    word = re.sub(pattern=".*\.\.\..*", string=word, repl=f"")

                    # 标点符号映射
                    word = "".join([punc_map.get(c, c) for c in word])

                    # 将所有字符转为小写
                    word = word.lower()

                    # 清洗完成若该词为空，不回写
                    if len(word) > 0:
                        words = word.split(" ")
                        for word in words:
                            if len(word) == 0:
                                continue
                            # 抽取标签
                            try:
                                label = label_map.get(word[-1], "_SPACE")
                            except IndexError:
                                continue
                            if label != "_SPACE":
                                word = word[:-1]

                            # 去除词中其他乱七八糟的标点
                            word = "".join(['' if c in label_map else c for c in word])

                            res_line.append(word)
                            res_label.append(label)
                    else:
                        continue

                try:
                    assert len(res_line) == len(res_label), "The length of label & transcript doesn't match"
                except Exception as e:
                    print(f"Skipping this line because {e}")
                    continue
                res.append(f"{audio_name}\t{' '.join(res_line)}\t{' '.join(res_label)}\n")

        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        with open(f"{os.path.join(args.out_path, dir)}.tsv", "w", encoding="utf-8") as o:
            o.writelines(res)

        print(f"{dir}: {len(res)} done.")

    print("All Done!")

elif args.mode == 1:
    # baseline模式，将语料处理成baseline使用的格式，不保存音频文件路径
    punc_map = {
        "'": ",",
        "\"": ",",
        "!": ".",
        ";": ",",
        ":": ",",
        "-": "",
    }

    label_map = {
        ",": ",COMMA",
        ".": ".PERIOD",
        "?": "?QUESTIONMARK",
    }

    # list all directories
    dirs = os.listdir(args.src_path)

    # scan all splits
    for dir in dirs:
        # filter all files in this split, find all the trans file
        trans_files = list(filter(lambda name: "trans" in name, sum([file for _, _, file in os.walk(os.path.join(args.src_path, dir))], [])))

        # scan all trans file
        res = []
        for file in trans_files:
            with open(
                    os.path.join(args.src_path, dir, dir, file.split(".")[0].split("_")[0], file.split(".")[0].split("_")[1],
                                 file), "r", encoding="utf-8") as f:
                lines = [e.strip() for e in f.readlines()]

            for line in tqdm(lines, desc=f"[Processing {file}]"):
                line = line.split("\t")

                # split audio name & normalized text
                transcript = line[-1]

                # wash the transcripts
                res_line = []
                for word in transcript.split(" "):
                    # 清洗句子
                    word = re.sub(pattern=".*'s", string=word, repl=f"{word[:-2]} is")
                    word = re.sub(pattern=".*--.*", string=word, repl="")
                    word = re.sub(pattern=".*'re", string=word, repl=f"{word[:-3]} are")
                    word = re.sub(pattern=".*'ve", string=word, repl=f"{word[:-3]} have")
                    word = re.sub(pattern=".*'t", string=word, repl=f"{word[:-2]} not")
                    word = re.sub(pattern=".*\.\.\..*", string=word, repl=f"")

                    # 标点符号映射
                    word = "".join([punc_map.get(c, c) for c in word])

                    # 清洗重复标点或句首标点
                    try:
                        if word[0] in label_map:
                            word = word[1:]
                        if word[-1] in label_map and word[-2] in label_map:
                            word = word[:-1]
                    except Exception as e:
                        print(f"Skipping word: {word} because {e}")
                        continue

                    # 将所有字符转为小写
                    word = word.lower()

                    if len(word) > 0:
                        label = label_map.get(word[-1], "_SPACE")
                        if label != "_SPACE":
                            word = f"{word[:-1]} {label}"
                            res_line.append(word)
                        else:
                            res_line.append(word)
                    else:
                        continue

                res.append(" ".join(res_line)+"\n")

        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        with open(f"{os.path.join(args.out_path, dir)}.tsv", "w", encoding="utf-8") as o:
            o.writelines(res)

        print(f"{dir}: {len(res)} done.")

    print("All Done!")
