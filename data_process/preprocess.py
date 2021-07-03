# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     preprocess
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/6/21
   Software:      PyCharm
'''
import argparse
import os
import re
import time

from tqdm import tqdm

'''
对原始数据集进行前处理
    1. 特殊语法剔除
        1) 's -> is
        2) 're -> are
        3) 剔除 --
        4) 剔除连字符 -
    2. 标点映射
        1) ' -> ,
        2) " -> ,
        3) ! -> .
        4) : -> ,
        5) ; -> ,
    3. 将所有字符转为小写字符
    4. 抽取平行句对
        ([squence, split with space], [label, split with space])
'''
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src-path", type=str, default="../dataset/raw/", help="where the origin dataset stored")
parser.add_argument("-o", "--out-path", type=str, default="../dataset/processed_for_bert", help="where the processed dataset stored")
parser.add_argument("-d", "--build-dict", type=bool, default=True, help="whether build dictionary automatically or not")
parser.add_argument("-m", "--mode", type=str, default="new", help="process the data into which format")

args = parser.parse_args()

if args.mode == "new":
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

    input_dict = {}
    label_dict = {}

    # 构建标签字典
    if args.build_dict:
        for k, v in enumerate(label_map.values()):
            label_dict[v] = k
        label_dict['_SPACE'] = len(label_dict)

    files = os.listdir(args.src_path)

    for file in files:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Processing file {file}")
        # read file
        with open(os.path.join(args.src_path, file), "r", encoding="utf-8") as f:
            lines = f.readlines()

        res = []
        # clean data
        for line in tqdm(lines, desc=f"[Processing]"):
            line = line.strip()

            res_line = []
            res_label = []
            for word in line.split(" "):
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

                        # 构造字典
                        if args.build_dict and "train" in file:
                            # 仅对训练集构造字典
                            input_dict[word] = input_dict.get(word, len(input_dict))

                        res_line.append(word)
                        res_label.append(label)
                else:
                    continue

            if len(res_line) != len(res_label):
                print(f"Sentence({len(res_line)}): {' '.join(res_line)}")
                print(f"Label({len(res_label)}): {' '.join(res_label)}")

            res.append((" ".join(res_line), " ".join(res_label)))
            # break

        # break

        # 写回到新文件
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        with open(f"{os.path.join(args.out_path, file.split('.')[0])}.tsv", "w+", encoding="utf-8") as o:
            for line in tqdm(res, desc="[Writing]"):
                o.write(f"{line[0]}\t{line[1]}\n")

    # print(input_dict)
    # 写回字典
    # 添加<UNK> <PAD>
    input_dict['<UNK>'] = len(input_dict)
    input_dict['<PAD>'] = len(input_dict)
    with open(os.path.join(args.out_path, "label.dict.tsv"), "w", encoding="utf-8") as f:
        for k, v in label_dict.items():
            f.write(f"{k}\t{v}")
            f.write("\n")
    with open(os.path.join(args.out_path, "input.dict.tsv"), "w", encoding="utf-8") as f:
        for k, v in input_dict.items():
            f.write(f"{k}\t{v}")
            f.write("\n")

    print("All Done!")

elif args.mode == "baseline":
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
    # 将数据集处理成baseline模型需要的格式
    files = os.listdir(args.src_path)

    for file in files:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Processing file {file}")
        with open(os.path.join(args.src_path, file), "r", encoding="utf-8") as f:
            lines = f.readlines()

        res = []
        for line in tqdm(lines, desc="[Processing]"):
            line = line.strip()

            res_line = []

            for word in line.split():
                # 清洗句子
                word = re.sub(pattern=".*'s", string=word, repl=f"{word[:-2]} is")
                word = re.sub(pattern=".*--.*", string=word, repl="")
                word = re.sub(pattern=".*'re", string=word, repl=f"{word[:-3]} are")
                word = re.sub(pattern=".*'t", string=word, repl=f"{word[:-2]} not")
                word = re.sub(pattern=".*\.\.\..*", string=word, repl=f"")

                # 标点符号映射
                word = "".join([punc_map.get(c, c) for c in word])

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

            res.append(res_line)
            # break

        # 写回文件
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        with open(os.path.join(args.out_path, f"IWSLT12.{file.split('.')[0]}.txt"), "w+", encoding="utf-8") as o:
            for line in tqdm(res, desc="[Writing]"):
                o.write(" ".join(line))
                o.write("\n")

    print("All Done!")