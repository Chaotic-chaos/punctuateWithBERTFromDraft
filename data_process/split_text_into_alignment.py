# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     split_text_into_alignment
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/30
   Software:      PyCharm
'''
import argparse
import os

from tqdm import tqdm

'''
Split the dataset's text into a word per line for the following alignment
'''

parser = argparse.ArgumentParser()

parser.add_argument("--src", default="../dataset/LibriTTS/processed_for_new/test-clean.tsv", help="Source dataset")
parser.add_argument("--tgt", default="../dataset/LibriTTS/processed_for_new/split_text/", help="Target path of the split files")

args = parser.parse_args()

if __name__ == '__main__':
    # read file
    with open(args.src, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="[Splitting]"):
            file_name = line.split("\t")[0]
            split_text = [f"{word}\n" for word in line.split("\t")[1].split(" ")]

            if not os.path.exists(args.tgt):
                os.mkdir(args.tgt)

            with open(os.path.join(args.tgt, f"{file_name}.txt"), "w", encoding="utf-8") as t:
                t.writelines(split_text)
            # print(111)
    print("All Done!")
