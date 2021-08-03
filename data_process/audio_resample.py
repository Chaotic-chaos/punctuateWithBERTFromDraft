# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     audio_resample
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/30
   Software:      PyCharm
'''
import argparse
import os
import subprocess

from tqdm import tqdm

'''
Resample the given audio files into 16kHz
'''

parser = argparse.ArgumentParser()

parser.add_argument("--src", default="../dataset/LibriTTS/processed_for_new/test-clean.tsv")
parser.add_argument("--audio-prefix", default=r"H:\Datasets\Opensource\LibriTTS\test-clean\test-clean")
parser.add_argument("--tgt-prefix", default=r"H:\Datasets\Opensource\LibriTTS\resample\test-clean")

args = parser.parse_args()

if __name__ == '__main__':
    with open(args.src, "r", encoding="utf-8") as s:
        for line in s:
            line = line.split("\t")
            audio = f"{os.path.join(args.audio_prefix, line[0].split('_')[0], line[0].split('_')[1], line[0])}.wav"
            audio_name = f"{line[0]}.wav"
            print("#"*20)
            print(f"Resampling {audio_name}")

            if not os.path.exists(args.tgt_prefix):
                os.mkdir(args.tgt_prefix)

            subprocess.call(f"sox {audio} -c 1 -b 16 -r 16k {os.path.join(args.tgt_prefix, audio_name)}")

            # for debug
            # break
