# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     pickle_data
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/8/14
   Software:      PyCharm
'''
import argparse
import os
import pickle
import sys

import psutil
from torch.utils.data import DataLoader
from tqdm import tqdm

from BERTandWav2vec.dataLoader import PunctuationDataset, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="../dataset/LibriTTS/processed_for_new/train-clean-100.tsv")
parser.add_argument("--label-vocab", default="../dataset/LibriTTS/processed_for_new/label.dict.tsv")
parser.add_argument("--audio-path-prefix", default=r"H:\Datasets\Opensource\LibriTTS\resample\train-clean-100")
parser.add_argument("--text-split-path-prefix", default="../dataset/LibriTTS/processed_for_new/split_text/")
parser.add_argument("--output-path", default="../dataset/LibriTTS/processed_for_new/pickled")
parser.add_argument("--batch-size", default=1)

args = parser.parse_args()


if __name__ == '__main__':
    dataset = PunctuationDataset(input_path=args.data_path, label_vocab_path=args.label_vocab, audio_path_prefix=args.audio_path_prefix, text_split_path_prefix=args.text_split_path_prefix)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)

    all_audios, all_sentences, all_labels = [], [], []
    # a = 0
    with tqdm(dataloader) as progress_bar:
        for audios, sentences, labels in progress_bar:
            all_audios.append(audios)
            all_sentences.append(sentences)
            all_labels.append(labels)

            # set message
            progress_bar.set_description(
                f"已占用: {(sys.getsizeof(all_audios) + sys.getsizeof(all_sentences) + sys.getsizeof(all_labels)) /1024 /1024:.3f}MB | " \
                f"已使用占比: {psutil.virtual_memory().percent}% | " \
                f"剩余: {psutil.virtual_memory().available /1024 /1024:.3f}MB"
            )

            # for debug
            # a += 1
            # if a == 5:
            #     break

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path, os.path.split(args.data_path)[-1][:-4]), "wb") as o:
        pickle.dump({
            "audios": all_audios,
            "sentences": all_sentences,
            "labels": all_labels
        }, o)

    print("All Done!")
