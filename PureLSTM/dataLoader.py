# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     dataLoader
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/6/26
   Software:      PyCharm
'''
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

'''
Dataset:(Dataset)
    1. __init__()
    2. __len__()
    3. __getitem__()

Dataloader:
    1. collate_fn()
'''

class PunctuationDataset(Dataset):
    def __init__(self, input_path, label_vocab_path):
        with open(input_path, "r", encoding="utf-8") as f:
            sentence_label_pair = [line.strip().split("\t") for line in f.readlines()]

        self.inputs = sentence_label_pair
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.tokenizer = BertTokenizer.from_pretrained("./pretrained_bert")
        self.label_vocab = self._read_dict(label_vocab_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sentence = self.inputs[index][1]
        label = self.inputs[index][2]

        # Convert to id
        label = self._w2i(self.label_vocab, label)
        sentence = self.tokenizer.convert_tokens_to_ids(sentence.split(" "))

        return sentence, label

    @staticmethod
    def _w2i(vocab, sequence):
        res = [int(vocab.get(w, vocab.get("<UNK>"))) for w in sequence.split(" ")]
        return res


    @staticmethod
    def _read_dict(dict_path):
        with open(dict_path, "r", encoding="utf-8") as dict:
            lines = [line.strip() for line in dict.readlines()]
        res = {e.split("\t")[0]: e.split("\t")[1] for e in lines}
        return res

def collate_fn(data):
    sentences, labels = zip(*data)

    # 获取本batch中最长的序列
    sequence_lengths = [len(line) for line in sentences]
    max_length = max(sequence_lengths)

    # 初始化两个结果矩阵，全置0，等待后续迭代替换非0元素，尺寸:batch_size x max_sequence_length
    res_sentences, res_labels = torch.zeros(len(sentences), max_length, dtype=torch.long), torch.zeros(len(labels), max_length, dtype=torch.long)

    # 使用原序列非零元素替换结果矩阵
    for index, sentence_label_pair in enumerate(zip(sentences, labels)):
        real_length = sequence_lengths[index]
        res_sentences[index, :real_length] = torch.LongTensor(sentence_label_pair[0])[:real_length]
        res_labels[index, :real_length] = torch.LongTensor(sentence_label_pair[1])[:real_length]

        assert res_sentences.size() == res_labels.size()

    return res_sentences, res_labels


if __name__ == '__main__':
    dataset = PunctuationDataset("../dataset/LibriTTS/processed_for_new/test-clean.tsv", "../dataset/LibriTTS/processed_for_new/label.dict.tsv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    for sentence, label in tqdm(dataloader):
        pass
        # print(f"sentence: {sentence}; label: {label}\n")
        # break
