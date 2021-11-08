# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     model
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/8/11
   Software:      PyCharm
'''
import fairseq.checkpoint_utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel
import torch.nn.functional as F

from BERTandWav2vec.dataLoader import PunctuationDataset, collate_fn

'''主模型文件
    - 使用BERT作为文本特征提取器
    - 使用wav2vec 1.0作为音频特征提取器
    - 对音频特征在squence_length的维度上进行求平均
    - 文本音频特征进行concanate操作
    - 过线性分类层，输出结果
'''

class PuncWithBERTandWav2vec(nn.Module):
    def __init__(self, label_size, device, wav2vec_path):
        super(PuncWithBERTandWav2vec, self).__init__()

        # use online bert
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.bert = BertModel.from_pretrained("../BERTWithLinear/pretrained_bert")

        # initialize wav2vec
        self.wav2vec, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        self.wav2vec = self.wav2vec[0]
        self.wav2vec.eval()

        self.linear = nn.Linear(1280, label_size)
        self.device = device

    def forward(self, audios, sentences):
        # calculate text feature
        attention_mask = torch.sign(sentences)
        attention_mask = attention_mask.to(self.device)
        input = {
            "input_ids": sentences,
            "attention_mask": attention_mask
        }
        x = self.bert(**input).last_hidden_state

        # calculate audio features
        audios = [torch.var(self.wav2vec.feature_aggregator(self.wav2vec.feature_extractor(a)).transpose(1, 2), dim=1) for a in audios]
        max_audio_length = max(audios, key=lambda x: x.size()[0]).size()[0]
        y = torch.zeros([len(audios), max_audio_length, 512], device=self.device, dtype=torch.float32)
        for index, audio in enumerate(audios):
            y[index, :audio.size()[0]] = audio

        # concatenate the features
        x = torch.cat([x, y], dim=2)

        # make predictions
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    model = PuncWithBERTandWav2vec(5, torch.device("cpu"), "./wav2vec_large.pt")
    dataset = PunctuationDataset(input_path="../dataset/LibriTTS/processed_for_new/train-clean-100.tsv",
                                 label_vocab_path="../dataset/LibriTTS/processed_for_new/label.dict.tsv",
                                 audio_path_prefix=r"H:\Datasets\Opensource\LibriTTS\resample\train-clean-100",
                                 text_split_path_prefix="../dataset/LibriTTS/processed_for_new/split_text/")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for audios, sentences, labels in dataloader:
        model(audios=audios, sentences=sentences)
        print(111)

