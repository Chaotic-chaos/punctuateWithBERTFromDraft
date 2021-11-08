# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     dataloader
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/20
   Software:      PyCharm
'''
import os

import librosa
import torch
import numpy as np
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from python_speech_features import mfcc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

'''
加载数据，封装batch
'''

class PunctuationDataset(Dataset):
    def __init__(self, input_path, label_vocab_path, audio_path_prefix, text_split_path_prefix):
        with open(input_path, "r", encoding="utf-8") as f:
            sentence_label_pair = [line.strip().split("\t") for line in f.readlines()]

        self.inputs = sentence_label_pair
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.tokenizer = BertTokenizer.from_pretrained("../BERTWithLinear/pretrained_bert")
        self.label_vocab = self._read_dict(label_vocab_path)
        self.audio_path_prefix = audio_path_prefix
        self.text_split_path_prefix = text_split_path_prefix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sentence = self.inputs[index][1]
        label = self.inputs[index][2]
        # audio_path = f"{os.path.join(self.audio_path_prefix, self.inputs[index][0].split('_')[0], self.inputs[index][0].split('_')[1], self.inputs[index][0])}.wav"
        audio_path = f"{os.path.join(self.audio_path_prefix, self.inputs[index][0])}.wav"
        align_text_path = f"{os.path.join(self.text_split_path_prefix, self.inputs[index][0])}.txt"

        # Convert to id
        label = self._w2i(self.label_vocab, label)
        sentence = self.tokenizer.convert_tokens_to_ids(sentence.split(" "))

        # compute the alignment between audio and text
        config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = audio_path
        task.text_file_path_absolute = align_text_path
        ExecuteTask(task).execute()
        sync_map = task.sync_map_leaves()
        # read the slices of the audio
        audio = [librosa.load(audio_path, sr=16000, offset=float(frag.begin), duration=float(frag.length))[0] for frag in sync_map[1:-1]]
        # print(111)

        return audio, sentence, label

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
    audios, sentences, labels = zip(*data)

    # 获取本batch中最长的序列
    sequence_lengths = [len(line) for line in sentences]
    max_length = max(sequence_lengths)

    # 提取所有的音频特征，送回原存储数组，节省储存空间
    res_audio = []
    for audio in audios:
        # audio = np.array([np.var(librosa.feature.melspectrogram(audio_seg, sr=16000, n_fft=1024, hop_length=512, n_mels=80, pad_mode="constant"), axis=1) for audio_seg in audio])

        # in case the audio length is 0 and there's no mfcc features can be extracted
        def _extract_mfcc(audio_seg):
            try:
                return mfcc(audio_seg, samplerate=16000, numcep=80, nfilt=80).transpose()
            except:
                return np.zeros([80, 1])

        audio = np.array([np.var(_extract_mfcc(audio_seg), axis=1) for audio_seg in audio])
        res_audio.append(audio)
    max_audio_length = max([audio.shape[0] for audio in res_audio])
    # padding audio
    res_audio = [np.pad(array=audio, pad_width=((0, max_audio_length-audio.shape[0]), (0, 0)), mode="constant") for audio in res_audio]
    res_audio = torch.from_numpy(np.array(res_audio))

    # 初始化两个结果矩阵，全置0，等待后续迭代替换非0元素，尺寸:batch_size x max_sequence_length
    res_sentences, res_labels = torch.zeros(len(sentences), max_length, dtype=torch.long), torch.zeros(len(labels), max_length, dtype=torch.long)
    # 使用原序列非零元素替换结果矩阵
    for index, sentence_label_pair in enumerate(zip(sentences, labels)):
        real_length = sequence_lengths[index]
        res_sentences[index, :real_length] = torch.LongTensor(sentence_label_pair[0])[:real_length]
        res_labels[index, :real_length] = torch.LongTensor(sentence_label_pair[1])[:real_length]

        assert res_sentences.size() == res_labels.size()

    assert res_sentences.size() == res_audio.size()[:-1]

    return res_audio, res_sentences, res_labels


if __name__ == '__main__':
    dataset = PunctuationDataset(input_path="../dataset/LibriTTS/processed_for_new/train-clean-100.tsv", label_vocab_path="../dataset/LibriTTS/processed_for_new/label.dict.tsv", audio_path_prefix=r"H:\Datasets\Opensource\LibriTTS\resample\train-clean-100", text_split_path_prefix="../dataset/LibriTTS/processed_for_new/split_text/")

    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

    for audio, sentence, label in tqdm(dataloader):
        # print(audio, sentence, label)
        pass
