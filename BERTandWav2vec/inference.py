# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     inference
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/9/27
   Software:      PyCharm
'''
import argparse

import librosa
import numpy as np
import torch
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from transformers import BertTokenizer

from BERTandWav2vec.model import PuncWithBERTandWav2vec

'''
单句推理模式
    1. 构造单句输入
    2. 构造模型
    3. 加载ckp
    4. 更改模型模式
    5. 零梯度推理
    6. argmax
    7. 读取字典
    8. 输出结果
'''

parser = argparse.ArgumentParser()

parser.add_argument("--ckp", default="./checkpoint/epoch1.pt", help="where the model file saved")
parser.add_argument("--label-vocab", default="../dataset/LibriTTS/processed_for_new/label.dict.tsv", help="the label dictionary")
parser.add_argument("--bert-path", default="../BERTWithLinear/pretrained_bert")
parser.add_argument("--wav2vec-path", default="./wav2vec_large.pt")
parser.add_argument("--device", default="cpu", help="whether use cpu or gpu")

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device(args.device)

    audio_path = "./test_data/237_126133_000002_000000.wav"
    align_text_path = "./test_data/split_text/temp.txt"
    sentence = "yes it must be confessed"

    # Construct sentence input
    sentence.lower()
    sentence = sentence.split(" ")
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    ids = tokenizer.convert_tokens_to_ids(sentence)
    input_sentence = torch.tensor(ids, device=device, dtype=torch.long)

    # Construct audio input
    # compute the alignment between audio and text
    config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = audio_path
    task.text_file_path_absolute = align_text_path
    ExecuteTask(task).execute()
    sync_map = task.sync_map_leaves()
    # read the slices of the audio
    audio = [librosa.load(audio_path, sr=16000, offset=float(frag.begin), duration=float(frag.length))[0] for frag in sync_map[1:-1]]
    # padding audio sequence
    '''这里的pad是针对audios中的每个音频分别进行pad，后续向wav2vec送的是每一个音频的不同切片组成一个batch'''
    max_audio_length = max(audio, key=lambda x: x.shape[0]).shape[0]
    input_audio = torch.tensor([np.pad(a, pad_width=((0, max_audio_length - a.shape[0])), mode="constant") for a in audio])

    # Construct model
    model = PuncWithBERTandWav2vec(label_size=5, device=device, wav2vec_path=args.wav2vec_path)

    # load ckp
    checkpoint = torch.load(args.ckp, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    # Set mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        pred = model(sentences=input_sentence.unsqueeze(0), audios=input_audio.unsqueeze(0))
    pred_ids = torch.argmax(model(sentences=input_sentence.unsqueeze(0), audios=input_audio.unsqueeze(0)).squeeze(0), 1).numpy().tolist()

    # load vocab
    with open(args.label_vocab, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    label_vocab = {int(e.split("\t")[1]): e.split("\t")[0][0] for e in lines}

    pred_label = [label_vocab.get(id) for id in pred_ids]

    res = []
    for word, label in zip(sentence, pred_label):
        res.append(f"{word}{label if label != '_' else ''}")

    print(" ".join(res))
