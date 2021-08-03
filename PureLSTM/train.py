# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     train
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/6/30
   Software:      PyCharm
'''
import argparse
import math
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoader import PunctuationDataset, collate_fn
from model import LSTMForPunctuator
from torch.utils.tensorboard import SummaryWriter
import numpy as np

'''
模型训练
    1. 初始化参数
    2. 初始化数据集，构建dataloader
    3. 构造模型
    4. 构造损失函数
    5. 构造优化器
    6. 迭代训练
        1）清空梯度
        2）计算预测
        3）计算loss
        4）反向传播
        5）参数更新
    7. 存数ckp
    8. 验证模型
'''

parser = argparse.ArgumentParser()

parser.add_argument("--train-set", default="../dataset/LibriTTS/processed_for_new/train-clean-100.tsv", help="train dataset path")
parser.add_argument("--valid-set", default="../dataset/LibriTTS/processed_for_new/dev-clean.tsv", help="valid dataset path")
parser.add_argument("--test-set", default=None, help="test dataset path")
parser.add_argument("--label-vocab", default="../dataset/LibriTTS/processed_for_new/label.dict.tsv", help="label vocabulary path")
parser.add_argument("--label-size", default=5, help="label dimension")
parser.add_argument("--lr", default=5e-5, type=float, help="learn rate")
parser.add_argument("--batch-size", default=128, help="batch size")
parser.add_argument("--epoch", default=35, help="train times")
parser.add_argument("--device", default="cuda", help="whether use gpu or not")
parser.add_argument("--ckp", default="./checkpoint", help="where to save the checkpoints")
parser.add_argument("--ckp-nums", default=15, help="how checkpoints to hold at the same time")
parser.add_argument("--tb", default="./tb", help="where the tensorboard saved")
parser.add_argument("--seed", default=1, help="random seed")

args = parser.parse_args()

if __name__ == '__main__':
    '''训练模型'''
    # choose device
    device = torch.device(args.device)

    # set fixed seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build dataloader
    print("Loading Data...")
    train_dataset = PunctuationDataset(input_path=args.train_set, label_vocab_path=args.label_vocab)
    valid_dataset = PunctuationDataset(input_path=args.valid_set, label_vocab_path=args.label_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # build model
    print("Building model...")
    model = LSTMForPunctuator(args.label_size, device)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)
    print(loss_func)
    print(optimizer)

    # move model to device
    model = model.to(device)

    # check if there's checkpoints
    if os.path.exists(args.ckp):
        # 存在checkpoint文件夹
        ckps = [os.path.join(args.ckp, file) for file in os.listdir(args.ckp)]
        if len(ckps) > 0:
            continue_train = input(f"Found {len(ckps)} checkpoints, [c]ontinue the training or [r]emove them all:\n")
            if continue_train == "c":
                checkpoint = torch.load(max(ckps, key=os.path.getctime), map_location=device)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                previous_epoch = checkpoint['epoch']
            elif continue_train == "r":
                # 删除所有已保存模型
                [os.remove(file) for file in ckps]
                previous_epoch = 0
    else:
        previous_epoch = 0

    # initiate tensorboard
    tb_writer_train = SummaryWriter(os.path.join(args.tb, "train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.tb, "valid"))

    batch = 0
    for epoch in range(1, args.epoch):
        # set train mode
        model.train()

        epoch += previous_epoch
        train_epoch_loss = 0
        for sentences, labels in tqdm(train_dataloader, desc=f"[Epoch {epoch}]"):
            # move data to device
            sentences = sentences.to(device)
            labels = labels.to(device)

            # zero gradient
            model.zero_grad()

            # forward
            try:
                preds = model(sentences)
            except Exception as e:
                print(f"Skipping this batch because {e}")
                continue

            # calculate loss
            loss = loss_func(preds.view(-1, 5), labels.view(-1))
            # write loss to tb
            tb_writer_train.add_scalar("loss", loss.item(), batch)

            # backward
            loss.backward()

            # update parameters
            optimizer.step()

            batch += 1
            train_epoch_loss += loss.item()

            # for debug
            # if batch == 5:
            #     break

        # print(f"Last batch loss: {loss.item()}")
        print(f"Starting validing...")
        # a = 0
        valid_epoch_loss = 0
        for sentences, labels in tqdm(valid_dataloader, desc="[Validing]"):
            sentences = sentences.to(device)
            labels = labels.to(device)

            model.eval()

            with torch.no_grad():
                try:
                    preds = model(sentences)
                except Exception as e:
                    print(f"Skipping this batch because {e}")
                    continue

                # calculate loss
                loss = loss_func(preds.view(-1, 5), labels.view(-1))

            valid_epoch_loss += loss.item()

            # a += 1

            # for debug
            # if a == 20:
            #     break

        # output valid and train result
        train_ppl = math.exp(train_epoch_loss / len(train_dataloader))
        valid_ppl = math.exp(valid_epoch_loss / len(valid_dataloader))
        tb_writer_train.add_scalar("PPL", train_ppl, epoch)
        tb_writer_valid.add_scalar("PPL", valid_ppl, epoch)
        print(f"Train PPL: {train_ppl: 7.3f}")
        print(f"Valid PPL: {valid_ppl: 7.3f}")

        # save checkpoint
        print("Saving checkpoint...")
        if not os.path.exists(args.ckp):
            os.mkdir(args.ckp)
        if len(os.listdir(args.ckp)) >= args.ckp_nums:
            files = [os.path.join(args.ckp, file) for file in os.listdir(args.ckp)]
            os.remove(min(files, key=os.path.getctime))
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, os.path.join(args.ckp, f"epoch{epoch}.pt"))
        print("Saved!")

