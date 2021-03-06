# -*- coding: utf-8 -*-
'''
   Project:       punctuateWithBERTFromDraft
   File Name:     model
   Author:        Chaos
   Email:         life0531@foxmail.com
   Date:          2021/7/20
   Software:      PyCharm
'''
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

'''
模型结构文件（BERT+Spectrogram音频特征）
    1) 文本(LongTensor) [bsz x length] -> BERT -> 文本特征 [bsz x length x 768]
    2) 音频(FloatTensor) [bsz x text_length x audio_length] -> librosa -> Spectrogram特征 [bsz x text_length x 80]
    3) 音频/文本特征进行 concatenate
    4) Linear(848, 5)
'''
class BRETWithAudio(nn.Module):
    def __init__(self, label_size, device):
        super(BRETWithAudio, self).__init__()

        # use online bert
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # use offline pretrained bert
        self.bert = BertModel.from_pretrained("../BERTWithLinear/pretrained_bert")

        self.linear = nn.Linear(848, label_size)
        self.device = device

    def forward(self, sentences, audios):
        # calculate text feature
        attention_mask = torch.sign(sentences)
        attention_mask = attention_mask.to(self.device)
        input = {
            "input_ids": sentences,
            "attention_mask": attention_mask
        }
        x = self.bert(**input).last_hidden_state

        # concat audios & texts
        x = torch.cat([x, audios], dim=2)

        # make predictions
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    model = BRETWithAudio(5, torch.device("cpu"))

    sentences = torch.tensor([[2023, 2003, 1037,  100, 3405]], dtype=torch.long)
    audios = torch.tensor([[[1.2948e-02, 2.0919e-03, 2.1391e-03, 2.9100e-03, 1.0826e-01,           7.7174e-01, 5.6296e-01, 1.1064e-01, 3.0940e-02, 9.2927e-02,           1.0021e+00, 4.1272e+00, 4.2015e+00, 9.6641e-01, 3.4157e-01,           3.3738e-01, 4.3109e-01, 2.2096e-01, 1.8201e-01, 2.0671e-02,           4.7948e-03, 3.6576e-03, 7.8271e-04, 9.0952e-04, 4.2715e-03,           2.8927e-03, 1.5106e-03, 1.8378e-03, 1.2409e-03, 7.6382e-04,           9.9819e-04, 1.0247e-03, 1.0808e-03, 2.3076e-03, 4.1538e-03,           7.2475e-03, 9.0676e-03, 7.4147e-03, 8.6665e-03, 1.2533e-02,           4.6634e-03, 5.1644e-03, 2.0778e-02, 1.6550e-02, 8.8889e-03,           1.5364e-02, 7.9426e-03, 2.3138e-02, 2.4667e-02, 2.3110e-03,           2.2659e-03, 6.8640e-03, 1.1764e-02, 8.4118e-03, 1.6651e-02,           8.0035e-03, 3.8480e-03, 1.4063e-03, 7.7778e-04, 1.5012e-03,           1.2275e-03, 1.1015e-03, 1.4930e-03, 1.5773e-03, 4.6446e-03,           1.3660e-03, 1.9084e-03, 5.8357e-03, 3.0278e-03, 2.9152e-03,           4.2500e-03, 7.0078e-03, 5.9907e-03, 8.6637e-03, 1.3423e-02,           1.3529e-02, 1.3213e-02, 6.8120e-03, 7.7308e-03, 5.1304e-04],          [1.0998e-02, 1.8709e-03, 2.4626e-03, 1.4622e-01, 5.3423e-01,           3.2231e-01, 1.2260e-01, 1.2940e-01, 1.3219e+00, 1.5518e+00,           7.3815e-01, 5.7946e-01, 5.2178e-01, 1.7157e+00, 6.6746e+00,           1.1741e+01, 4.0700e+00, 1.4190e+00, 2.6468e-01, 3.2777e-02,           3.0664e-02, 1.7294e-02, 5.3676e-03, 6.8599e-03, 6.2454e-03,           5.8368e-03, 4.8242e-03, 4.2972e-03, 4.2497e-03, 1.8366e-03,           3.1868e-03, 5.2422e-03, 3.9730e-03, 9.5524e-03, 2.9925e-02,           1.1830e-02, 1.2515e-02, 2.4262e-02, 2.7706e-02, 9.4961e-03,           1.3070e-02, 3.1479e-02, 1.7357e-02, 1.9662e-02, 4.8671e-03,           3.2533e-03, 1.3318e-02, 5.1398e-02, 4.2132e-02, 2.0648e-02,           5.5841e-03, 1.2597e-02, 1.4203e-02, 8.6737e-03, 2.0611e-02,           2.3072e-02, 1.2829e-02, 4.3241e-03, 1.9758e-03, 5.2685e-03,           4.4659e-03, 2.3140e-03, 1.2359e-03, 7.4143e-04, 1.7895e-03,           2.5252e-03, 8.2669e-03, 2.0838e-02, 4.2377e-03, 1.8913e-03,           1.3445e-03, 4.1001e-03, 5.1962e-03, 7.4944e-03, 7.9519e-03,           1.3257e-02, 1.3934e-02, 4.2878e-03, 6.6146e-03, 9.8450e-04],          [1.3892e-02, 4.9221e-03, 8.2786e-03, 1.0307e-01, 3.2578e-01,           3.0091e-01, 8.9705e-03, 5.9165e-02, 2.1620e-01, 2.3541e-01,           2.4163e+00, 3.0698e+00, 2.7775e-01, 7.6395e-01, 6.7848e-01,           1.7796e+00, 3.6562e+00, 1.8577e+00, 1.5517e+00, 2.1350e-01,           3.8966e-01, 2.9467e-01, 1.6607e-01, 7.2093e-02, 4.4865e-01,           2.5592e-01, 7.6333e-02, 6.8812e-02, 1.1421e-01, 6.6269e-02,           4.5545e-02, 2.4129e-01, 2.5937e-01, 1.0420e-01, 1.3096e-01,           2.0554e-01, 2.7946e-01, 2.6720e-01, 5.0122e-02, 5.4154e-02,           5.1158e-02, 7.6868e-03, 1.1164e-02, 2.2366e-02, 4.6327e-03,           4.6293e-03, 7.3291e-03, 1.2616e-02, 1.0852e-02, 2.3640e-03,           3.9992e-03, 5.6436e-03, 2.2267e-03, 6.5333e-04, 8.7822e-04,           9.8919e-04, 8.2366e-04, 3.0545e-04, 4.0057e-04, 6.7953e-04,           5.4847e-04, 1.0332e-03, 2.5547e-03, 1.1513e-03, 1.6129e-03,           3.7380e-04, 4.4658e-04, 1.2786e-03, 9.0991e-04, 6.6815e-04,           9.9514e-04, 2.8340e-04, 3.0848e-04, 3.3691e-04, 1.9750e-04,           3.9317e-04, 7.3268e-04, 4.6781e-04, 1.6632e-04, 7.8530e-06],          [1.3607e-02, 2.9463e-03, 5.8042e-03, 1.8066e-02, 6.7822e-02,           2.2067e-01, 1.5043e-01, 3.9999e-02, 4.2075e-02, 1.6008e-01,           9.4570e-02, 2.4418e-01, 3.5856e-01, 1.0168e-01, 4.4636e-01,           4.1338e-01, 1.7234e-01, 2.7184e-01, 3.2621e-01, 5.8716e-02,           1.0538e-02, 2.9852e-03, 3.5505e-03, 1.2386e-02, 1.4571e-01,           1.2562e-01, 5.2542e-03, 3.9642e-03, 8.4113e-03, 1.7327e-02,           8.5734e-02, 8.8026e-02, 1.7925e-02, 1.3415e-02, 5.2640e-02,           2.4748e-01, 4.8713e-02, 4.6117e-02, 9.4381e-02, 3.9793e-02,           1.0051e-02, 2.4457e-02, 1.8671e-02, 1.1929e-02, 1.4018e-03,           2.0630e-03, 8.5383e-03, 5.4735e-03, 4.5959e-03, 2.0198e-03,           3.1904e-03, 2.6591e-03, 1.4999e-03, 1.9936e-03, 4.0079e-03,           4.7889e-03, 3.1100e-03, 2.8167e-03, 3.9242e-03, 4.7174e-03,           2.8889e-03, 6.1471e-03, 6.4355e-03, 4.2545e-03, 3.8391e-03,           2.1449e-03, 1.7283e-03, 9.7255e-03, 9.0994e-03, 2.1031e-03,           3.1780e-03, 1.8662e-03, 1.3815e-03, 6.1921e-04, 7.8057e-04,           1.0625e-03, 1.9003e-03, 1.5489e-03, 6.1543e-04, 5.8680e-05],          [9.7676e-03, 3.6228e-03, 1.0314e-03, 6.4408e-04, 3.0500e-04,           2.2081e-04, 5.9809e-04, 5.4025e-04, 2.0617e-04, 3.4368e-04,           7.9913e-04, 5.1067e-04, 2.4488e-04, 2.6934e-04, 2.9041e-04,           2.2238e-04, 4.0834e-04, 2.2858e-04, 5.6836e-05, 5.2757e-05,           4.9769e-05, 7.8033e-06, 1.0069e-05, 3.1037e-05, 4.8264e-05,           1.2928e-04, 9.8935e-05, 7.3173e-05, 3.0127e-05, 5.7115e-05,           1.1523e-05, 2.4577e-05, 6.4627e-05, 7.8946e-05, 7.8926e-05,           1.3283e-04, 1.0463e-04, 1.4961e-04, 1.0056e-04, 3.2452e-05,           3.2823e-05, 4.1315e-05, 4.6353e-05, 4.7182e-05, 3.9291e-05,           7.5852e-05, 2.1259e-04, 4.8517e-04, 5.3796e-04, 1.3019e-04,           4.2529e-05, 8.2437e-05, 8.2083e-05, 4.9340e-05, 6.5914e-05,           2.2609e-05, 1.1198e-05, 8.5456e-06, 1.0670e-05, 2.3742e-05,           3.3870e-05, 4.3764e-05, 3.3656e-05, 1.0039e-05, 1.8416e-05,           1.7350e-05, 3.0975e-05, 4.3547e-05, 1.7303e-05, 1.3104e-05,           2.1875e-05, 1.2284e-05, 6.4142e-06, 8.0816e-06, 1.0168e-05,           1.2169e-05, 7.8047e-06, 8.9064e-06, 1.9437e-06, 1.0471e-06]]])
    labels = torch.tensor([[4, 4, 4, 4, 2]])

    preds = model(sentences=sentences, audios=audios)

    loss_func = nn.CrossEntropyLoss()

    loss = loss_func(preds.view(-1, 5), labels.view(-1))

    print(loss)
