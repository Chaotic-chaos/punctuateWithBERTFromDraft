# 使用BERT预训练模型实现标点恢复任务

---

## Requirements

- pytorch >= 1.9.0
- python >= 3.8
- transformers >= 4.7.0
- tqdm >= 4.61.1

### Install Requirements

`pip install -r BERTWithLinear/requirements.txt`



## Prepare Dataset

`python data_process/preprocess.py --src-path <origin dataset path> --out-path <processed dataset path>`



## Test a model

> You can find a pretrained model [here](https://drive.google.com/file/d/1wqZ3uKmCPdxSRIwUBsIdszWjNRh2zPZ1/view?usp=sharing)

- alter `BERTWithLinear/test.py`
- run it!  `python BERTWithLinear/test.py`



## Train a new model

### prepare dataset

- Option①: using the script

- Option②: 

  - processed your own dataset into the format below

    > it can be a very complicated thing the ocean\t_SPACE _SPACE _SPACE _SPACE _SPACE _SPACE ,COMMA _SPACE .PERIOD

  - build label dictionary

    > ,COMMA\t1
    > .PERIOD\t2
    > ?QUESTIONMARK\t3
    > _SPACE\t4

### set the hyperparameters

- alter `BERTWithLinear/train.py`
- run it! `python BERTWithLinear/train.py`



## Make real-time inference

- alter `BERTWithLinear/inference.py` to set the hyperparameters
- alter `BERTWithLinear/inference.py` to set the sentence
- run it! `python BERTWithLinear/inference.py`



