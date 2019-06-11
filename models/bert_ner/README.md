# 结果

## 使用说明

1. 下载预训练的 BERT Model: [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). 解压到 bert_models 文件夹中. 
最终的目录结构如下:
```
├── bert
├── bert_models
│   └── chinese_L-12_H-768_A-12
├── ...
└── ...
```

2. 进入项目根目录下, 执行 `bash run_bert.sh`


## BERT + Softmax

- *dev_set*

```
INFO:tensorflow:processed 139524 tokens with 4817 phrases; found: 4876 phrases; correct: 4618.

INFO:tensorflow:accuracy:  99.41%; precision:  94.71%; recall:  95.87%; FB1:  95.29

INFO:tensorflow:              LOC: precision:  95.49%; recall:  96.59%; FB1:  96.04  2373

INFO:tensorflow:              ORG: precision:  89.58%; recall:  92.31%; FB1:  90.92  1286

INFO:tensorflow:              PER: precision:  98.60%; recall:  98.12%; FB1:  98.36  1217
```

- *test_set*

```
INFO:tensorflow:processed 162275 tokens with 5342 phrases; found: 5352 phrases; correct: 5080.

INFO:tensorflow:accuracy:  99.39%; precision:  94.92%; recall:  95.10%; FB1:  95.01

INFO:tensorflow:              LOC: precision:  96.53%; recall:  95.26%; FB1:  95.89  2708

INFO:tensorflow:              ORG: precision:  90.33%; recall:  93.29%; FB1:  91.79  1293

INFO:tensorflow:              PER: precision:  96.08%; recall:  96.43%; FB1:  96.26  1351
```
