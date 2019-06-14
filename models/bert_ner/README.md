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


## 结果
### BERT + Softmax

- *dev_set*

```
INFO:tensorflow:processed 142957 tokens with 4911 phrases; found: 4950 phrases; correct: 4719.

INFO:tensorflow:accuracy:  99.44%; precision:  95.33%; recall:  96.09%; FB1:  95.71

INFO:tensorflow:              LOC: precision:  95.81%; recall:  96.56%; FB1:  96.19  2435

INFO:tensorflow:              ORG: precision:  90.66%; recall:  92.73%; FB1:  91.68  1295

INFO:tensorflow:              PER: precision:  99.34%; recall:  98.62%; FB1:  98.98  1220
```

- *test_set*

```
INFO:tensorflow:processed 172601 tokens with 6200 phrases; found: 6204 phrases; correct: 5878.

INFO:tensorflow:accuracy:  99.39%; precision:  94.75%; recall:  94.81%; FB1:  94.78

INFO:tensorflow:              LOC: precision:  96.02%; recall:  94.42%; FB1:  95.21  2838

INFO:tensorflow:              ORG: precision:  89.97%; recall:  93.01%; FB1:  91.47  1376

INFO:tensorflow:              PER: precision:  96.23%; recall:  96.57%; FB1:  96.40  1990
```

### BERT + crf

- *dev_set*

```
INFO:tensorflow:processed 142957 tokens with 4911 phrases; found: 4996 phrases; correct: 4665.

INFO:tensorflow:accuracy:  99.34%; precision:  93.37%; recall:  94.99%; FB1:  94.18

INFO:tensorflow:              LOC: precision:  94.57%; recall:  95.90%; FB1:  95.23  2450

INFO:tensorflow:              ORG: precision:  86.58%; recall:  90.68%; FB1:  88.58  1326

INFO:tensorflow:              PER: precision:  98.36%; recall:  97.64%; FB1:  98.00  1220
```

- *test_set*

```
INFO:tensorflow:processed 172601 tokens with 6200 phrases; found: 6277 phrases; correct: 5876.

INFO:tensorflow:accuracy:  99.29%; precision:  93.61%; recall:  94.77%; FB1:  94.19

INFO:tensorflow:              LOC: precision:  95.77%; recall:  95.01%; FB1:  95.39  2863

INFO:tensorflow:              ORG: precision:  85.60%; recall:  91.59%; FB1:  88.49  1424

INFO:tensorflow:              PER: precision:  96.23%; recall:  96.57%; FB1:  96.40  1990
```
