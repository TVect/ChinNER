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
