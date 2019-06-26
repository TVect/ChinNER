# ChinNER

Chinese NER Models.

整理实现常见的中文命名实体识别模型.

## 目录结构
```
├── bin    # 常见运行脚本
├── dataset
│   └── msra    # msra ner 数据集
├── models
│   ├── bert_ner    # bert ner 模型
│   └── lstm_crf    # lstm crf ner 模型
└── utils
    ├── conlleval.py
    ├── evaluate.py    # 评估脚本
    ├── sentence_cutter.py    # 句子切分工具
    └── tagging_utils.py
```

## [数据集](https://github.com/TVect/ChinNER/blob/master/dataset/README.md)
- msra ner

## 评估
- conlleval

  来源: https://github.com/spyysalo/conlleval.py

## 模型
- [lstm+crf](https://github.com/TVect/ChinNER/tree/master/models/lstm_crf)

- [bert](https://github.com/TVect/ChinNER/tree/master/models/bert_ner)
