# MSRA NER Dataset

语言：简体中文
编码：utf-8

**包含专名**：

    |标签| LOC | ORG | PER |
    |----|-----|-----|-----|
    |含义|地名 |组织名|人名|

**训练集**：

    |  句数  |  字符数  |  LOC数  |  ORG数  |  PER数  |
    |--------|----------|---------|---------|---------|
    |  45000 | 2171573  |  36860  |  20584  |  17615  |

**测试集**：

    |  句数  |  字符数  |  LOC数  |  ORG数  |  PER数  |
    |--------|----------|---------|---------|---------|
    |  3442  |  172601  |  2886   |  1331   |  1973   |


**标注格式**：

	[字符]	[标签]	# 分隔符为"\t"

	其中标签采用BIO规则，即非专名为"O",专名首部字符为"B-[专名标签]"，专名中部字符为"I-[专名标签]"

	例如：

		历	B-LOC
		博	I-LOC
		、	O
		古	B-ORG
		研	I-ORG
		所	I-ORG

## 数据来源

1. [SUDA-HLT/NewStudents](https://github.com/SUDA-HLT/NewStudents)

2. [paddlehub](https://github.com/PaddlePaddle/PaddleHub)

```
# _DATA_URL: https://paddlehub-dataset.bj.bcebos.com/msra_ner.tar.gz
import paddlehub as hub
dataset = hub.dataset.MSRA_NER()
```

