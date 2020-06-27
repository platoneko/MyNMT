# 说明

该框架系于2019年7月在学校实验室从事基于深度学习的对话系统研究时设计，因为出于实验性目的，很多地方还有待完善，模型配置相关描述不是很完整（也不想继续完善），最终版本的模型只有RNN seq2seq和transformer、优化器也只有Adam和SGD、不支持预训练词向量。并不推荐当做完整框架直接使用，更适合在该框架上自行扩充或取用所需的模块。主要的模块在 [my-pytorch-modules](https://github.com/platoneko/my-pytorch-modules) 有备份。

如要使用，要求输入的train/valid/test文件都是json（jsonl）格式。json文件格式如下（使用的test数据文件也要求包含tgt field）：

```
{'src': '...', 'tgt': ...}
{'src': '...', 'tgt': ...}
...
```

tokenizer只是将输入字符串序列简单split，如有必要请自行进行分词并将token以' '分隔作为输入。

训练（请根据需要自行修改shell文件中相关配置参数）:

```
./run_train.sh
```

测试（请根据需要自行修改shell文件中相关配置参数）:

```
./run_test.sh
```

