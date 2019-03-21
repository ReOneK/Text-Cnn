使用keras进行搭建text-cnn文本分类模型

1 数据集为aclImdb，可在此下载http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

![IMDB数据集结构](https://github.com/ReOneK/Text-Cnn/blob/master/imdb.png)

2 read_file.py 进行文本读取，设置相应数据的标签（积极的文本设为1，消极的设为0）

3 data_precessing.py对文件进行预处理。
①使用Keras的Tokenizer模块将输入文本转换为数字特征
②使数字特征长度相同。原因是text-cnn中含有全连接层，因此输入必须是定长
③使用Embedding层将每个词编码转换为词向量

！[model introduce](https://github.com/ReOneK/Text-Cnn/blob/master/model.png)




