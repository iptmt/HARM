## 运行步骤
1. 在当前目录创建文件夹：data, dump, log, output
2. 将train.csv, dev.csv, test.csv置入data文件夹中
3. 将glove的词向量(300d)置入data文件夹中（https://nlp.stanford.edu/projects/glove/）
4. 运行data_util.py，生成词典和词向量矩阵
5. 使用：python main.py train/test, 来训练模型和测试模型
6. 测试结果位于output文件夹中

## 注
1. 超参数一般定义在data_util.py, model.py, main.py的开始几行
2. 通过在dev数据集上测量MRR来判定模型效果
2. 数据的问题很多，不要对效果抱有太大的希望 = =