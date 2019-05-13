# cnn_chinese
cnn对手写中文的识别
参考https://github.com/DeepCompute/cnn

数据集为CASIA在线和离线中文手写数据库
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

我下载的是GNT格式
关于GNT格式的具体定义请参考数据库官网

在进行GNT转PNG图片时，不能进行转换，不知道哪方面原因（“我看他们pytho n可以

然后在gnttodata.java中将数据转为28×28，接一个数字标签，一个中文标签
转换结果在开头有％的出现，所以利用datatotraindata.java再进行处理

数据集的分类样本太多，所以在训练数据方面只是将50个文字进行训练
