# TextClassification
Chinese text classification

# Data
[THUCNews](http://thuctc.thunlp.org/), total 740k titles, 14 classes, (train, valid, test) = (80%, 10%, 10%).

pretrained word vector: [Sogou News Word+Character 300d](https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw)

# Evaluation
|Method|Test Acc(%)|
|--|--|
|BagOfWord|80.1273|
|TextCNN|92.0473|
|TextRNN|92.0473|
|TextRNN with Attention|92.1095|
|FastText|93.0688|
|Transformer|88.1804|
|Bert|95.0590|

