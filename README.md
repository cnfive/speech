# speech
中文语音识别训练，使用3层cnn+多层bilstm实现。使用lmdb存放音频特征MFCC，调用caffe 的函数在存放到lmdb里面做个矩阵转换。交叉熵损失函数和ctc_loss都做了尝试。
