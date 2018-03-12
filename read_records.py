# coding:UTF-8

import tensorflow as tf
import numpy as np





#for serialized_example in tf.python_io.tf_record_iterator("mfcc.tfrecords"):
#    example = tf.train.Example()
#    example.ParseFromString(serialized_example)

#    image = example.features.feature['img_raw'].bytes_list.value
#    label = example.features.feature['label'].bytes_list.value
    # 可以做一些预处理之类的
#    print image, label


def read_and_decode():
    #根据文件名生成一个队列
    filename="mfcc.tfrecords"
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([31],  tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'],  tf.float64)
    img = tf.reshape(img, [20, 420, 1])
  
    label =features['label']
    #label = tf.cast(label,tf.int32)
    print(label)
   
  
    return img, label
img,label=read_and_decode()


img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=1, capacity=10,
                                                min_after_dequeue=3)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(2):
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12) 
        print(val)
        print( l[0])
   
 
