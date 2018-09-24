# coding:UTF-8

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#import cifar10
import pickle
import os, sys
import tensorflow as tf
import numpy as np  
#import lmdb  
import librosa 
import sys  


# 训练过程的全局常量
MOVING_AVERAGE_DECAY = 0.9999  # 移动平均衰减
NUM_EPOCHS_PER_DECAY = 350.0  # 当学习速率开始下降的(期数)Epochs
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习速率衰减因子
INITIAL_LEARNING_RATE = 0.1  # 初始化学习速率

TOWER_NAME = 'tower'

#DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

init=tf.truncated_normal_initializer(stddev=0.01)

def average_gradients(tower_grads):

  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

category_num=0

def tower_loss(scope,images,labels):

  print(images)

  learning_rate=0.0001

  global_step = tf.Variable(-1, trainable=False, name='global_step')






  num_layer=4

  num_units=80

  y_label=tf.placeholder(dtype=tf.float32, shape=[52])

#shu chu fen lei shu mu
  #category_num=len(dic)


  x=tf.placeholder(dtype=tf.float32, shape=[1, 20, 671,1])

  x=images
  print('___________________________________________________-')
  print(images)





  conv2=tf.layers.conv2d(
      inputs=x,
      filters=52,
      kernel_size=10,
      padding="same",
      strides=2,
      activation=tf.nn.relu,
      kernel_initializer=init
      )
  conv3=tf.layers.conv2d(
      inputs=conv2,
      filters=52,
      kernel_size=5,
      padding="same",
      strides=2,
      activation=tf.nn.relu,
      kernel_initializer=init
      )
  conv4=tf.layers.conv2d(
      inputs=conv3,
      filters=52,
      kernel_size=3,
      padding="same",
      strides=1,
      activation=tf.nn.relu,
      kernel_initializer=init
      )





  yc=conv4





  yc=tf.squeeze(yc)
  print("yc.shape:",yc.shape)



 # Variables
  keep_prob = tf.placeholder(tf.float32, [])


  keep_prob=0.1

    
  
  cell_fw = [lstm_cell(num_units, keep_prob) for _ in range(num_layer)]
  cell_bw = [lstm_cell(num_units, keep_prob) for _ in range(num_layer)]
   
  inputs = tf.unstack(yc, 52, axis=2)
  print(len(inputs))
#for i in inputs:
    #print(i)

  output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
  print(len(output))
#for i in output:
  #  print(i)
  output = tf.stack(output, axis=2)
  print("+++++++++++++++++++++++++++")
  print('Output:', output)

  sentence_length=52
  output = tf.reshape(output, [sentence_length, -1])
  print('Output Reshape', output)
#for ot in output:
   # print(ot)



#y2=tf.placeholder(dtype=tf.float32, shape=[160,32])


# Output Layer
  with tf.variable_scope('outputs'):
     w = weight([800, category_num])
     b = bias([category_num])
     y2 = tf.matmul(output, w) + b
     print("w:",w)
     print("b:",b)
     print("y.shape:",y2.shape)

        
     y_predict = tf.cast(tf.argmax(y2, axis=1), tf.int32)
     print('Output Y shape:', y_predict.shape)
       


#tf.summary.histogram('y_predict', y_predict)
#yyy=y
  y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
  print('Y Label Reshape', y_label_reshape)
  print("y_predict.shape:",y_predict.shape)
  print("y_label.shape",y_label.shape)
    
#y_predict2=tf.reshape(y_predict,[1,-1]) 

  y_label_reshape2=tf.reshape(y_label_reshape,[1,-1])  
    # Prediction
  correct_prediction = tf.equal(y_predict, y_label_reshape)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.summary.scalar('accuracy', accuracy)
    
  print('Prediction', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,logits=tf.cast(y2, tf.float32)))
  tf.add_to_collection('losses', loss)  

  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')







  return total_loss

#它负责将不同GPU计算出的梯度进行合成
#合成的输入参数tower_grads是梯度的双层列表，
# 外层列表是不同GPU计算得到的梯度，内存列表是某个GPU内计算的不同Variable对应的梯度
#最内层元素为（grads,variable）,即tower_grads的基本原始为二元组（梯度，变量），
#具体形式为：[[(grad0_gpu0,var0_gpu0),(grad1_gpu0,var1_gpu0)],[(grad0_gpu1,var0_gpu1),(grad1_gpu1,var1_gpu1)],....]]
#
#然后用循环便利这个双层列表
def average_gradients(tower_grads):

  average_grads = []#首先创建平均梯度的列表average_grads,它负责将梯度在不同GPU间进行平均

  #zip(*tower_grads)将这个双层列表转置,变成
  #[[(grad0_gpu0,var0_gpu0),(grad0_gpu1,var0_gpu1)],[(grad1_gpu0,var1_gpu0),(grad1_gpu1,var1_gpu1)],....]]
  #然后使用循环遍历其元素
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      #每个循环获取的元素grad_and_vars,是同一个Variable的梯度在不同GPU计算的副本，需要计算其梯度的均值
      #如果这个梯度是N维的向量，需要在每个维度上都进行平均。
      # Add 0 dimension to the gradients to represent the tower.
      #先使用tf.expand_dims给这些梯度添加一个冗余的维度0,然后把这些梯度都放到列表grad中。
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      #然后把这些梯度都放到列表grad中。
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    #接着使用tf.concat将他们在维度0合并
    grad = tf.concat(grads, 0)
    #最后使用tf.reduce_mean针对维度0上求平均，即将其他维度全部平均。
    grad = tf.reduce_mean(grad, 0)

    #最后将平均后的梯度根Variable组合得到原有的二元组（梯度，变量）格式，并添加到列表average_grads
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    #当所有梯度都求完均值后，返回average_grads
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  dic=[]
#dic.append("999")
  dic.append("B")
  filepath="pinyin"
  file = open(filepath) 
  filename=[]
  file_and_id={}
  max_len=0
  for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    length=len(s_t)
    if length>max_len:
       max_len=length
  print("单句最长词个数，max_len:",max_len)
  max_len=50

  file.close()

  file = open(filepath) 
  filename=[]
  file_and_id={}

  for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    #print(s_t[0])
    filename.append(s_t[0])
    s_t.pop(0)


#print(s_t)
    for s in s_t:
        #print(s)
        if s not in dic:
       
            dic.append(s)


  file.close()

#add blank tag b



  file = open(filepath) 

  print("dic length",len(dic))
#print(dic)
  sentence_to_id=[]

  s_to_id=[]

#s_to_id.append(dic.index("B"))
  max_len=max_len+3

  for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    f_name=s_t[0]
    s_t.pop(0)
    sentence_to_id.append(dic.index("B"))
    for s in s_t:
    
          sentence_to_id.append(dic.index(s))
          #print(s)


   
    sentence_to_id.append(dic.index("B"))
    l=len(sentence_to_id)
    dis=0
    if l<max_len:
       dis=max_len-l
    for i in range(1,dis):
       sentence_to_id.append(dic.index("B"))
    #print(sentence_to_id)
    s_to_id.append(sentence_to_id)
    file_and_id[f_name]=sentence_to_id
    
    sentence_to_id=[]

  file.close()

  for key in file_and_id:
    print(key)



#s_to_id.append(dic.index("B"))
  maxlength=0
  minlength=100
  for id_l in s_to_id:
 
    #print(id_l)
    length=len(id_l)
    if length >maxlength:
        maxlength=length
    if minlength > length:
        minlength=length
  print("maxlength:",maxlength)
  print("minlength:",minlength)
  s_to_id=[]

  n=1

  num_batch=len(s_to_id)
  f_name=[]


  category_num=len(dic)
  rootdir='/media/hadoop/娱乐/a33'
  print(rootdir)

  n=1
  maxlen=671
  mfcc_dic={}
  for (dirpath, dirnames, filenames) in os.walk(rootdir):
			
			for filename in filenames:
				print(filename)
				if   filename.find('wav')>-1:
					filename_path = os.sep.join([dirpath, filename])
					print(filename_path)
					string_n=filename_path.split("/")
					key=string_n[5].replace(".wav","").replace(".WAV","")
					print("key:",key)
             	
					y, sr = librosa.load(filename_path, mono=True)
					mfcc=librosa.feature.mfcc(y=y, sr=sr)


					amin, amax = mfcc.min(), mfcc.max() # 

					mfcc = (mfcc-amin)/(amax-amin) # 
					print(mfcc.shape)
					m=mfcc.tolist()

					print("-----------------------------------")
					print(len(m))

					batch_size = 16


					print(mfcc.shape)
					print(len(mfcc))
					print(filename)
			
					n=n+1
					length_mfcc=mfcc.shape[1]
					if length_mfcc>maxlen:
						maxlen=length_mfcc
					print("maxlen:",maxlen)

					d=maxlen-mfcc.shape[1]
					print("d:",d)
					if d>0:
						s = (20,d)
						zero=np.zeros(s) 
           
						mfcc=tf.concat([mfcc,zero],1)

					m=tf.expand_dims(mfcc,0) 
					m=tf.expand_dims(m,3)
					#print(k)
					print(mfcc.shape) 
					print(m.shape)
					mfcc_dic[key.strip()]=m  
      
					
					

					print(n)







  sess=tf.Session()
  sess.run(tf.global_variables_initializer())
  with tf.Graph().as_default(), tf.device('/cpu:0'):
  
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

  
    lr = tf.train.exponential_decay(0.1,
                                    global_step,
                                    100,
                                    0.96,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)



    images2=[]
    labels2=[]
    
    temp_img=''
    for k in file_and_id:
  
         #print("key:",k)
         if k=="":
             continue
        
         mf=''
         for key in mfcc_dic:
             #print("k=","++"+k+"++")
             #print("key=","++"+key+"++")
             if k==key:
                mf=mfcc_dic[key]
                #print(mf)
                m=mf.eval(session=sess)
                yl=file_and_id[k]
                labels2.append(yl)
                images2.append(m)  
                print(m)

                #temp_img=
                #print(yl)


   
    tower_grads = []
    
    print("list length:"+str(len(images2)))

    #for img in  images2:    #stack images
    images2=tf.stack(images2,axis=1) 
   # images2=tf.squeeze(images2,[0])
    #print("images2.shape:",images2.shape)
    print(labels2)

    labels2=tf.convert_to_tensor(labels2)

    batch_size=1
    images2=tf.squeeze(images2,[0])
    print("image2 shape:",images2.shape)
    images2 = tf.cast(images2, tf.float32)

    input_queue = tf.train.slice_input_producer([images2, labels2], shuffle=False)
 
    image,label = tf.train.batch(input_queue,batch_size=batch_size,num_threads=1)
    print(tf.squeeze(image,[0]))
    print("image shape:",image.shape)
    print(label)
    num_gpus=1
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([image,label], capacity=2 * num_gpus)

    with tf.variable_scope(tf.get_variable_scope()):
         for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                           #image_batch=mf
                           image_batch, label_batch = batch_queue.dequeue()
                           #label_batch=yl
                           print(image_batch)
                           loss = tower_loss(scope, image_batch, label_batch)

                           tf.get_variable_scope().reuse_variables()

         
                           grads = opt.compute_gradients(loss)

                           tower_grads.append(grads)


    grads = average_gradients(tower_grads)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


    variable_averages = tf.train.ExponentialMovingAverage(0.98, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

  
    init = tf.global_variables_initializer()


    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
      )
    sess.run(init)

   

    for step in xrange(100000):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss],{})
      duration = time.time() - start_time






#def main(argv=None):  # pylint: disable=unused-argument
 



if __name__ == '__main__':


  train()
