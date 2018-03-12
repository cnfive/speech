# coding:UTF-8
from __future__ import print_function
import librosa
import librosa.display

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

 
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os

import numpy as np
from scipy import signal



learning_rate=0.01

global_step = tf.Variable(-1, trainable=False, name='global_step')
rootdir = '/media/hadoop/娱乐/A2'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
mfcc_list=[]
maxlen=0
for i in range(0,len(list)):
     path = os.path.join(rootdir,list[i])
     if os.path.isfile(path):
          #print(path)
          #if path[-3:-1]=="wav":
              #print(path[-4:0])
         # print(path.find("wav"))
          if path.find("wav")>0:
             print(path)
             	
             y, sr = librosa.load(path, mono=True)
             mfcc=librosa.feature.mfcc(y=y, sr=sr)
#print(mfcc)

             amin, amax = mfcc.min(), mfcc.max() # 

             mfcc = (mfcc-amin)/(amax-amin) # 
             print(mfcc.shape)
             m=mfcc.tolist()
#print(m[1])
             print("-----------------------------------")
             print(len(m))

             batch_size = 16

#mfcc=mfcc.T
             print(mfcc.shape)
             print(len(mfcc))
         

#import wavio


             librosa.display.waveplot(mfcc)
#plt.pcolormesh(mfcc)
#plt.ylabel('Frequency [Hz]')
             x = np.linspace(0, 10)

             plt.xlabel('Time [sec]')
             #plt.show()


             #mfcc=np.transpose(mfcc, [1,0])
             #mfcc.append([0]*20)
            
            
             print(mfcc.shape[1])
             length_mfcc=mfcc.shape[1]
             if length_mfcc>maxlen:
                 maxlen=length_mfcc
             mfcc_list.append(mfcc)

print(maxlen)

s = (20,1)
zero=np.zeros(s) 
print(zero)
mfcc_list2=[]
for mfcc in mfcc_list:
      d=maxlen-mfcc.shape[1]
      if maxlen-mfcc.shape[1]>0:
             for j in range(1,d):
                    mfcc=tf.concat([mfcc,zero],1)

      m=tf.expand_dims(mfcc,0) 
      m=tf.expand_dims(m,3)    
      mfcc_list2.append(m)
for m in mfcc_list2:
      
      
      print(m.shape)




dic=[]
filepath="/media/hadoop/娱乐/A2/train.txt"
file = open(filepath) 
for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    
    s_t.pop(0)


#print(s_t)
    for s in s_t:
        #print(s)
        if s not in dic:
       
            dic.append(s)


file.close()

#add blank tag b
dic.append("B")
w=" "
for word in dic:
#     w=w+word
     print(dic.index(word))
     print(word)



file = open(filepath) 

print("dic length",len(dic))
#print(dic)
sentence_to_id=[]

s_to_id=[]

#s_to_id.append(dic.index("B"))

for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    
    s_t.pop(0)
    sentence_to_id.append(dic.index("B"))
    for s in s_t:
    
          sentence_to_id.append(dic.index(s))
          #print(s)


   
    sentence_to_id.append(dic.index("B"))
    l=len(sentence_to_id)
    dis=0
    if l<32:
       dis=32-l
    for i in range(1,dis):
       sentence_to_id.append(dic.index("B"))
    #print(sentence_to_id)
    s_to_id.append(sentence_to_id)
    sentence_to_id=[]

file.close()

#s_to_id.append(dic.index("B"))
maxlength=0
minlength=100
for id_l in s_to_id:
 
    print(id_l)
    length=len(id_l)
    if length >maxlength:
        maxlength=length
    if minlength > length:
        minlength=length
print(maxlength)
print(minlength)
for id_l in s_to_id:
     length=len(id_l)
     if length<32:
        id_l.append(dic.index("B"))
for id_l in s_to_id:
     print(id_l)




num_layer=2

num_units=80

y_label=tf.placeholder(dtype=tf.float32, shape=[32])

#shu chu fen lei shu mu
category_num=len(dic)


x=tf.placeholder(dtype=tf.float32, shape=[1, 20, 621,1])


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())



conv2=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=10,
      padding="same",
      strides=2,
     
      )
conv3=tf.layers.conv2d(
      inputs=conv2,
      filters=32,
      kernel_size=5,
      padding="same",
      strides=2,
   
      )






yc=conv3





yc=tf.squeeze(yc)
print(yc.shape)



 # Variables
keep_prob = tf.placeholder(tf.float32, [])



def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)






keep_prob=0.1

    
  
cell_fw = [lstm_cell(num_units, keep_prob) for _ in range(num_layer)]
cell_bw = [lstm_cell(num_units, keep_prob) for _ in range(num_layer)]
   
inputs = tf.unstack(yc, 32, axis=2)
print(len(inputs))
for i in inputs:
    print(i)

output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
print(len(output))
for i in output:
    print(i)
output = tf.stack(output, axis=2)
print("+++++++++++++++++++++++++++")
print('Output:', output)

sentence_length=32
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
       


tf.summary.histogram('y_predict', y_predict)
#yyy=y
y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
print('Y Label Reshape', y_label_reshape)
print("y_predict.shape:",y_predict.shape)
print("y_label.shape",y_label.shape)
    

    
    # Prediction
correct_prediction = tf.equal(y_predict, y_label_reshape)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
    
print('Prediction', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,logits=tf.cast(y2, tf.float32)))

tf.summary.scalar('loss', cross_entropy)
    
# Train
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    
    # Saver
saver = tf.train.Saver()


  
    

#全连接层
#sess=tf.InteractiveSession()  
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#

# Global step
gstep = 0

summaries_dir="/media/hadoop/娱乐/summaries/"
    # Summaries
#summaries = tf.summary.merge_all()
#writer = tf.summary.FileWriter(summaries_dir,sess.graph)

f1 = open('mfcc_and_sentence.txt', 'w')
f1.writelines(s_to_id)
f1.writelines(mfcc_list2)
f1.close()





num_batch=len(s_to_id)
for n in range(num_batch):
  mf=mfcc_list2[n].eval(session=sess)
  print(mf.shape)
  yl=s_to_id[n]     
  for i in range(1000):
    
    result=sess.run(train ,feed_dict={x: mf,y_label:yl})

    #sess.run(yyy) 
    print(n)
    #rt=sess.run(y_predict)
    #print(rt)
    print(result)
    #writer.add_summary(summaries, n)

    # Calculate batch accuracy
    acc = sess.run(accuracy, feed_dict={x: mf,y_label:yl})
            # Calculate batch loss
    print("accuracy:",acc)
    loss = sess.run(cross_entropy, feed_dict={x: mf,y_label:yl})
    print("loss:",loss)
    y_out = sess.run(y_predict, feed_dict={x: mf,y_label:yl})
    print(y_out)
    print(yl)
    if acc>0.999:
       break;
   
   
#writer.close()
sess.close()





