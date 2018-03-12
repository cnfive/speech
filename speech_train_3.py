# coding:UTF-8

from tqdm import trange
from time import sleep
import time
from tqdm import *
import pickle
dic=[]
filepath="/media/hadoop/娱乐/A2/train.txt"
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
dic.append("B")


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
    f_name=s_t[0]
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
print(maxlength)
print(minlength)
for id_l in s_to_id:
     length=len(id_l)
     if length<32:
        id_l.append(dic.index("B"))


s_to_id=[]


#print(file_and_id)

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import lmdb
import os, sys
import tensorflow as tf
import numpy as np  
import lmdb  
  
import sys  
import caffe  
from caffe.proto import caffe_pb2  
def initialize():
    env = lmdb.open("mfcc");
    return env;

def insert(env, sid, name):
    txn = env.begin(write = True);
    txn.put(str(sid), name);
    txn.commit();

def delete(env, sid):
    txn = env.begin(write = True);
    txn.delete(str(sid));
    txn.commit();

def update(env, sid, name):
    txn = env.begin(write = True);
    txn.put(str(sid), name);
    txn.commit();

def search(env, sid):
    txn = env.begin();
    name = txn.get(str(sid));
    return name;

def display(env):
    txn = env.begin();
    cur = txn.cursor();
    for key, value in cur:
        print (key, value);

env = initialize();




learning_rate=0.01

global_step = tf.Variable(-1, trainable=False, name='global_step')






num_layer=2

num_units=80

y_label=tf.placeholder(dtype=tf.float32, shape=[32])

#shu chu fen lei shu mu
category_num=len(dic)


x=tf.placeholder(dtype=tf.float32, shape=[1, 20, 420,1])






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
#for i in inputs:
    #print(i)

output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
print(len(output))
#for i in output:
  #  print(i)
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
saver = tf.train.Saver(max_to_keep=2)


  

#全连接层
#sess=tf.InteractiveSession()  
sess=tf.Session()
sess.run(tf.global_variables_initializer())
#

# Global step
gstep = 0

summaries_dir="/media/hadoop/娱乐/summaries/"


def task():
    x = 0
    for _ in range(10 ** 6):
        x = 0
    return x



n=1

num_batch=len(s_to_id)
f_name=[]
with open("filename.pkl", 'rb') as f:
     f_name = pickle.load(f)
print(f_name)


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



sess=tf.Session() 
sess.run(init)
threads = tf.train.start_queue_runners(sess=sess)


for k in f_name:
  #mf=mfcc_list2[n].eval(session=sess)
  datum=caffe_pb2.Datum() 

  
  name = search(env, k);

  datum = caffe.proto.caffe_pb2.Datum()
  datum.ParseFromString(name)

  flat_x = np.fromstring(datum.data)
  mf = flat_x.reshape(1, datum.height, datum.width,datum.channels)
 
  #print(mf.shape)
  if k in file_and_id:
      yl=file_and_id[k]  
  else:
      continue
   
  for i in range(1000):
  #for i in tqdm(rapicklenge(1000)):
    result=sess.run(train ,feed_dict={x: mf,y_label:yl})

    #sess.run(yyy) 
    print("it's training at:",n)
    #rt=sess.run(y_predict)
    #print(rt)
    #print(result)
    #writer.add_summary(summaries, n)

    # Calculate batch accuracy
    acc = sess.run(accuracy, feed_dict={x: mf,y_label:yl})
            # Calculate batch loss
    print("accuracy:",acc)
    loss = sess.run(cross_entropy, feed_dict={x: mf,y_label:yl})
    print("loss:",loss)
    y_out = sess.run(y_predict, feed_dict={x: mf,y_label:yl})
    
    
    #time.sleep(0.01)
  
    if acc>0.999 and loss< 0.01 :
       saver.save(sess, "Model/model.ckpt",global_step=i)
       print(y_out)
       print(yl)
       break;
   

  n=n+1


env.close();

sys.exit(0)

