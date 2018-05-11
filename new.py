# coding:UTF-8

#from tqdm import trange
from time import sleep
import time
#from tqdm import *
import pickle
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
max_len=98

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
#增加一个无用字符
dic.append("!")
print("dic length",len(dic))
print(dic)
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
          sentence_to_id.append(dic.index("B"))
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
#for id_l in s_to_id:
#    length=len(id_l)
#     if length<32:
#        id_l.append(dic.index("B"))


s_to_id=[]


#print(file_and_id)


#import lmdb
import os, sys
import tensorflow as tf
import numpy as np  
#import lmdb  
import librosa 
import sys  
#import caffe  
#from caffe.proto import caffe_pb2  




learning_rate=0.01

global_step = tf.Variable(-1, trainable=False, name='global_step')






num_layer=1

num_units=80

y_label=tf.placeholder(dtype=tf.float32, shape=[100])

#shu chu fen lei shu mu
category_num=len(dic)


x=tf.placeholder(dtype=tf.float32, shape=[1, 20, 671,1])






conv2=tf.layers.conv2d(
      inputs=x,
      filters=105,
      kernel_size=10,
      padding="same",
      strides=2,
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )
conv3=tf.layers.conv2d(
      inputs=conv2,
      filters=105,
      kernel_size=5,
      padding="same",
      strides=2,
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )
conv4=tf.layers.conv2d(
      inputs=conv3,
      filters=105,
      kernel_size=3,
      padding="same",
      strides=1,
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )





yc=conv4





yc=tf.squeeze(yc)
print("yc.shape:",yc.shape)



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
   
inputs = tf.unstack(yc, 105, axis=2)
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

sentence_length=100
output = tf.reshape(output, [sentence_length, -1])
print('Output Reshape', output)
#for ot in output:
   # print(ot)



#y2=tf.placeholder(dtype=tf.float32, shape=[160,32])


# Output Layer
with tf.variable_scope('outputs'):
     w = weight([840, category_num])
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
#correct_prediction = tf.equal(y_predict, y_label_reshape)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.summary.scalar('accuracy', accuracy)
    
#print('Prediction', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,logits=tf.cast(y2, tf.float32)))







logits=tf.cast(y2, tf.float32)

logits=tf.convert_to_tensor(logits)
logits=tf.reshape(logits,[1,100,category_num])
logits2=tf.transpose(logits,(1, 0, 2))

indices = tf.where(tf.not_equal(tf.cast(y_label_reshape2,  tf.int32), 0))


target = tf.SparseTensor(indices=indices, values=tf.gather_nd(y_label_reshape2, indices), dense_shape=tf.cast(tf.shape(y_label_reshape2), tf.int64))


sequence_len=np.ones(1)*100
loss = tf.nn.ctc_loss(target,logits2,   sequence_len )


#tf.summary.scalar('loss',loss)

cost = tf.reduce_mean(loss) 


train = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(cost, global_step=global_step)

#ctc_beam_search_decoder方法,是每次找最大的K个概率分布 #还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder 
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits2, sequence_len, merge_repeated=False) 
acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), target))

tf.summary.scalar('acc',acc)
    
# Train
#train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    
    # Saver
saver = tf.train.Saver(max_to_keep=2)


  
predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) 




sess=tf.Session()
sess.run(tf.global_variables_initializer())
ummary_waiter = tf.summary.FileWriter("log",tf.get_default_graph())

gstep = 0



def task():
    x = 0
    for _ in range(10 ** 6):
        x = 0
    return x



n=1

num_batch=len(s_to_id)
f_name=[]



rootdir='/media/hadoop/娱乐/A2'
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







for c in range(100000):
    for k in file_and_id:
  
         #print("key:",k)
         if k=="":
             continue
         #datum=caffe_pb2.Datum() 
  
  
         #name = search(env, k);
         mf=''
         for key in mfcc_dic:
             #print("k=","++"+k+"++")
             #print("key=","++"+key+"++")
             if k==key:
                mf=mfcc_dic[key]
            

       
         #if  datum.width>649:
         #     continue

                yl=file_and_id[k] 
                #print(yl) 
 
         #yl2=tf.reshape(y_predict,[1,-1]) 
                m=mf.eval(session=sess)
                _,acc2,loss2=sess.run([train,acc,loss] ,feed_dict={x: m,y_label:yl})

    
                print("it's training at:",n)
  
         #acc2 = sess.run(acc, feed_dict={x: mf,y_label:yl})
            # Calculate batch loss
                print("distance:",acc2)
         #loss2 = sess.run(loss, feed_dict={x: mf,y_label:yl})
                print("loss:",loss2)
      
         
  
                if acc2<0.1 :
                     saver.save(sess, "Model9/model.ckpt",global_step=i)
                     y_out = sess.run(predict, feed_dict={x: mf,y_label:yl})
                     print(y_out)
                     print(yl)
             #break;
         n=n+1

  



sys.exit(0)
