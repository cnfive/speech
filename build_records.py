# coding:UTF-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import tensorflow as tf 
from PIL import Image

import lmdb

import caffe  
from caffe.proto import caffe_pb2 

import pickle 
import numpy as np


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

writer = tf.python_io.TFRecordWriter("mfcc.tfrecords")
env = initialize();

txn = env.begin();
cur = txn.cursor();

with open("filename.pkl", 'rb') as f:
     f_name = pickle.load(f)
print(f_name)


for k in f_name:
  #mf=mfcc_list2[n].eval(session=sess)
  datum=caffe_pb2.Datum() 

  
  name = search(env, k);

  datum = caffe.proto.caffe_pb2.Datum()
  datum.ParseFromString(name)

  flat_x = np.fromstring(datum.data)
  mf = flat_x.reshape(1, datum.height, datum.width,datum.channels)
  mf=mf.tostring()
  print("datum.channels:",datum.channels)
  #print(mf.shape)
  if k in file_and_id:
      yl=file_and_id[k]  
  else:
      continue
  print("yl.shape:",len(yl))
  #yl=np.array(yl).tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=yl)),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mf]))
        }))
  #print(example)
  writer.write(example.SerializeToString())  #序列化为字符串



writer.close()

