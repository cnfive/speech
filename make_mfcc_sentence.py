# coding:UTF-8
from __future__ import print_function
import librosa
#import librosa.display

#import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np

 
#import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import os

import numpy as np
from scipy import signal
import pickle

import lmdb
import os, sys
#import caffe

#import caffe
#from caffe.proto import caffe_pb2
os.system("rm -r mfcc");
def initialize():
    env = lmdb.open("mfcc",map_size=100000000);
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


rootdir = '/home/yang/speech/train/train/A2'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
mfcc_list=[]
mfcc_list_d={}
f_name=[]

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
             string_n=path.split("/")
             key=string_n[5].replace(".wav","")
             print(key)
             	
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


             #librosa.display.waveplot(mfcc)
#plt.pcolormesh(mfcc)
#plt.ylabel('Frequency [Hz]')
             x = np.linspace(0, 10)

             #plt.xlabel('Time [sec]')
             #plt.show()


             #mfcc=np.transpose(mfcc, [1,0])
             #mfcc.append([0]*20)
            
            
             print(mfcc.shape[1])
             length_mfcc=mfcc.shape[1]
             if length_mfcc>maxlen:
                 maxlen=length_mfcc
             #mfcc_list.append(mfcc)
             if  length_mfcc<421:
                 mfcc_list_d[key]=mfcc
                 f_name.append(key)


with open('filename.pkl', 'wb') as f:
    pickle.dump(f_name, f)





maxlen=420
mfcc_list2=[]

mfcc_list_d_2={}


#maxlen=420

print("sequence max length:",maxlen)
#datum = caffe_pb2.Datum()
n=0
for k  in mfcc_list_d:
      print("------------begin-----------")
      print("shape[1]:",mfcc_list_d[k].shape[1])
      
      mfcc=mfcc_list_d[k]
      d=maxlen-mfcc_list_d[k].shape[1]
      print("d:",d)
      if d>0:
            s = (20,d)
            zero=np.zeros(s) 
            #print(zero)
           
            mfcc=tf.concat([mfcc_list_d[k],zero],1)

      m=tf.expand_dims(mfcc,0) 
      m=tf.expand_dims(m,3)
      print(k)
      print(mfcc_list_d[k].shape) 
      print(m.shape)  
      
      #mf=m.eval(session=sess)
 
     # datum = caffe.proto.caffe_pb2.Datum()
     # datum.channels = mf.shape[3]
     # datum.height = mf.shape[1]
     # datum.width = mf.shape[2]
    #  datum.data = mf.tobytes()   # or .tostring() if numpy < 1.9
     # datum.label = int(1)
     

        # The encode is only essential in Python 3
      # txn.put(str_id.encode('ascii'), datum.SerializeToString())

      insert(env, k, m);
      print("-------------",n)

      print("------------end-----------")
      n=n+1














print("Insert 3 records.")


#


env.close();



