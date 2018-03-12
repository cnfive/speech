# coding:UTF-8
import os
import lmdb
import librosa
import gc
from time import sleep, time
import numpy as np
import tensorflow as tf
import caffe
from caffe.proto import caffe_pb2
gc.set_threshold(700, 10, 5)
os.system("rm -r mfcc");
sess=tf.Session()
#sess.run(tf.global_variables_initializer())
def initialize():
    env = lmdb.open("mfcc",map_size=10000000000);
    return env;

def insert(env, sid, name):
    txn = env.begin(write = True);
    txn.put(str(sid).encode('ascii'), name);
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

rootdir = '/home/yang/speech/train/train/'
wav_files = []
n=1
maxlen=649
for (dirpath, dirnames, filenames) in os.walk(rootdir):
	gc.collect() 
	sleep(30)
	for filename in filenames:
		if filename.endswith('.wav') or filename.endswith('.WAV'):
			filename_path = os.sep.join([dirpath, filename])
			print(filename_path)
			string_n=filename_path.split("/")
			key=string_n[7].replace(".wav","")
			print(key)
             	
			y, sr = librosa.load(filename_path, mono=True)
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
			print(filename)
			
			n=n+1
			length_mfcc=mfcc.shape[1]
			if length_mfcc>maxlen:
				maxlen=length_mfcc
			print("maxlen:",maxlen)

#mfcc=mfcc_list_d[k]
			d=maxlen-mfcc.shape[1]
			print("d:",d)
			if d>0:
				s = (20,d)
				zero=np.zeros(s) 
            #print(zero)
				mfcc=tf.concat([mfcc,zero],1)

			m=tf.expand_dims(mfcc,0) 
			m=tf.expand_dims(m,3)
			#print(k)
			print(mfcc.shape) 
			print(m.shape)  
      
			mf=m.eval(session=sess)

			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = mf.shape[3]
			datum.height = mf.shape[1]
			datum.width = mf.shape[2]
			datum.data = mf.tostring()   # or .tostring() if numpy < 1.9
			datum.label = int(1)
     

        # The encode is only essential in Python 3
      # txn.put(str_id.encode('ascii'), datum.SerializeToString())

			#insert(env, key, m);
			insert(env, key,datum.SerializeToString()  );
			del(datum)
			#del(mfcc_d)
			del(mfcc)
			del(mf)
			del(m)
			del(y)
			del(sr)
			del(amin)
			del(amax)

			print(n)
print("------------end-----------")
env.close()
			
