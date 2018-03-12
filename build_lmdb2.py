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

#os.system("rm -r mfcc");
sess=tf.Session()
#sess.run(tf.global_variables_initializer())
def initialize():
    env = lmdb.open("mfcc",map_size=1000000000);
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
	#sleep(10)
	for filename in filenames:
		if filename.endswith('.wav') or filename.endswith('.WAV'):
			filename_path = os.sep.join([dirpath, filename])
			print(filename_path)
			string_n=filename_path.split("/")
			key=string_n[7].replace(".wav","")
			print(key)
			n=n+1

			print(n)
print("------------end-----------")
env.close()
			
