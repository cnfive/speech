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
import pickle


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

word=["AA","BB","CC","DD","EE","FF","JJ","KK","LL","MM","NN","OO","PP","QQ","RR","SS","TT"]

rootdir=""

word_list=[]
filename="/home/yang/speech/tag.pkl"
if os.path.exists(filename):
	with open("tag.pkl", 'rb') as f:
	 
		name = pickle.load(f)
		word_list=name
		#print(word_list)
for t in word:
	if t not in word_list:
		rootdir='/home/yang/speech/train/train/'+t
		print(rootdir)
		word_list.append(t)
		with open('tag.pkl', 'wb') as f:
			pickle.dump(word_list, f)
		break;	


print(word_list)		
			#print(dirp)
