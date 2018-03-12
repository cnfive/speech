
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
    name = txn.get(str(sid).encode('ascii'));
    return name;

def display(env):
    txn = env.begin();
    cur = txn.cursor();
    n=1
    for key, value in cur:
        print (key);
        print(n)
        n=n+1

env = initialize();


#display(env)
datum=caffe_pb2.Datum() 

#print( "Get the name of student whose sid = 3.")
name = search(env, "B8_250");
#name=name.decode("UTF-8","ignore").encode("UTF-8")
datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(name)

flat_x = np.fromstring(datum.data)
x = flat_x.reshape(1, datum.height, datum.width,datum.channels)
print(x)

print(x.shape)
env.close();

