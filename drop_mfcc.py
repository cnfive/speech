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
