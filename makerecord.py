# -*- coding = utf-8 -*-

from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import  walk
from os.path import join


DATA_DIR = 'flower_photos/'


IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3
NUM_TRAIN = 7000
NUM_VALIDARION = 1144

def read_images(path):
    filenames = next(walk(path))[2]
    num_files = len(filenames)
    images = np.zeros((num_files,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = np.zeros((num_files, ), dtype=np.uint8)
    f = open('label.txt')
    lines = f.readlines()
  
    for i,filename in enumerate(filenames):
        img = imread(join(path,filename))
        img = imresize(img,(IMG_HEIGHT,IMG_WIDTH))
        images[i] = img
        labels[i] = int(lines[i])
    f.close()
    return images,labels


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(images,labels,name):
    
    num = images.shape[0]
   
    filename = name+'.tfrecords'
    print('Writting',filename)
  
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
     
        img_raw = images[i].tostring()
      
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[i])),
            'image_raw': _bytes_feature(img_raw)}))
      
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')

def main(argv):
    print('reading images begin')
    start_time = time.time()
    train_images,train_labels = read_images(DATA_DIR)
    duration = time.time() - start_time
    print("reading images end , cost %d sec" %duration)

    #get validation
    validation_images = train_images[:NUM_VALIDARION,:,:,:]
    validation_labels = train_labels[:NUM_VALIDARION]
    train_images = train_images[NUM_VALIDARION:,:,:,:]
    train_labels = train_labels[NUM_VALIDARION:]

    #convert to tfrecords
    print('convert to tfrecords begin')
    start_time = time.time()
    convert(train_images,train_labels,'train')
    convert(validation_images,validation_labels,'validation')
    duration = time.time() - start_time
    print('convert to tfrecords end , cost %d sec' %duration)

if __name__ == '__main__':
    tf.app.run()
