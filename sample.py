import tensorflow as tf

import matplotlib.pyplot as plt;
import numpy as np
from PIL import Image
import TensorflowUtils as utils

with tf.device('/cpu:0'):
     sess=tf.Session()    
#First let's load meta graph and restore weights
     saver = tf.train.import_meta_graph('Model19/model.ckpt-99.meta')
     saver.restore(sess,tf.train.latest_checkpoint('Model19/'))
# Access saved Variables directly
#print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
#graph=tf.import_graph_def()
#w1 = graph.get_tensor_by_name("w1:0")
#w2 = graph.get_tensor_by_name("w2:0")
#feed_dict ={w1:13.0,w2:17.0}
#Now, access the op that you want to run. 
#op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#print sess.run(op_to_restore,feed_dict)
#w=graph.get_tensor_by_name("conv_4")
#sess.run("conv_4:0")
#

summary_write = tf.summary.FileWriter("/home/yang/logdir" , graph)


#operations = graph.get_operations()
#print(operations[0].node_def)

#for op in graph.get_operations():
#	print(op.node_def)



for op in sess.graph.get_operations():
        print(op.name)
        #graphlist.write("\n")

feature = graph.get_operation_by_name('max_pooling2d_4/MaxPool').outputs[0] 
#feature = graph.get_operation_by_name('conv_4:0')  
#feature2=graph.get_tensor_by_name('c2:0')
#print(feature2)
#sess.run(feature)
#print(sess.run(feature2))

#= slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')

conv5=tf.layers.conv2d(
      inputs=feature,
      filters=128,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu,
      )
conv6=tf.layers.conv2d(
      inputs=feature,
      filters=256,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu,
      )
conv7=tf.layers.conv2d(
      inputs=feature,
      filters=10,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu,
      )
x=graph.get_tensor_by_name('x:0')
y_=graph.get_tensor_by_name('y_:0')

deconv_shape1 = feature.get_shape()
NUM_OF_CLASSESS=10

wt=utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")

#deconv1 = tf.nn.conv2d_transpose(conv7, wt, [1, 12, 12, 128], [1, 2, 2, 1], 'SAME')  
deconv1=tf.layers.conv2d_transpose (conv7, filters=10, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu )
deconv2=tf.layers.conv2d_transpose (deconv1, filters=10, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu )

deconv3=tf.layers.conv2d_transpose (deconv2, filters=10, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu )
deconv4=tf.layers.conv2d_transpose (deconv3, filters=10, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu )
#image_raw_data_jpg = tf.gfile.FastGFile('/home/yang/out_flower_photos/sunflowers/678714585_addc9aaaef.jpg', 'rb').read()  
#img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) #图像解码  
#img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8) #改变图像数据的类型
#img=np.asarray(img_data_jpg,np.float32)

image_value = tf.read_file('/home/yang/rpn/cifar-10-batches-py/test/2_4034.jpg')

#image_value2=tf.expand_dims(image_value, 0)
img = tf.image.decode_jpeg(image_value, channels=3)
image_value3=tf.image.resize_images(img, (32, 32), method=0)
print("image_value3:",image_value3.shape)
img2=tf.expand_dims(image_value3, 0)
#print(img2)

with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      data=sess.run(deconv4, feed_dict={x: img2.eval(),y_:[0]})
      annotation_pred = tf.argmax(data,dimension=3 , name="prediction")
      print(annotation_pred)

      data2=tf.squeeze(data)
      print("data.shape:",data2.shape)
      #imgd = tf.image.decode_jpeg(data2, channels=3)

      data3=data2.eval(session=sess)
      #print(data3)
      #data4=tf.image.encode_jpeg(data3) 
      #imgg=tf.image.convert_image_dtype([data3[:,:,0],data3[:,:,1],data3[:,:,2]], dtype=tf.uint8)
      annotation_pred2=tf.squeeze(annotation_pred)
      a=sess.run(annotation_pred2)
      print(a)
      for j in a:
          print(j)
      image = Image.fromarray(data3[:,:,1])
      plt.figure(1) #图像显示  
      plt.imshow(data3[:,:,1])  
      plt.show()

