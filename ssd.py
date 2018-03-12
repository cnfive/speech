import tensorflow as tf

conv1=tf.layers.conv2d(
      inputs=x,
      filters=64,
      kernel_size=3,
      padding="same",
      strides=1,
      #activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )
e=tf.layers.batch_normalization(conv1,training=True)

aconv1=tf.nn.relu(e, 'relu')


conv2=tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=3,
      padding="same",
      strides=1,
      #activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )
e2=tf.layers.batch_normalization(conv2,training=True)

aconv2=tf.nn.relu(e2, 'relu')
pool2=tf.layers.max_pooling2d(inputs=aconv2, pool_size=[2, 2], strides=2)


conv4=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=3,
      padding="same",
      strides=1,
      #activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )
e3=tf.layers.batch_normalization(conv4,training=True)

aconv5=tf.nn.relu(e3, 'relu')
conv5=tf.layers.conv2d(
      inputs=aconv5,
      filters=128,
      kernel_size=3,
      padding="same",
      strides=1,
      #activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
      )
e4=tf.layers.batch_normalization(conv5,training=True)
aconv6=tf.nn.relu(e4, 'relu')
pool6=tf.layers.max_pooling2d(inputs=aconv6, pool_size=[2, 2], strides=2)

