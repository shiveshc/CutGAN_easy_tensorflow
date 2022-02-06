import tensorflow as tf

def net(x):
    x = tf.reshape(x, (-1, x.shape[1]*x.shape[2]*x.shape[3]))

    fc1_w = tf.get_variable('fc1_w', shape=(x.shape[1], 256), initializer=tf.contrib.layers.xavier_initializer())
    fc1_b = tf.get_variable('fc1_b', shape=(256), initializer=tf.contrib.layers.xavier_initializer())
    fc1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(x, fc1_w), fc1_b))

    fc2_w = tf.get_variable('fc2_w', shape=(fc1.shape[1], 256), initializer=tf.contrib.layers.xavier_initializer())
    fc2_b = tf.get_variable('fc2_b', shape=(256), initializer=tf.contrib.layers.xavier_initializer())
    fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2_w), fc2_b)

    return fc2