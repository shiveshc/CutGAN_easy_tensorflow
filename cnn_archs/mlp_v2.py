import tensorflow as tf

def net(x):

    fc1_w = tf.get_variable('fc1_w', shape=(x.shape[2], 256), initializer=tf.contrib.layers.xavier_initializer())
    fc1_b = tf.get_variable('fc1_b', shape=(256), initializer=tf.contrib.layers.xavier_initializer())
    fc1 = tf.nn.relu(tf.nn.bias_add(tf.tensordot(x, fc1_w, axes= [[2], [0]]), fc1_b))

    fc2_w = tf.get_variable('fc2_w', shape=(fc1.shape[2], 256), initializer=tf.contrib.layers.xavier_initializer())
    fc2_b = tf.get_variable('fc2_b', shape=(256), initializer=tf.contrib.layers.xavier_initializer())
    fc2 = tf.nn.bias_add(tf.tensordot(fc1, fc2_w, axes= [[2], [0]]), fc2_b)

    out = fc2* tf.sqrt(tf.reduce_sum(tf.square(fc2), axis= 2, keepdims= True))

    return out