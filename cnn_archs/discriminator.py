import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv2d_transpose(x, W, b, output, strides=2):
    # Conv2D transpose wrapper, with bias and relu activation
    x = tf.nn.conv2d_transpose(x, W, output_shape= output, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

def make_wts_and_bias(weights, biases, input_channels, output_channels, type):
    curr_weights_num = len(weights)
    curr_biases_num = len(biases)
    if type == 'normal':
        new_w = 'wc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(3, 3, input_channels, output_channels), initializer=tf.contrib.layers.xavier_initializer())
        new_b = 'b' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    elif type == 'transpose':
        new_w = 'wuc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(3, 3, output_channels, input_channels), initializer=tf.contrib.layers.xavier_initializer())
        new_b = 'ub' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    return weights, biases, new_w, new_b

def conv_block(x, weights, biases, output_channels):
    input_channels =  x.shape[3]
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, input_channels, output_channels, 'normal')
    y1 = conv2d(x, weights[new_w], biases[new_b])

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, output_channels, output_channels, 'normal')
    y2 = conv2d(y1, weights[new_w], biases[new_b])

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, output_channels, output_channels, 'normal')
    y3 = conv2d(y2, weights[new_w], biases[new_b])
    return y1, y2, y3, weights, biases


def conv_net(x):
    weights = {}
    biases = {}

    # block1
    conv1, conv2, conv3, weights, biases = conv_block(x, weights, biases, 32)

    # block2
    mp1 = maxpool2d(conv3, 2)
    conv4, conv5, conv6, weights, biases = conv_block(mp1, weights, biases, 32)

    # block3
    mp2 = maxpool2d(conv6, 2)
    conv7, conv8, conv9, weights, biases = conv_block(mp2, weights, biases, 32)

    # block4
    mp3 = maxpool2d(conv9, 2)
    conv10, conv11, conv12, weights, biases = conv_block(mp3, weights, biases, 32)

    # block5
    mp4 = maxpool2d(conv12, 2)
    conv13, conv14, conv15, weights, biases = conv_block(mp4, weights, biases, 32)

    flat = tf.reshape(conv15, shape= (-1, conv15.shape[1]*conv15.shape[2]*conv15.shape[3]))

    fc1_w = tf.get_variable('fc1_w', shape=(flat.shape[1], 512), initializer=tf.contrib.layers.xavier_initializer())
    fc1_b = tf.get_variable('fc1_b', shape=(512), initializer=tf.contrib.layers.xavier_initializer())
    fc1 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(flat, fc1_w), fc1_b))

    fc2_w = tf.get_variable('fc2_w', shape=(fc1.shape[1], 128), initializer=tf.contrib.layers.xavier_initializer())
    fc2_b = tf.get_variable('fc2_b', shape=(128), initializer=tf.contrib.layers.xavier_initializer())
    fc2 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(fc1, fc2_w), fc2_b))

    fc3_w = tf.get_variable('fc3_w', shape=(fc2.shape[1], 32), initializer=tf.contrib.layers.xavier_initializer())
    fc3_b = tf.get_variable('fc3_b', shape=(32), initializer=tf.contrib.layers.xavier_initializer())
    fc3 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(fc2, fc3_w), fc3_b))

    o_w = tf.get_variable('o_w', shape=(fc3.shape[1], 1), initializer=tf.contrib.layers.xavier_initializer())
    o_b = tf.get_variable('o_b', shape=(1), initializer=tf.contrib.layers.xavier_initializer())
    out = tf.nn.bias_add(tf.matmul(fc3, o_w), o_b)

    return out