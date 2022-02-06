import tensorflow as tf

def conv2d(x, W, b, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv2d_transpose(x, W, b, output, strides=2):
    # Conv2D transpose wrapper, with bias and relu activation
    x = tf.nn.conv2d_transpose(x, W, output_shape= output, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

def make_wts_and_bias(weights, biases, input_channels, output_channels, kernel_size, type):
    curr_weights_num = len(weights)
    curr_biases_num = len(biases)
    if type == 'normal':
        new_w = 'wc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(kernel_size, kernel_size, input_channels, output_channels), initializer=tf.contrib.layers.xavier_initializer())
        new_b = 'b' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    elif type == 'transpose':
        new_w = 'wuc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(kernel_size, kernel_size, output_channels, input_channels), initializer=tf.contrib.layers.xavier_initializer())
        new_b = 'ub' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    return weights, biases, new_w, new_b


def batch_norm(x, bns):
    mean, var = tf.nn.moments(x, axes= [1, 2], keep_dims= True)
    epsilon = 1e-8

    curr_bn = int(len(bns)/2)
    beta = tf.get_variable('beta' + str(curr_bn + 1), shape= (x.shape[3]), initializer= tf.initializers.zeros)
    gamma = tf.get_variable('gamma' + str(curr_bn + 1), shape= (x.shape[3]), initializer= tf.initializers.ones)
    bns.append(beta)
    bns.append(gamma)

    # beta = tf.get_variable(tf.zeros([x.shape[3]]))
    # gamma = tf.Variable(tf.ones([x.shape[3]]))

    x = tf.divide(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
    x = gamma*x + beta

    return x


def conv_block(x,
               weights,
               biases,
               bns,
               output_channels,
               kernel_size,
               use_batch_norm,
               stride= 1,
               activation= 'linear'):

    input_channels =  x.shape[3]
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, input_channels, output_channels, kernel_size, 'normal')
    y1 = conv2d(x, weights[new_w], biases[new_b], stride)

    if use_batch_norm == 1:
        y1 = batch_norm(y1, bns)

    if activation == 'relu':
        y1 = tf.nn.relu(y1)
    elif activation == 'tanh':
        y1 = tf.nn.tanh(y1)
    elif activation == 'leaky_relu':
        y1 = tf.nn.leaky_relu(y1)
    else:
        pass

    return y1


def conv_net(x):
    weights = {}
    biases = {}
    bns = []


    conv1 = conv_block(x, weights, biases, bns, 64, 4, 0, stride= 2, activation='leaky_relu')

    conv2 = conv_block(conv1, weights, biases, bns, 128, 4, 1, stride= 2, activation='leaky_relu')

    conv3 = conv_block(conv2, weights, biases, bns, 256, 4, 1, stride= 1, activation='leaky_relu')

    conv4 = conv_block(conv3, weights, biases, bns, 512, 4, 1, stride= 1, activation='leaky_relu')

    conv5 = conv_block(conv4, weights, biases, bns, 1, 4, 0)

    return conv5