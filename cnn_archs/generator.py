import tensorflow as tf

def conv2d(x, W, b, strides=1, activation= 'relu'):
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
    return x

def make_wts_and_bias(weights, biases, input_channels, output_channels, kernel_size, type):
    curr_weights_num = len(weights)
    curr_biases_num = len(biases)
    if type == 'normal':
        new_w = 'wc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(kernel_size, kernel_size, input_channels, output_channels), initializer= tf.random_normal_initializer(0, 0.02))
        new_b = 'b' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer= tf.random_normal_initializer(0, 0.02))
    elif type == 'transpose':
        new_w = 'wuc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(kernel_size, kernel_size, output_channels, input_channels), initializer= tf.random_normal_initializer(0, 0.02))
        new_b = 'ub' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer= tf.random_normal_initializer(0, 0.02))
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

def convT_block(x,
               weights,
               biases,
               bns,
               output_shape,
               kernel_size,
               use_batch_norm,
               stride= 2,
               activation= 'linear'):

    input_channels =  x.shape[3]
    output_channels = output_shape.shape[3]
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, input_channels, output_channels, kernel_size, 'transpose')
    y1 = conv2d_transpose(x, weights[new_w], biases[new_b], output_shape, stride)

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

def down_block(x,
               weights,
               biases,
               bns,
               output_channels,
               kernel_size= 3,
               use_batch_norm = 1,
               stride = 1,
               activation= 'linear'):

    y1 = conv_block(x, weights, biases, bns, output_channels, kernel_size, use_batch_norm, stride, activation)
    y2 = conv_block(x, weights, biases, bns, output_channels, kernel_size, use_batch_norm, stride, activation)
    y3 = conv_block(x, weights, biases, bns, output_channels, kernel_size, use_batch_norm, stride, activation)

    return y1, y2, y3


def conv_net(x, output_shape):
    weights = {}
    biases = {}
    bns = []

    ####### Unet 1 #########
    # block1
    conv1, conv2, conv3 = down_block(x, weights, biases, bns, 32, activation= 'leaky_relu')

    # block2
    mp1 = maxpool2d(conv3, 2)
    conv4, conv5, conv6 = down_block(mp1, weights, biases, bns, 64, activation= 'leaky_relu')

    # block3
    mp2 = maxpool2d(conv6, 2)
    conv7, conv8, conv9 = down_block(mp2, weights, biases, bns, 128, activation= 'leaky_relu')

    # block3
    mp3 = maxpool2d(conv9, 2)
    conv10, conv11, conv12 = down_block(mp3, weights, biases, bns, 256, activation= 'leaky_relu')

    # up block 1
    uconv1 = convT_block(conv12, weights, biases, bns, [tf.shape(conv3)[0], 64, 64, 128], 3, 1, activation= 'leaky_relu')
    uconv1 = tf.concat([uconv1, conv9], axis=3)
    conv13, conv14, conv15 = down_block(uconv1, weights, biases, bns, 128, activation= 'leaky_relu')

    # up block 2
    uconv2 = convT_block(conv15, weights, biases, bns, [tf.shape(conv3)[0], 128, 128, 64], 3, 1, activation='leaky_relu')
    uconv2 = tf.concat([uconv2, conv6], axis=3)
    conv16, conv17, conv18 = down_block(uconv2, weights, biases, bns, 64, activation= 'leaky_relu')

    # up block 3
    uconv3 = convT_block(conv18, weights, biases, bns, [tf.shape(conv3)[0], 256, 256, 32], 3, 1, activation= 'leaky_relu')
    uconv3 = tf.concat([uconv3, conv3], axis=3)
    conv19, conv20, conv21 = down_block(uconv3, weights, biases, bns, 32, activation= 'leaky_relu')

    conv22 = conv_block(conv21, weights, biases, bns, output_shape, use_batch_norm= 0, activation= 'tanh')
    out1 = conv22

    return out1, [conv12, conv15, conv18, conv21]