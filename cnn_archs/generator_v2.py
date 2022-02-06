import tensorflow as tf

def conv2d(x, W, b, strides):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding= 'SAME')
    x = tf.nn.bias_add(x, b)
    return x


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv2d_transpose(x, W, b, output, strides):
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
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(kernel_size, kernel_size, output_channels, input_channels), initializer=tf.random_normal_initializer(0, 0.02))
        new_b = 'ub' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.random_normal_initializer(0, 0.02))
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
    else:
        pass

    return y1


def convT_block(x,
                weights,
                biases,
                bns,
                kernel_size,
                output_shape,
                use_batch_norm,
                stride= 2,
                activation= 'linear'):

    input_channels =  x.shape[3]
    output_channels = output_shape[3]
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, input_channels, output_channels, kernel_size, 'transpose')
    y1 = conv2d_transpose(x, weights[new_w], biases[new_b], output_shape, stride)

    if use_batch_norm == 1:
        y1 = batch_norm(y1, bns)

    if activation == 'relu':
        y1 = tf.nn.relu(y1)
    elif activation == 'tanh':
        y1 = tf.nn.tanh(y1)
    else:
        pass

    return y1


def ResBlock(x,
             weights,
             biases,
             bns,
             output_channels,
             kernel_size,
             use_batch_norm):

    y1 = conv_block(x,
                    weights,
                    biases,
                    bns,
                    output_channels,
                    kernel_size,
                    use_batch_norm,
                    activation= 'relu')

    y2 = conv_block(y1,
                    weights,
                    biases,
                    bns,
                    output_channels,
                    kernel_size,
                    use_batch_norm,
                    activation='relu')

    return y2 + x



def conv_net(x, output_shape):
    weights = {}
    biases = {}
    bns = []
    layers = {}
    l = 0

    conv1 = conv_block(x, weights, biases, bns, 64, 7, 1, activation= 'relu')
    layers[l] = conv1
    l = l + 1

    conv2 = conv_block(conv1, weights, biases, bns, 128, 3, 1, stride= 2, activation= 'relu')
    layers[l] = conv2
    l = l + 1

    conv3 = conv_block(conv2, weights, biases, bns, 256, 3, 1, stride= 2, activation= 'relu')
    layers[l] = conv3
    l = l + 1

    conv_x = conv3
    for i in range(9):
        conv_x = ResBlock(conv_x, weights, biases, bns, 256, 3, 1)
        layers[l] = conv_x
        l = l + 1

    conv4 = convT_block(conv_x, weights, biases, bns, 3, [tf.shape(conv_x)[0], 128, 128, 128], 1, stride= 2, activation= 'relu')
    layers[l] = conv4
    l = l + 1

    conv5 = convT_block(conv4, weights, biases, bns, 3, [tf.shape(conv4)[0], 256, 256, 64], 1, stride= 2, activation='relu')
    layers[l] = conv5
    l = l + 1

    out = conv_block(conv5, weights, biases, bns, output_shape, 7, 0, activation= 'tanh')
    layers[l] = out
    l = l + 1

    return out, [layers[3], layers[5], layers[7], layers[9], layers[11]]

