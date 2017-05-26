import tensorflow as tf

def conv2d(layer, weights, biases, name, strides=[1,1,1,1], padding='SAME'):
    activs = tf.nn.conv2d(layer, weights, strides=strides, padding=padding, name=name+"_conv2d")
    activs = tf.nn.bias_add(activs, biases)
    return tf.nn.relu(activs, name=name+'_relu')

def dense(layer, weights, biases, name):
    activs = tf.matmul(layer, weights, name=name+"_matmul")
    return tf.nn.bias_add(activs, biases, name=name+"_biased")

def global_avg_pooling(layer, name):
    shape = tf.shape(layer)
    return tf.nn.avg_pool(layer, ksize=[1,shape[1],shape[2],1],
                                strides=[1,1,1,1], padding="VALID",
                                name=name+'_gapool')

def max_pooling(layer, name, k=2):
    return tf.nn.max_pool(layer, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name+'maxp')

def avg_pooling(layer, name, k=2):
    return tf.nn.avg_pool(layer, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name+'avgp')

def batch_norm(layer, name):
    return tf.layers.batch_normalization(inputs, name=name+"_batch_norm")

def res_block(layer, depths, name, filt_size=3):
    shape = tf.shape(layer, name=name+"_shape")
    ws = [tf.Variable(tf.truncated_normal([filt_size,filt_size,shape[-1],d]), name=name+"_w"+str(i)) for i,d in enumerate(depths)]
    bs = [tf.Variable(tf.zeros(d), name=name+"_b"+str(i)) for i,d in enumerate(depths)]
    activs = layer
    counts = [i for i in range(len(ws))]
    for i,w,b in zip(counts,ws,bs):
        activs = conv2d(activs, w, b, name+str(i))
        activs = batch_norm(activs, name+str(i))
    eye = tf.identity(layer, name=name+"_eye")
    return tf.add(eye,activs, name=name+"_add")

def create(inputs, n_classes):
    shape = tf.shape(inputs)
    init_weight = tf.Variable(tf.truncated_normal([5,5,shape[-1],20]), name="init_w")
    init_bias = tf.Variable(tf.zeros(20), name='init_b')

    normed = batch_norm(inputs)
    conv_1 = conv2d(normed, init_weight, init_bias, 'init_conv')

    res_1 = res_block(conv_1,[20,20,20],'res1',5)
    pool1 = max_pooling(res_1, 'res1')

    res_2 = res_block(pool1, [20,20,20], 'res2',3)
    pool2 = max_pooling(res_2, 'res2')

    res_3 = res_block(pool1, [20,20,20], 'res3',3)
    pool3 = max_pooling(res_3, 'res3')

    res_4 = res_block(pool1, [20,20,20], 'res4',3)
    pool4 = max_pooling(res_4, 'res4')

    penult_weight = tf.Variable(tf.truncated_normal([3,3,20,20]), name="penult_w")
    penult_bias = tf.Variable(tf.zeros(20), name='penult_b')
    penult_layer = conv2d(pool4, penult_weight, penult_bias, 'penult')
    penult_layer = global_avg_pooling(last_layer,'penultavg')

    fc_weight = tf.Variable(tf.truncated_normal([20, n_classes]))
    fc_bias = tf.Variable(tf.zeros([n_classes]))
    final_layer = dense(penult, fc_weight, fc_bias, name='final')

    return tf.nn.softmax(final_layer)
