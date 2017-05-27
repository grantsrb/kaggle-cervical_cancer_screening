import numpy as np
import tensorflow as tf

def conv2d(layer, weights, biases, name, strides=[1,1,1,1], padding='SAME'):
    activs = tf.nn.conv2d(layer, weights, strides=strides, padding=padding, name=name+"_conv2d")
    activs = tf.nn.bias_add(activs, biases)
    return tf.nn.relu(activs, name=name+'_relu')

def dense(layer, weights, biases, name):
    activs = tf.matmul(layer, weights, name=name+"_matmul")
    return tf.nn.bias_add(activs, biases, name=name+"_biased")

def global_avg_pooling(layer,k, name):
    return tf.nn.avg_pool(layer, ksize=[1,k,k,1],
                                strides=[1,1,1,1], padding="VALID",
                                name=name+'_gapool')

def max_pooling(layer, name, k=2):
    return tf.nn.max_pool(layer, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name+'maxp')

def avg_pooling(layer, name, k=2):
    return tf.nn.avg_pool(layer, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name+'avgp')

def batch_norm(layer, name):
    return tf.layers.batch_normalization(layer, name=name+"_batch_norm")

def res_block(layer, ws, bs, eye, name):
    activs = layer
    if eye != None:
        layer = tf.matmul(layer, eye, name=name+"eye")
    counts = [i for i in range(len(ws))]
    for i,w,b in zip(counts,ws,bs):
        activs = conv2d(activs, w, b, name+str(i))
        activs = batch_norm(activs, name+str(i))
    return tf.add(layer,activs, name=name+"_add")

def get_tensors(img_shape, n_classes=3, res_depths=[20,20,20], n_resblocks=4):
    tensordict = dict()
    tensordict['n_classes'] = n_classes
    tensordict['res_depths'] = res_depths
    tensordict['n_resblocks'] = n_resblocks

    init_weight = tf.Variable(tf.truncated_normal([5,5,img_shape[-1],res_depths[-1]]), name="init_w")
    init_bias = tf.Variable(tf.zeros(20), name='init_b')
    tensordict['init'] = (init_weight, init_bias)

    for i in range(n_resblocks):
        ws = [tf.Variable(tf.truncated_normal([3,3, res_depths[j],res_depths[j+1]]),
                                            name="resblock"+str(i)+"_w"+str(j)) for j in range(len(res_depths)-1)]
        bs = [tf.Variable(tf.zeros(res_depths[j+1]),
                            name="resblock"+str(i)+"_b"+str(j)) \
                            for j in range(len(res_depths)-1)]
        initial_eye = np.asarray([np.eye(img_shape[0]//(2**i)) for j in range(res_depths[-1])])
        initial_eye = np.transpose(initial_eye,(2,0,1))
        eye = tf.Variable(initial_value=initial_eye, name="ident"+str(i))
        tensordict['resblock'+str(i)] = (ws, bs, eye)

    penult_weight = tf.Variable(tf.truncated_normal([3,3,res_depths[-1],res_depths[0]]), name="penult_w")
    penult_bias = tf.Variable(tf.zeros(res_depths[0]), name='penult_b')
    tensordict['penult'] = (penult_weight, penult_bias)

    tensordict['glob_avg_pool'] = img_shape[0]//(2**n_resblocks)

    dense_w = tf.Variable(tf.truncated_normal([res_depths[0], n_classes]))
    dense_b = tf.Variable(tf.zeros([n_classes]))
    tensordict['dense'] = (dense_w, dense_b)

    return tensordict


def create(inputs, tensordict, identity=False):

    normed = batch_norm(inputs, name='normed')
    init_weight, init_bias = tensordict['init']
    res_layer = conv2d(normed, init_weight, init_bias, 'init_conv')

    n_resblocks = tensordict['n_resblocks']
    for i in range(n_resblocks):
        ws, bs, eye = tensordict['resblock'+str(i)]
        if not identity: eye = None
        res_layer = res_block(res_layer, ws, bs, eye, 'res'+str(i))
        res_layer = max_pooling(res_layer, 'res'+str(i))

    penult_weight, penult_bias = tensordict['penult']
    penult_layer = conv2d(res_layer, penult_weight, penult_bias, 'penult')

    k = tensordict['glob_avg_pool']
    penult_layer = global_avg_pooling(penult_layer,k,'penult')
    vector_len = tensordict['res_depths'][0]
    penult_layer = tf.reshape(penult_layer, [-1, vector_len])

    dense_w, dense_b = tensordict['dense']
    final_layer = dense(penult_layer, dense_w, dense_b, name='final')

    return final_layer
