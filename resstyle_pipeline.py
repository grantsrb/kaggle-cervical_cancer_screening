import numpy as np
from sklearn.utils import shuffle
import os
import time

# Internal Imports
from utilities import inout
from utilities import image_manipulation as imanip
from models import tf_res_style as res
from utilities import miscellaneous as misc


n_classes = 3
batch_size = 110

image_shape = (256,256,3)

training_csv = 'csvs/train_set.csv'
valid_csv = 'csvs/valid_set.csv'


############# Read in Data
X_train_paths, y_train = inout.get_split_data(training_csv)
X_valid_paths, y_valid = inout.get_split_data(valid_csv)
n_classes = max(y_train)+1

y_train = imanip.one_hot_encode(y_train, n_classes)
y_valid = imanip.one_hot_encode(y_valid, n_classes)


############### Image Generator Parameters

add_random_augmentations = True
resize_dims = None


n_train_samples = len(X_train_paths)
train_steps_per_epoch = misc.get_steps(n_train_samples,batch_size,n_augs=1)

n_valid_samples = len(X_valid_paths)
valid_steps_per_epoch = misc.get_steps(n_valid_samples,batch_size,n_augs=0)

train_generator = inout.image_generator(X_train_paths,
                                  y_train,
                                  batch_size,
                                  resize_dims=resize_dims,
                                  randomly_augment=add_random_augmentations)
valid_generator = inout.image_generator(X_valid_paths, y_valid,
                                  batch_size, resize_dims=resize_dims,
								  rand_order=False)


############ Training Section
import tensorflow as tf

tfimgs = tf.placeholder(tf.float32, [None]+[s for s in image_shape])
tflabels = tf.placeholder(tf.float32, [None, n_classes])

tensors = res.get_tensors(image_shape)
logits = res.create(tfimgs, tensors)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tflabels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

n_equals = tf.equal(tf.argmax(logits, 1), tf.argmax(tflabels, 1))
accuracy = tf.reduce_mean(tf.cast(n_equals, tf.float32))

saver = tf.train.Saver()
init = tf.global_variables_initializer()

epochcost = []
epochacc = []

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess, "./res_weights.ckpt")
    print("Begin Session")
    for epoch in range(100):
        basetime = time.time()
        traincost, trainacc = 0, 0
        for batch in range(train_steps_per_epoch):
            imgs, labs = next(train_generator)
            opt, cost, acc = sess.run([optimizer, loss, accuracy], feed_dict={tfimgs: imgs, tflabels: labs})
            print((batch+1), "/ {} ".format(train_steps_per_epoch), "  Cost:",cost, "- Acc:", acc, end='\r')
            traincost += cost
            trainacc += acc
        print("Training Cost:", traincost/train_steps_per_epoch, "- Accuracy:", trainacc/train_steps_per_epoch)

        validcost, validacc = 0, 0
        for batch in range(valid_steps_per_epoch):
            imgs, labs = next(train_generator)
            opt, cost, acc = sess.run([optimizer, loss, accuracy], feed_dict={tfimgs: imgs, tflabels: labs})
            validcost += cost
            validacc += acc
        print("Validation Cost:", validcost/valid_steps_per_epoch, "- Accuracy:", validacc/valid_steps_per_epoch)

        saver.save(sess, 'res_weights.ckpt')
        print("Execution Time:",time.time()-basetime, 's')




#
