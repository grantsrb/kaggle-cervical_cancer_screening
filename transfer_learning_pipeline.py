# External Imports
import numpy as np
from sklearn.utils import shuffle
import os

# Internal Imports
from utilities import inout
from utilities import image_manipulation as imanip
from utilities import miscellaneous as misc
from models import inceptionV4 as incept

############# User Defined Variables

batch_size = 100
image_shape = (299,299,3)
feature_extraction_only = False

training_csv = 'csvs/incept_train_set.csv'
valid_csv = 'csvs/incept_valid_set.csv'


############# Read in Data
X_train_paths, y_train = inout.get_split_data(training_csv)
X_valid_paths, y_valid = inout.get_split_data(valid_csv)
n_labels = max(y_train)+1

y_train = imanip.one_hot_encode(y_train, n_labels)
y_valid = imanip.one_hot_encode(y_valid, n_labels)


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
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense

init, flat_layer, weights = incept.create_inception_v4()
flat_layer = Dense(1001, activation='elu')(flat_layer)
outs = Dense(3, activation='elu')(flat_layer)

model = Model(inputs=init,outputs=outs)
model.load_weights(weights, by_name=True)

if feature_extraction_only:
    for i in range(len(model.layers[:-4])):
        model.layers[i].trainable = False

learning_rate = .0001
for i in range(20):
    if i > 4:
        learning_rate = .00001 # Anneals the learning rate
    adam_opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    history = model.fit_generator(train_generator, train_steps_per_epoch, epochs=1,
                        validation_data=valid_generator,validation_steps=valid_steps_per_epoch, max_q_size=1)
    model.save('./weights/inception_model.h5')
print('History test', history.history)
