# External Imports
import numpy as np
from sklearn.utils import shuffle
import os

# Internal Imports
import inout
import image_manipulation as imanip

############# User Defined Variables

n_labels = 3
first_conv_shapes = [(4,4),(3,3),(5,5)]
conv_shapes = [(3,3),(5,5)]
conv_depths = [12,12,11,8,8]
dense_shapes = [100,50,n_labels]

image_shape = (256,256,3)

training_csv = 'train_set.csv'
valid_csv = 'valid_set.csv'


############# Read in Data
X_train_paths, y_train = inout.get_split_data(training_csv)
X_valid_paths, y_valid = inout.get_split_data(valid_csv)
n_labels = max(y_train)+1

y_train = imanip.one_hot_encode(y_train, n_labels)
y_valid = imanip.one_hot_encode(y_valid, n_labels)


############### Image Generator Parameters
batch_size = 100
add_random_augmentations = True
resize_dims = None


n_train_samples = len(X_train_paths)
if add_random_augmentations:
    n_train_samples = 2*len(X_train_paths)

train_steps_per_epoch = n_train_samples//batch_size + 1
if n_train_samples % batch_size == 0: train_steps_per_epoch = n_train_samples//batch_size

valid_steps_per_epoch = len(X_valid_paths)//batch_size

train_generator = inout.image_generator(X_train_paths,
                                  y_train,
                                  batch_size,
                                  resize_dims=resize_dims,
                                  randomly_augment=add_random_augmentations)
valid_generator = inout.image_generator(X_valid_paths, y_valid,
                                  batch_size, resize_dims=resize_dims)




############ Keras Section
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, concatenate
from keras.layers.normalization import BatchNormalization
from keras import optimizers


stacks = []
pooling_filter = (2,2)
pooling_stride = (2,2)

inputs = Input(shape=image_shape)
zen_layer = BatchNormalization()(inputs)

for shape in first_conv_shapes:
    stacks.append(Conv2D(conv_depths[0], shape, padding='same', activation='elu')(inputs))
layer = concatenate(stacks,axis=-1)
layer = BatchNormalization()(layer)
layer = MaxPooling2D(pooling_filter,strides=pooling_stride,padding='same')(layer)
layer = Dropout(0.05)(layer)

for i in range(1,len(conv_depths)):
    stacks = []
    for shape in conv_shapes:
        stacks.append(Conv2D(conv_depths[i],shape,padding='same',activation='elu')(layer))
    layer = concatenate(stacks,axis=-1)
    layer = BatchNormalization()(layer)
    layer = Dropout(i*10**-2+.05)(layer)
    layer = MaxPooling2D(pooling_filter,strides=pooling_stride, padding='same')(layer)

layer = Flatten()(layer)
fclayer = Dropout(0.2)(layer)

for i in range(len(dense_shapes)-1):
    fclayer = Dense(dense_shapes[i], activation='elu')(fclayer)
#     if i == 0:
#         fclayer = Dropout(0.5)(fclayer)
    fclayer = BatchNormalization()(fclayer)

outs = Dense(dense_shapes[-1], activation='softmax')(fclayer)

model = Model(inputs=inputs,outputs=outs)
model.load_weights('model.h5')
learning_rate = .0001
for i in range(10):
    adam_opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    history = model.fit_generator(train_generator, train_steps_per_epoch, epochs=1,
                        validation_data=valid_generator,validation_steps=valid_steps_per_epoch, max_q_size=1)
    model.save('gpu_model.h5')
