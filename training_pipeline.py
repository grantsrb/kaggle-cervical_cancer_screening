# External Imports
import numpy as np
from sklearn.utils import shuffle
import os

# Internal Imports
import inout
import image_manipulation as imanip
import model as mod

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

print(len(X_train_paths)) 
for dir_name, sdir, f_list in os.walk('./resized'):
	if 'Type_1' in dir_name:
		for f in f_list:		
			if '.jpg' in f and 'b15' in f:
				X_train_paths.append(os.path.join(dir_name,f))
				y_train.append(0)
print(len(X_train_paths)) 

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




############ Training Section
from keras.models import Sequential, Model
from keras import optimizers

inputs, outs = mod.cnn_model(first_conv_shapes, conv_shapes, conv_depths, dense_shapes, image_shape, n_labels)

model = Model(inputs=inputs,outputs=outs)
model.load_weights('gpu_model.h5')
learning_rate = .001
for i in range(20):
    if i > 5:
        learning_rate = .0001
    adam_opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    history = model.fit_generator(train_generator, train_steps_per_epoch, epochs=1,
                        validation_data=valid_generator,validation_steps=valid_steps_per_epoch, max_q_size=1)
    model.save('gpu_model_update.h5')
