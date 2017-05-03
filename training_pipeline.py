# External Imports
import numpy as np
from sklearn.utils import shuffle
import os

# Internal Imports
import inout
import image_manipulation as imanip
import model as mod
import miscellaneous as misc

############# User Defined Variables

n_labels = 3
first_conv_shapes = [(4,4),(3,3),(5,5)]
conv_shapes = [(3,3),(5,5)]
conv_depths = [12,12,11,8,8]
dense_shapes = [100,50,n_labels]
batch_size = 100

image_shape = (256,256,3)

training_csv = 'train_set.csv'
valid_csv = 'valid_set.csv'


############# Read in Data
X_train_paths, y_train = inout.get_split_data(training_csv)
X_valid_paths, y_valid = inout.get_split_data(valid_csv)
n_labels = max(y_train)+1

<<<<<<< HEAD
=======

>>>>>>> 1e0572835c341b958f53ed623ced42a5b4c5386f
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

inputs, outs = mod.cnn_model(first_conv_shapes, conv_shapes, conv_depths, dense_shapes, image_shape, n_labels)

model = Model(inputs=inputs,outputs=outs)

model.load_weights('model_update.h5')
learning_rate = .00001

for i in range(20):
	adam_opt = optimizers.Adam(lr=learning_rate)
	model.compile(loss='categorical_crossentropy',
								optimizer=adam_opt,
								metrics=['accuracy'])
	if i > 0:
		model.fit_generator(train_generator,
						# train_steps_per_epoch,
						1,
						epochs=1,
						max_q_size=1)
	model.save('pseudo_model.h5')
	valid_predictions = model.predict_generator(valid_generator,
											valid_steps_per_epoch,
											max_q_size=1)

	correct = np.equal(np.argmax(valid_predictions,axis=1),
							np.argmax(y_valid,axis=1))
	acc = np.mean(correct)
	print("Validation accuracy:",acc)

	# Pseudolabeled Generator
	pseudo_fraction = 1/4.
	train_generator = inout.pseudo_generator(X_train_paths,
										y_train,
										X_valid_paths,
										valid_predictions,
										batch_size,
										resize_dims=resize_dims,
										pseudo_fraction=pseudo_fraction,
										randomly_augment=add_random_augmentations)

	n_samples = n_train_samples + n_valid_samples
	pseud_batch_size = batch_size*(1-pseudo_fraction)
	train_steps_per_epoch = misc.get_steps(n_samples,pseud_batch_size,n_augs=1)


#
# model.load_weights('./gpu_model_update.h5')
# learning_rate = .0001
# for i in range(20):
#     if i > 4:
#         learning_rate = .0001
#     adam_opt = optimizers.Adam(lr=learning_rate)
#     model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
#     history = model.fit_generator(train_generator, train_steps_per_epoch, epochs=1,
#                         validation_data=valid_generator,validation_steps=valid_steps_per_epoch, max_q_size=1)
#     model.save('./models/gpu_model_update.h5')
