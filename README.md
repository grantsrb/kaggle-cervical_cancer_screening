# Kaggle, Intel, and MobileODT Cervical Cancer Screening
### April 27, 2017 (Ongoing)
## Satchel Grant


_April 27th, 2017 Performance Update: Model is achieving ~60% accuracy on validation set.
Pipeline Improvement TODO:
* Anneal learning rate
* Increase Type 1 image count
* Remove zoom-in scaling
* Visualize correct and incorrect predictions
* Pseudo Labeling
* Transfer Learning
* Ensembling_

## Overview
This is a project to use the medical images provided by Kaggle, Intel, and MobileODT to create a classification pipeline for cervical type. This can be useful for determining treatments and testing procedures when treating and diagnosing cervical cancer.

The pipeline in this project uses a Convolutional Neural Net (CNN) model written in Python using the Keras functional API for image classification. The model also uses various image manipulation libraries. The rest of this readme walks you through the different parts of the project.

## Image Preprocessing
#### Image Sizing
The provided images are of varying sizes and frequently have a high resolution. To create a CNN, the incoming data needs to be of uniform size. The data also needs to have a high enough resolution to be able to distinguish key characteristics in classification, but a low enough resolution to avoid computational limits.

Initially I resized each image to 256x256 pixels without maintaining the aspect ratio. It was difficult to get the model to do better than 30-40%. I figured that the resolution change in aspect ratio may make it more difficult for the model to train. It was also taking a long time to do the image manipulations, so I created a [script](./create_resized_imgs.py) to create and save resized images to disk without affecting their aspect ratio.

#### Image Distribution
The quantity of images was rather low even with the extra data provided by the competition(~6000 images). The images also had an unequal distribution of cervical types. About 50% of the data was Type 2 images. To equalize the data, I added and saved a mirrored version of each Type 1 and Type 3 image during the resizing step. This helped equalize the distribution with image counts of 2877, 4345, 4852 for types 1, 2, and 3 respectively. Obviously Type 1 image count still lagged so I added copies with increased brightness for two thirds of the dataset bringing it up to 4775.

#### Image Augmentations
To further increase the total images in the dataset, I added an option to add a randomly augmented version of each image to every batch. The random augmentations can be a rotation, translation, or scaling. This doubles the total data and theoretically should not affect the classifier's ability to classify the data. The only potentially difficult augmentation would be a down scaling. This could effectively reduce the resolution of the image which may make classification more difficult for the model.

#### Image Generator
Finally, I created an image generator to read each batch of images in on a separate thread in conjunction with the Keras fit_generator function. Each image is then duplicated with a random augmentation and used for training. The same is done for validation but without the image augmentations. Reading each image in individually reduces the amount of RAM required to train the classifier.

## CNN Model
#### Overview
When constructing the CNN architecture I reasoned that the first layer would likely be the most important due to the nuances of cervical types. I also was aware that the bigger the model, the longer it would take to train which costs money on AWS ):.

I started with a relatively small model and tried to fit a portion of the training set with limited dropout. This proved unsuccessful (accuracy ~35%) which indicated that the model was not complex enough to distinguish the cervical types or the image resolution was too low for the model to gain insight.

I increased the size and complexity of the model. This is the current state of the model and it is getting about 60% accuracy.

#### Model Architecture
To avoid picking a filter size for the convolutions, I decided to run a 3x3, 4x4, and 5x5 filter in parallel at the first layer with the largest convolutional depth. For subsequent convolutional layers I only ran a 3x3 and 5x5 filter with decreasing depths due to RAM limits.

I end the model with 2 fully connected layers decreasing in size followed by an output layer.


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 256x256x3 image   							|
| BatchNormalization         		| Centers and normalizes image pixels   							|
| Convolution 3x3, 4x4, 5x5     	| 1x1 stride, same padding, depth 12, outputs 256x256x36 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 128x128x36 				|
| Convolution 3x3, 5x5     	| 1x1 stride, same padding, depth 12, outputs 128x128x36	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 64x64x36 				|
| Convolution 3x3, 5x5     	| 1x1 stride, same padding, depth 11, outputs 64x64x33 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 32x32x33 				|
| Convolution 3x3, 5x5     	| 1x1 stride, same padding, depth 8, outputs 32x32x24 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 16x16x24 				|
| Convolution 3x3, 5x5     	| 1x1 stride, same padding, depth 8, outputs 16x16x24 	|
| ELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 8x8x24 				|
| Dropout	      	| 0.2 probability 				|
| Fully connected x100		| 1536x100, outputs x100        									|
| ELU					|												|
| Fully connected x50		| 100x50, outputs x50        									|
| ELU					|												|
| Fully connected	x3	| 50x3, outputs x3        									|


I chose to use ELU activations due to their protection against 'dead neurons' of the RELU activation.

RAM was a large consideration in the construction of the model. I would have liked to have greater depth in the initial convolutions and fully connected layers, but the g2.2xlarge instance would max out with too complex of an architecture. This additionally forced a smaller batch size and reduced worker queue in the Keras fit_generator function.
