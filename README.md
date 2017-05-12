# Kaggle, Intel, and MobileODT Cervical Cancer Screening
### April 27, 2017 (Ongoing)
## Satchel Grant


_May 3rd, 2017 Performance Update: Model is achieving ~65% accuracy on validation set._

## Overview
This is a project to use the medical images provided by Kaggle, Intel, and MobileODT to create a classification pipeline for cervical type. This can be useful for determining treatments and testing procedures when treating and diagnosing cervical cancer.

The pipeline in this project uses a Convolutional Neural Net (CNN) model written in Python using the Keras functional API for image classification. The model also uses various image manipulation libraries. The rest of this readme walks you through the different parts of the project.

## Improvement TODO

My current plan to improve results consists of the following.

#### Add 1x1 Convolution Layers
Google's inception net uses 1x1 convolutions before their 3x3 and 5x5 filters. The reason is that a 1x1 convolution can effectively pool the depth of a layer which can reduce the computational power needed to use and train the net. A layer that has a depth of 60 can be unwieldy for both a 3x3 and 5x5 filter to convolve. A 1x1 filter can reduce the depth to something more manageable. Additionally this step adds another nonlinearity (elu or otherwise) which can help the model. So I plan on constructing a new model with these considerations in mind.

#### Increase Depths (both total model depth and individual layer depths)
Another thing worth considering is increasing the total depth of the model and increasing the depth of later layers. I currently decrease the depths of each layer sequentially. This was under the theory that there are fewer features to look for later in the net after the initial features have been noticed. This theory, however, is potentially wrong. In Imagenet classification, the models often have increasing depths later in the process which, [when visualized](http://yosinski.com/deepvis), seem to be capturing an increasing number of possibilities built from the previous features in past convolutional layers. Greater depth in both the overall architecture and in the individual layers will hopefully improve the model's ability to classify cervixes.

#### Visualize Layers
To know if the model is working, it would be prudent to actually visualize what the model is detecting. Visualization should help inform what types of layers are working and what are not. The problem is that I have no idea how to classify cervixes. Thus seeing layer visualizations may not provide me with useful information.

#### Pseudo Labeling
This is a semi-suprevised learning technique. The goal is to use the unlabeled test data as training data to improve test predictions. After the model has been somewhat trained on the training set, the test data can be included in training by using the test predictions as their labels giving them "pseudo-labels".

In theory, the classifier can find structure in the pseudo-labeled data to help with classification. Some big questions are, how much pseudo labeling should be included in each epoch? In each batch? At what classification accuracies does pseudo labeling help? What types of datasets lend themselves more readily to pseudo labeling?

According to [these guys](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks), all of the unlabeled data should be used as pseudo data. The loss, however, should be calculated separately from the labeled data and multiplied by a constant. The intensity of this constant should increase as the model makes better and better predictions about the unlabeled data. This way, the model places increasing significance on the pseudo training as the pseudo labels get better and better.

An important implementation note is that the pseudo labels (the test set predictions) should be updated with each epoch. Another note is that I tried this technique for 2 epochs while using 50% training and 50% pseudo data for each training batch.

I'm actually unsure how to use different weights for different training data, so I may do a more rudimentary version of this technique.

#### Transfer Learning
A potential improvement would be to use a previously trained model that has been made for Imagenet classification. These models are made by really smart people with massive amounts of resources. It is easy to use their trained convolutional layers as the backbone of a new model. We can then append both fully connected (dense) and convolutional layers to the model and train only the added layers. It is most common to only add fully connected layers. It is worth noting that the Imagenet images are very different in nature from the cervix dataset. This may mean that the convolutional layers of a transferred model would need to be trained in addition to the fine-tuning layers for good results.

#### Ensembling
The process of ensembling is to train multiple models and use their combined outputs for prediction. Ensembling is often more stable and less prone to overfitting than individual models. I plan on trying to combine the transfer learning model with my own model for an ensembling implementation.


## Image Preprocessing
#### Image Sizing
The provided images are of varying sizes and frequently have a high resolution. To create a CNN, the incoming data needs to be of uniform size. The data also needs to have a high enough resolution to be able to distinguish key characteristics in classification, but a low enough resolution to avoid computational limits.

Initially I resized each image to 256x256 pixels without maintaining the aspect ratio. It was difficult to get the model to do better than 30-40%. I figured that the resolution change in aspect ratio may make it more difficult for the model to train. It was also taking a long time to do the image manipulations, so I created a [script](./create_resized_imgs.py) to create and save resized images to disk without affecting their aspect ratio.

#### Image Distribution
The quantity of images was rather low even with the extra data provided by the competition(~6000 images). The images also had an unequal distribution of cervical types. About 50% of the data was Type 2 images. To equalize the data, I added and saved a mirrored version of each Type 1 and Type 3 image during the resizing step. This helped equalize the distribution with image counts of 2877, 4345, 4852 for types 1, 2, and 3 respectively. Obviously Type 1 image count still lagged so I added copies with increased brightness bringing the count up to 4100.

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
