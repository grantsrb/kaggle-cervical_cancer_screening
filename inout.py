# External Imports
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import scipy.misc as sci
import os

# Internal Imports
import image_manipulation as imanip

def read_paths(path, no_labels=False, label_type=None):
    # ** Takes a root path and returns all of the file
    # paths within the root directory. It uses the
    # subdirectories to create a corresponding label array **

    # path - the path to the root directory
    # no_labels - optional argument to use file
    #             names as labels instead of subdirectory

    file_paths = []
    labels = []
    labels_to_nums = dict()
    n_labels = None

    for dir_name, subdir_list, file_list in os.walk(path):
        if len(subdir_list) > 0:
            n_labels = len(subdir_list)
            for i,subdir in enumerate(subdir_list):
                labels_to_nums[subdir] = i
        else:
            type_ = dir_name.split('/')[-1]

        for img_file in file_list:
            if '.jpg' in img_file.lower():
                file_paths.append(os.path.join(dir_name,img_file))

                if no_labels: labels.append(img_file)
                elif type(label_type) is int: labels.append(label_type)
                else: labels.append(labels_to_nums[type_])

    if type(n_labels) is int: return file_paths, labels, n_labels
    else: return file_paths, labels, max(labels)+1

def save_paths(csv_file_path, paths, labels):
    # ** Saves each path and label to a line in the given csv file **

    # csv_file_path - string of file path to save information to
    # paths - list of image file paths as strings
    # labels - list of corresponding label types for images

    with open(csv_file_path, 'w') as csv_file:
        for path,label in zip(paths,labels):
            csv_file.write(path + ',' + str(label) + '\n')


def get_split_data(csv_file_path):
    # ** Returns image file paths and corresponding labels from csv file as lists **

    # csv_file - string of file path to save information to

    paths = []
    labels = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            split_line = line.strip().split(',')
            paths.append(split_line[0])
            labels.append(int(split_line[1]))
    return paths,labels

def convert_images(paths, labels, resize_dims=None, randomly_augment=False):
    # ** Reads in images from their paths, resizes the images and returns
    # the images with their corresponding labels. **

    # paths - the file paths to the images
    # labels - a numpy array of the corresponding labels to the images
    # resize_dims - the resizing dimensions for the image
    # add_zooms - optional parameter to add randomly scaled copies of the images to the output
    # randomly_augment - optional parameter to add randomly rotated,
    #                    translated, and scaled images to the output

    images = []
    new_labels = []
    for i,path in enumerate(paths):
        label = labels[i]
        try:
            img = mpimg.imread(path)
            if resize_dims:
                img = sci.imresize(img, resize_dims)
        except OSError:
            if i == 0:
                img = mpimg.imread(paths[i+1])
                if resize_dims:
                    img = sci.imresize(img, resize_dims)
                    img = imanip.random_augment(img)
                else:
                    img = imanip.random_augment(img)
                labels[i] = labels[i+1]
            elif i > 0:
                sub_index = -1
                if randomly_augment:
                    sub_index = -2
                img = imanip.random_augment(images[sub_index])
                labels[i] = labels[i-1]
            label = labels[i]
        images.append(img)
        if randomly_augment:
            images.append(imanip.random_augment(img))
            new_labels.append(label)
            new_labels.append(label)
    if randomly_augment:
        return np.array(images,dtype=np.float32), np.array(new_labels,dtype=np.float32)
    return np.array(images,dtype=np.float32), labels


def image_generator(file_paths, labels, batch_size, resize_dims=None, randomly_augment=False):
    # ** Generator function to convert image file paths to images with labels in batches. **

    # file_paths - an array of the image file paths as strings
    # labels - a numpy array of labels for the corresponding images
    # batch_size - the desired size of the batch to be returned at each yield
    # resize_dims - the desired x and y dimensions of the images to be read in
    # add_zooms - optional parameter add an additional randomly zoomed image to the batch for each file path
    # randomly_augment - optional parameter add an additional randomly rotated, translated,
    #         and zoomed image to the batch for each file path

    if randomly_augment:
        batch_size = int(batch_size/2) # the other half of the batch is filled with augmentations
    while 1:
        file_paths,labels = shuffle(file_paths,labels)
        for batch in range(0, len(file_paths), batch_size):
            images, batch_labels = convert_images(file_paths[batch:batch+batch_size],
                                                  labels[batch:batch+batch_size],
                                                  resize_dims=resize_dims,
                                                  randomly_augment=randomly_augment)
            yield images, batch_labels
