# External Imports
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import scipy.misc as sci
import os

# Internal Imports
import image_manipulation as imanip

def read_paths(path, no_labels=False, label_type=None):
    # ** Takes a directory path and returns all of the file
    # paths within the directory. It uses the
    # subdirectories to create a corresponding label array **

    # path - string of path to the root directory
    # no_labels - optional boolean to use file
    #             names as labels instead of subdirectory
    # label_type - optional integer label for all paths to be read in

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
    # labels - list of corresponding label integers for images

    with open(csv_file_path, 'w') as csv_file:
        for path,label in zip(paths,labels):
            csv_file.write(path + ',' + str(label) + '\n')

def save_predictions(csv_file_path, names, predictions, header=None):
    # ** Saved predictions into csv with proper submission format for Kaggle **

    # csv_file_path - path to csv save file as string
    # names - list of the test image file names as strings
    # predictions - list of one_hot encoded predictions
    # header - first line of csv as string

    with open(csv_file_path, 'w') as f:
        f.write(header+'\n')
        for name,logit in zip(names,predictions):
            f.write(name+',')
            for i,element in enumerate(logit):
                if i == logit.shape[0]-1: f.write(str(element)+'\n')
                else: f.write(str(element)+',')


def get_split_data(csv_file_path):
    # ** Returns image file paths and corresponding labels from csv file as lists **

    # csv_file_path - string of file path to save information to

    paths = []
    labels = []
    with open(csv_file_path, 'r') as f:
        for line in f:
            split_line = line.strip().split(',')
            paths.append(split_line[0])
            labels.append(int(split_line[1]))
    return paths,labels

def convert_images(paths, labels, resize_dims=None, randomly_augment=False):
    # ** Reads in images from their paths, returns the images with their
    #   corresponding labels. **

    # paths - list of file paths to the images as strings
    # labels - a numpy array or list of the corresponding labels to the images
    # resize_dims - tuple of integer output dimensions to resize image.
    #               (does not maintain aspect ratio)
    # randomly_augment - optional boolean to add randomly rotated,
    #                    translated, and scaled images to the output

    images = []
    new_labels = []
    for i,path in enumerate(paths):
        try:
            img = mpimg.imread(path)
            if resize_dims:
                img = sci.imresize(img, resize_dims)
        except OSError:
            # Uses augmented version of next image in list
            if i == 0:
                img = mpimg.imread(paths[i+1])
                if resize_dims:
                    img = sci.imresize(img, resize_dims)
                    img = imanip.random_augment(img)
                else:
                    img = imanip.random_augment(img)
                labels[i] = labels[i+1]

            # Uses most recent original image
            elif i > 0:
                sub_index = -1
                if randomly_augment:
                    sub_index = -2
                img = imanip.random_augment(images[sub_index])
                labels[i] = labels[i-1]

        images.append(img)
        if randomly_augment:
            images.append(imanip.random_augment(img))
            new_labels.append(labels[i])
            new_labels.append(labels[i])
    if randomly_augment:
        return np.array(images,dtype=np.float32), np.array(new_labels,dtype=np.float32)
    return np.array(images,dtype=np.float32), labels


def image_generator(file_paths, labels, batch_size, resize_dims=None, randomly_augment=False):
    # ** Generator to convert image file paths to batches of images with labels. **

    # file_paths - an array of the image file paths as strings
    # labels - a numpy array of labels for the corresponding images
    # batch_size - integer of size of the batch to be returned at each yield
    # resize_dims - tuple of the desired x and y dimensions of the images
    # randomly_augment - boolean to add a randomly rotated, translated,
    #                   and zoomed version of each image to the batch

    if randomly_augment:
        batch_size = int(batch_size/2) # maintains batch size despite image additions

    while True:
        file_paths,labels = shuffle(file_paths,labels)
        for batch in range(0, len(file_paths), batch_size):
            images, batch_labels = convert_images(file_paths[batch:batch+batch_size],
                                                  labels[batch:batch+batch_size],
                                                  resize_dims=resize_dims,
                                                  randomly_augment=randomly_augment)
            yield images, batch_labels


def save_brightness(path,delta):
    img = mpimg.imread(path)
    sunshine = imanip.change_brightness(img,delta)
    save_img = Image.fromarray(sunshine.astype(np.uint8))
    split_path = path.split('/')
    split_path[-1] = 'b'+str(delta)+path
    new_path = '/'.join(split_path)
    save_img.save(new_path)

