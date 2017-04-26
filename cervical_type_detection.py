import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import scipy.misc as sci
import time
import random
import scipy.ndimage.interpolation as scizoom

%matplotlib inline

def show(img):
    plt.imshow(img)
    plt.show()


import os

def read_paths(path, no_labels=False, label_type=None):
    # ** Takes a root path and returns all of the file
    # paths within the root directory. It uses the
    # subdirectories to create a corresponding label array **

    # path - the path to the root directory
    # no_labels - optional argument to use file
    #             names as labels instead of subdirectory
    # label_type - optional argument to set specific label
    #             type for all files in path

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
    if n_labels: return file_paths, labels, n_labels
    return file_paths, labels


def one_hot_encode(labels, n_classes):
    # ** Takes labels as values and converts them into one_hot labels.
    # Returns numpy array of one_hot encodings **

    # labels - array or numpy array of single valued labels
    # n_classes - number of potential classes in labels
    one_hots = []
    for label in labels:
        one_hot = [0]*n_classes
        if label >= len(one_hot):
            print("Labels out of bounds\nCheck your n_classes parameter")
            return
        one_hot[label] = 1
        one_hots.append(one_hot)
    return np.array(one_hots,dtype=np.float32)


def get_split_data(file_name):
    paths = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            split_line = line.strip().split(',')
            paths.append(split_line[0])
            labels.append(int(split_line[1]))
    return paths,labels

X_train_paths, y_train = get_split_data('train_set.csv')
X_valid_paths, y_valid = get_split_data('valid_set.csv')
n_labels = max(y_train)+1

y_train = one_hot_encode(y_train, n_labels)
y_valid = one_hot_encode(y_valid, n_labels)

def rotate(image, angle, ones=None, random_fill=True, color_range=255):
    # ** Rotates an image by the specified angle amount
    # and fills in resulting space with random values **

    # image - the image as numpy array to be rotated
    # angle - the desired amount of rotation in degrees
    # ones - an numpy array of ones like the image with the same rotation
    #        (used for broadcasting random filling into black space from rotation)
    # no_random - optional boolean to remove random filling in black space
    # color_range - the range of color values for the random filling

    if not random_fill:
        return sci.imrotate(image, angle).astype(np.float32)
    elif ones == None:
        ones = sci.imrotate(np.ones_like(image),angle)
    rot_image = sci.imrotate(image, angle).astype(np.float32)
    edge_filler = np.random.random(rot_image.shape).astype(np.float32)*color_range
    rot_image[ones[:,:,:]!=1] = edge_filler[ones[:,:,:]!=1]
    return rot_image

def translate(img, row_amt, col_amt, color_range=255):
    # ** Returns a translated copy of an image by the specified row and column amount
    # and fills in the empty space with random values **

    # image - the source image as numpy array to be translated
    # row_shift - the maximum vertical translation in both directions in pixels
    # col_shift - the maximum horizontal translation in both directions in pixels
    # color_range - the range of color values for the random filling
    translation = np.random.random(img.shape).astype(img.dtype)*color_range
    if row_amt > 0:
        if col_amt > 0:
            translation[row_amt:,col_amt:] = img[:-row_amt,:-col_amt]
        elif col_amt < 0:
            translation[row_amt:,:col_amt] = img[:-row_amt,-col_amt:]
        else:
            translation[row_amt:,:] = img[:-row_amt,:]
    elif row_amt < 0:
        if col_amt > 0:
            translation[:row_amt,col_amt:] = img[-row_amt:,:-col_amt]
        elif col_amt < 0:
            translation[:row_amt,:col_amt] = img[-row_amt:,-col_amt:]
        else:
            translation[:row_amt,:] = img[-row_amt:,:]
    else:
        if col_amt > 0:
            translation[:,col_amt:] = img[:,:-col_amt]
        elif col_amt < 0:
            translation[:,:col_amt] = img[:,-col_amt:]
        else:
            return img.copy()
    return translation

def random_zoom(image, max_zoom=1/3.):
    # ** Returns a randomly zoomed (scaled) copy of an image within the scaling amount.
    # if the scaling zooms outward, the empty space is filled with random values **

    # image - the source image as numpy array to be scaled
    # max_zoom - the maximum scaling amount in either direction

    color_range = 255
    zoom_factor = 1 + (random.random()-0.5)*max_zoom
    while zoom_factor == 1:
        zoom_factor = 1 + (random.random()-0.5)*max_zoom
    # scipy's zoom function returns different size array
    # The following code ensures the zoomed image has same pixel size as initial image
    img_height, img_width = image.shape[:2]
    zoomed_h = round(img_height*zoom_factor)
    zoomed_w = round(img_width*zoom_factor)
    diff_h = abs(zoomed_h-img_height)
    diff_w = abs(zoomed_w-img_width)
    start_row = round(diff_h/2)
    start_col = round(diff_w/2)

    # Zoom in on image
    if zoom_factor > 1:
        end_row = start_row+img_height
        end_col = start_col+img_width
        zoom_img = scizoom.zoom(image,(zoom_factor,zoom_factor,1),output=np.uint8)[start_row:end_row,
                                                               start_col:end_col]
    # Zoom out on image
    elif zoom_factor < 1:
        temp = scizoom.zoom(image,(zoom_factor,zoom_factor,1),output=np.uint8)
        temp_height, temp_width = temp.shape[:2]
        zoom_img = np.random.random(image.shape)*color_range # Random pixels instead of black space for out zoom
        zoom_img[start_row:start_row+temp_height,
                 start_col:start_col+temp_width] = temp[:,:]
    else:
        return image.copy()
    return zoom_img.astype(np.float32)

def random_augment(image, rotation_limit=180, shift_limit=10, zoom_limit=1/3.):
    # ** Returns a randomly rotated, translated, or scaled copy of an image. **

    # image - source image as numpy array to be randomly augmented
    # rotation_limit - maximum rotation degree in either direction
    # shift_limit - maximum translation amount in either direction
    # zoom_limit - maximum scaling amount in either direction

    augmentation_type = random.randint(0,2)

    # Rotation
    if augmentation_type == 0:
        random_angle = random.randint(-rotation_limit,rotation_limit)
        while random_angle == 0:
            random_angle = random.randint(-rotation_limit,rotation_limit)
        aug_image = rotate(image,random_angle,random_fill=False)

    elif augmentation_type == 1:
        # Translation
        row_shift = random.randint(-shift_limit, shift_limit)
        col_shift = random.randint(-shift_limit, shift_limit)
        aug_image = translate(image,row_shift,col_shift)

    else:
        # Scale
        aug_image = random_zoom(image,max_zoom=zoom_limit)

    return aug_image
