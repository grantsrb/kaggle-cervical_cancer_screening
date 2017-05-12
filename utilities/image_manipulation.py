# External Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.misc as sci
import scipy.ndimage.interpolation as scizoom
from PIL import Image
import PIL
import random

# Internal Imports
from utilities import inout

def show(img):
    plt.imshow(img)
    plt.show()

def resize(path, maxsizes=(256,256,3), save_path=None, add_flip=False):
    # ** Takes an image file path, reads in the image, and resizes the image to
    #   the specified dimensions without distortion. Uses ratioresize() **

    # path - image file path to be read and resized
    # maxsizes - the dimensions to be resized to
    # save_path - optional path to save resized image to
    # add_flip - if marked true, a mirrored copy of the image is also created. If
    #           a save_path is given, the flipped image will be saved to the same
    #           path but with 'flipped_' placed infront of the file name.

    img = Image.open(path)
    resized_img = ratioresize(img,maxsizes)
    if save_path:
        resized_img.save(save_path)
    if add_flip:
        flip = resized_img.transpose(Image.FLIP_LEFT_RIGHT)
        if save_path:
            split_path = save_path.split('/')
            flip_path = '/'.join(split_path[:-1] + ['flipped_'+split_path[-1]])
            flip.save(flip_path)
        return np.array(resized_img, dtype=np.float32), np.array(flip,dtype=np.float32)
    return np.array(resized_img, dtype=np.float32)

def ratioresize(img, maxsizes):
    # ** Takes image as PIL image or numpy array and resizes the image maintaining
    #   the aspect ratio **

    # img - image as PIL image or numpy array
    # maxsizes - tuple of resize dimensions

    if type(img) == type(np.array([])):
        img = Image.fromarray(img)
    img.thumbnail(maxsizes, PIL.Image.ANTIALIAS)
    rand_img = (np.random.random(maxsizes)*255).astype(np.uint8)
    padded_img = Image.fromarray(rand_img)
    padded_img.paste(img, ((maxsizes[0]-img.size[0])//2,
                            (maxsizes[1]-img.size[1])//2))
    return padded_img

def change_brightness(image, delta):
    # ** Adds delta to each pixel in image. Pixel values stop at 255 **

    # image - image as numpy array
    # delta - amount to add to each pixel

    img = np.where((255 - image) < delta,255,image+delta) # avoids negative values
    return img

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

def random_zoom(image, max_zoom=1/6., allow_out_zooms=False):
    # ** Returns a randomly zoomed (scaled) copy of an image. If the
    #   scaling zooms outward, the empty space is filled with random values **

    # image - the source image as numpy array to be scaled
    # max_zoom - float of the maximum scaling amount in either direction
    # allow_out_zooms - boolean determining if outward zooming is permitted

    color_range = 255
    if allow_out_zooms:
        zoom_factor = 1 + (random.random()-.5)*max_zoom*2
    else:
        zoom_factor = 1 + random.random()*max_zoom
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

    return zoom_img.astype(np.float32)

def random_augment(image, rotation_limit=180, shift_limit=10,
                    zoom_limit=1/3., random_fill=True):
    # ** Returns a randomly rotated, translated, or scaled copy of an image. **

    # image - source image as numpy array to be randomly augmented
    # rotation_limit - maximum rotation degree in either direction
    # shift_limit - maximum translation amount in either direction
    # zoom_limit - maximum scaling amount in either direction
    # random_fill - adds random values to empty pixels after rotation.

    augmentation_type = random.randint(0,2)

    # Rotation
    if augmentation_type == 0:
        random_angle = random.randint(-rotation_limit,rotation_limit)
        while random_angle == 0:
            random_angle = random.randint(-rotation_limit,rotation_limit)
        aug_image = rotate(image,random_angle,random_fill=random_fill)

    # Translation
    elif augmentation_type == 1:
        row_shift = random.randint(-shift_limit, shift_limit)
        col_shift = random.randint(-shift_limit, shift_limit)
        aug_image = translate(image,row_shift,col_shift)

    # Scale
    else:
        aug_image = random_zoom(image,max_zoom=zoom_limit)

    return aug_image

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
