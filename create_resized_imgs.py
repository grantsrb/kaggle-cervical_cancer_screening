# External Imports
from itertools import count
import os

# Internal Imports
import inout
import image_manipulation as imanip

new_img_shape = (256,256,3)

# Read in file paths of images to be resized
path = 'train'
image_paths, labels, n_labels = inout.read_paths(path)

# root_paths = ['Type_1', 'Type_2', 'Type_3']
# for i,root_path in enumerate(root_paths):
#     new_paths, new_labels, _ = inout.read_paths(root_path,label_type=i)
#     image_paths += new_paths
#     labels += new_labels

# Resize and save images to same path in 'resized' folder
# If of Cervix type 1 or 3, a mirrored image is additionally saved
for i,path,label in zip(count(),image_paths,labels):
    split_path = path.split('/')
    new_path = 'size'+str(new_img_shape[0])+'_'+split_path[-1]
    new_path = '/'.join(['resized']+split_path[:-1]+[new_path])
    add_flip = True
    if label == 1:
        add_flip = False
    try:
        imanip.resize(path, save_path=new_path, add_flip=add_flip)
    except OSError:
        print("Error at path " + path)
