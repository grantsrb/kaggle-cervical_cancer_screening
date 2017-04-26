import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy.misc as sci
from PIL import Image
import PIL
from itertools import count
import inout

def show(img):
    plt.imshow(img)
    plt.show()

def resize(path, maxsize=(256,256,3), save_path=None, add_flip=False):
    img = Image.open(path)
    img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    rand_img = (np.random.random(maxsize)*255).astype(np.uint8)
    padded_img = Image.fromarray(rand_img)
    padded_img.paste(img, ((maxsize[0]-img.size[0])//2,(maxsize[1]-img.size[1])//2))
    if save_path:
        padded_img.save(save_path)
    if add_flip:
        flip = padded_img.transpose(Image.FLIP_LEFT_RIGHT)
        if save_path:
            split_path = save_path.split('/')
            flip_path = '/'.join(split_path[:-1] + ['flipped_'+split_path[-1]])
            flip.save(flip_path)
        # return np.array(padded_img, dtype=np.float32), np.array(flip,dtype=np.float32)
    # return np.array(padded_img, dtype=np.float32)

def add_flips(paths, labels, types, save_imgs=True):
    flip_map = dict()
    for type_ in types:
        flip_map[type_] = True
    new_paths = []
    new_labels = []
    for i,path,label in zip(count(), paths, labels):
        if i % len(paths)//4 == 0: print("Finish quadrant")
        if label in flip_map:
            try:
                img = mpimg.imread(path)
                flip = np.fliplr(img)
                split_path = path.split('/')
                split_path[-1] = 'flipped_'+split_path[-1]
                new_path = '/'.join(split_path)
                new_paths.append(new_path)
                new_labels.append(label)
                if save_imgs:
                    im = Image.fromarray(flip)
                    im.save(new_path)
            except OSError:
                print("Error at sample " + path + ', type ' + str(i+1))
    return paths+new_paths, labels+new_labels


home = os.getcwd()
images_location = '/Volumes/WhiteElephant/cervical_cancer/'
os.chdir(images_location)

root_path = 'train'
image_paths, labels, n_labels = inout.read_paths(root_path)

root_paths = ['Type_1', 'Type_2', 'Type_3']
for i,root_path in enumerate(root_paths):
    new_paths, new_labels, _ = inout.read_paths(root_path,label_type=i)
    image_paths += new_paths
    labels += new_labels

maxsize = (256,256,3)
for i,path,label in zip(count(),image_paths,labels):
    split_path = path.split('/')
    new_path = 'size'+str(maxsize[0])+'_'+split_path[-1]
    new_path = '/'.join(['resized']+split_path[:-1]+[new_path])
    add_flip = True
    if label == 1:
        add_flip = False
    try:
        resize(path, save_path=new_path, add_flip=add_flip)
    except OSError:
        print("Error at path " + path)

print(len(image_paths))
print(len(labels))
for i,l in enumerate(labels):
    if type(l) is not int:
        print('Error at ' + str(i))
