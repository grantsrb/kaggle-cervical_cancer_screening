import os
import inout

brightness_delta = 15
brighten_type = 'Type_1'

for dir_name, subdir_list, file_list in os.walk(path):
    if brighten_type in dir_name:
        for f in file_list:
            if '.jpg' in f:
                inout.save_brightness(f,brightness_delta)
