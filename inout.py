import os

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
    return file_paths, labels, max(labels)+1

def save_paths(file_name, paths, labels):
    with open(file_name, 'w') as csv_file:
        for path,label in zip(paths,labels):
            csv_file.write(path + ',' + str(label) + '\n')


def get_split_data(file_name):
    paths = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            split_line = line.strip().split(',')
            paths.append(split_line[0])
            labels.append(int(split_line[1]))
    return paths,labels
