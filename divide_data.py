from utilities import inout
from sklearn.utils import shuffle

train_csv = 'train_set.csv'
valid_csv = 'valid_set.csv'

root_path = 'resized/train'
image_paths, labels, n_labels = inout.read_paths(root_path)

# root_paths = ['resized/Type_1', 'resized/Type_2', 'resized/Type_3']
# for i,root_path in enumerate(root_paths):
#     new_paths, new_labels, _ = inout.read_paths(root_path,label_type=i)
#     image_paths += new_paths
#     labels += new_labels

image_paths, labels = shuffle(image_paths, labels)

training_portion = .8
split_index = int(training_portion*len(image_paths))
X_train_paths, y_train = image_paths[:split_index], labels[:split_index]
X_valid_paths, y_valid = image_paths[split_index:], labels[split_index:]

print("Train size: ")
print(len(X_train_paths))
print("Valid size: ")
print(len(X_valid_paths))

inout.save_paths(train_csv, X_train_paths, y_train)
inout.save_paths(valid_csv, X_valid_paths, y_valid)
