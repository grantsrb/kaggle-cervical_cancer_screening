import os
import inout

path = './resized'
brightness_delta = 15
brighten_type = 'Type_1'
addition_limit = int(.66*2000)
count = 0

with open('train_set.csv', 'r') as f:
    for line in f:
        split_path = line.split(',')
        if brighten_type in split_path[0] and count < addition_limit:
            inout.save_brightness(split_path[0],brightness_delta)
            count += 1

total = 0
for d, s, flist in os.walk('./resized'):
	if 'Type_1' in d:
		for f in flist:
			total+=1
print(total)

