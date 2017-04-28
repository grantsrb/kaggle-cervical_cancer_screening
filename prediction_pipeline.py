import time
import numpy as np

import inout
import prediction as pred
import model as mod

from multiprocessing.pool import ThreadPool

############### User Defined Variables
data_path = './test'
model_path = 'gpu_model_update.h5'
resize_dims = (256,256,3)
test_divisions = 20 # Used for segmenting image evaluation in threading
batch_size = 100 # Batch size used for keras predict function

############## Create Model
from keras.models import Sequential, Model

ins, outs = mod.cnn_model()
model = Model(inputs=ins,outputs=outs)
model.load_weights(model_path)

############# Read in Data
test_paths, test_labels, _ = inout.read_paths(data_path,no_labels=True)
print(str(len(test_paths))+' testing images')


############# Make Predictions
predictions = []
pool = ThreadPool(processes=1)
portion = len(test_paths)//test_divisions+1 # Number of images to read in per pool

async_result = pool.apply_async(inout.convert_images,(test_paths[0*portion:portion*(0+1)],
                                                test_labels[0*portion:portion*(0+1)],resize_dims))


total_base_time = time.time()
test_imgs = []
for i in range(1,test_divisions+1):
    base_time = time.time()

    print("Begin set " + str(i))
    while len(test_imgs) == 0:
        test_imgs,_ = async_result.get()
    img_holder = test_imgs
    test_imgs = []

    if i < test_divisions:
        async_result = pool.apply_async(inout.convert_images,(test_paths[i*portion:portion*(i+1)],
                                                        test_labels[0*portion:portion*(0+1)],resize_dims))

    predictions.append(model.predict(img_holder,batch_size=batch_size,verbose=0))
    print("Execution Time: " + str((time.time()-base_time)/60)+'min\n')

predictions = np.concatenate(predictions, axis=0)
print("Total Execution Time: " + str((time.time()-total_base_time)/60)+'mins')

############### Adjust confidence and Save Predictions

conf = .95 # Prediction confidence
predictions = pred.confidence(predictions, conf)

header = 'image_name,Type_1,Type_2,Type_3'
inout.save_predictions('submission.csv', test_labels, predictions, header)
