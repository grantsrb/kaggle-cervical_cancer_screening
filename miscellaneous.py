def confidence(predictions,conf):
    # ** Sets a confidence for the predicted class. The remaining confidence (out
    #    of 1) is equally distributed to the other 9 classes. **

    # predictions - list of one_hot encoded predictions
    # conf - float from 0-1 of intended confidence level for prediction

    for i,prediction in enumerate(predictions):
        max_i = max_index(prediction)
        predictions[i][max_i] = conf
        for j in range(len(prediction)):
            if j != max_i:
                predictions[i][j] = (1-conf)/(len(prediction)-1)
    return predictions

def max_index(array):
    # ** Returns index of maximum value in an array **
    max_i = 0
    for j in range(1,len(array)):
        if array[j] > array[max_i]: max_i = j
    return max_i


def get_steps(n_samples,batch_size,n_augs=1):
    # ** Returns number of generation steps in a single epoch for a sample set **

    # n_samples - integer of number of samples in the dataset
    # n_augs - boolean denoting if random augmentations will be added per data pt

    n_samples = n_samples*(n_augs+1)

    steps_per_epoch = n_samples//batch_size + 1

    if n_samples % batch_size == 0:
        train_steps_per_epoch = n_samples//batch_size
    return steps_per_epoch
