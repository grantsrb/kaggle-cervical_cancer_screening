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

def histdict(arr, n_labels):
    histd = dict()
    for x in arr:
        if x in histd:
            histd[x] += 1
        else:
            histd[x] = 1
    return histd

def get_accuracy(loss, num_samples, confidence, n_classes, start_acc=.5, acc_increment=0.0000001, acc_precision=1):
    # ** Calculates approximate accuracy given a kaggle log-loss evaluation.
    #   Returns both approximate accuracy calculation and resulting loss
    #   calculation from approximated accuracy **

    # loss - Real loss (from Kaggle) as float
    # num_samples - integer of number of samples used in loss calculation
    # confidence - float of the confidence of the prediction (assumes remaining confidence equally distributed)
    # n_classes - integer of number of classes in prediction
    # start_acc - optional float indicating lower bound of approximate accuracy
    # acc_increment - optional float indicating degree of increment in accuracy approximation
    # acc_precision - optional float indicating degree of precision in accuracy approximation
    
    loss = loss*num_samples
    calculated_loss = 0
    accuracy = start_acc
    while (calculated_loss-loss)**2 > acc_precision and accuracy <= 1:
        accuracy += acc_increment
        num_right = int(num_samples*accuracy)
        num_wrong = num_samples-num_right
        calculated_loss = -(num_right*math.log(confidence) + num_wrong*math.log((1-confidence)/(n_classes-1)))
    return accuracy, calculated_loss/num_samples
