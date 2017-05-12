from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, concatenate
from keras.layers.normalization import BatchNormalization


def cnn_model(first_conv_shapes=[(4,4),(3,3),(5,5)], conv_shapes=[(3,3),(5,5)], conv_depths=[12,12,11,8,8], dense_shapes=[100,50,3], image_shape=(256,256,3), n_labels=3):
    stacks = []
    pooling_filter = (2,2)
    pooling_stride = (2,2)

    inputs = Input(shape=image_shape)
    zen_layer = BatchNormalization()(inputs)

    for shape in first_conv_shapes:
        stacks.append(Conv2D(conv_depths[0], shape, padding='same', activation='elu')(inputs))
    layer = concatenate(stacks,axis=-1)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pooling_filter,strides=pooling_stride,padding='same')(layer)
    layer = Dropout(0.05)(layer)

    for i in range(1,len(conv_depths)):
        stacks = []
        for shape in conv_shapes:
            stacks.append(Conv2D(conv_depths[i],shape,padding='same',activation='elu')(layer))
        layer = concatenate(stacks,axis=-1)
        layer = BatchNormalization()(layer)
        layer = Dropout(i*10**-2+.05)(layer)
        layer = MaxPooling2D(pooling_filter,strides=pooling_stride, padding='same')(layer)

    layer = Flatten()(layer)
    fclayer = Dropout(0.1)(layer)

    for i in range(len(dense_shapes)-1):
        fclayer = Dense(dense_shapes[i], activation='elu')(fclayer)
    #     if i == 0:
    #         fclayer = Dropout(0.5)(fclayer)
        fclayer = BatchNormalization()(fclayer)

    outs = Dense(dense_shapes[-1], activation='softmax')(fclayer)

    return inputs, outs
