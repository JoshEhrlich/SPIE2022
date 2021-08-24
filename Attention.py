# U-Net Model Construction - Attention Model

import unittest

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import l1, l2

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def segmentation_unet(input_size, num_classes, filter_multiplier=10, regularization_rate=0.):
    input_ = Input((input_size, input_size, 1))
    skips = []
    output = input_

    num_layers = 4
    down_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    down_filter_numbers = np.zeros([num_layers], dtype=int)
    up_conv_kernel_sizes = np.zeros([num_layers], dtype=int)
    up_filter_numbers = np.zeros([num_layers], dtype=int)
    
    for layer_index in range(num_layers):
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = (2**(layer_index)) * 16
        up_conv_kernel_sizes[layer_index] = int(2)
    up_filter_numbers = [128, 64, 32, 16]

    count = 0
    
    for shape, filters in zip(down_conv_kernel_sizes, down_filter_numbers):
        
        output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu"
                        )(output)
        output = BatchNormalization()(output)
        output = SpatialDropout2D(0.3)(output)
        output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu"
                        )(output)
        output = BatchNormalization()(output)
        skips.append(output)
        output = MaxPooling2D(pool_size = (2,2))(output)
        count += 1
    
    output = Conv2D((2*filters), (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu")(output)
    output = BatchNormalization()(output)
    output = SpatialDropout2D(0.3)(output)
    output = Conv2D((2*filters), (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu")(output)
    output = BatchNormalization()(output)
    count = 0
    

    for shape, filters in zip(up_conv_kernel_sizes, up_filter_numbers):
        
        output = Conv2DTranspose(filters, (shape,shape), strides=(2,2), padding='same')(output)
        skip_output = skips.pop()
        print([skip_output])
        
        #attention block starts
        a = Conv2D(filters, (1, 1), activation="relu", padding='same', strides = (1,1),
                            kernel_initializer = "he_normal")(output)
        b = Conv2D(filters, (1, 1), activation="relu", padding='same', strides = (1,1),
                            kernel_initializer = "he_normal")(skip_output)
        AttentionOutput = Activation("relu")(add([a,b]))
        
        AttentionOutput = Conv2D(1, (1, 1), activation="relu", padding='same', strides = (1,1),
                            kernel_initializer = "he_normal")(AttentionOutput)
        
        AttentionOutput = Activation("sigmoid")(AttentionOutput)
        AttentionOutput = multiply([a, AttentionOutput])
        #attention block ends
        
        output = concatenate([output, AttentionOutput])
        output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu"
                        )(output)  
        output = BatchNormalization()(output)
        output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu"
                        )(output)  
        output = BatchNormalization()(output)
    
    output = concatenate([output, output])
    output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu")(output) #, bias_regularizer=l1(regularization_rate),
    output = BatchNormalization()(output)
    output = Conv2D(filters, (shape, shape), kernel_initializer="he_normal", padding = "same", strides=(1,1),  activation="relu")(output) #, bias_regularizer=l1(regularization_rate),  
    output = BatchNormalization()(output)
        
    output = Conv2D(2, (shape, shape), strides = (1,1), activation="softmax", padding="same", bias_regularizer=l1(regularization_rate))(output)

    assert len(skips) == 0
    return Model([input_], [output])



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)

        return loss

    return loss


class SagittalSpineUnetTest(unittest.TestCase):
    def test_create_model(self):
        model = segmentation_unet(128, 2)

if __name__ == '__main__':
    unittest.main()