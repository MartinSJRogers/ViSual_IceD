# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:51:07 2023

@author: Martin Rogers: marrog@bas.ac.k
"""

from keras.layers import Conv2D, Input, MaxPooling2D
from keras.layers import Activation, Dropout, concatenate, BatchNormalization
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.optimizers import Adam
import h5py


def ofuse_pixel_error(y_true, y_pred):
    """
    Calculate pixelwise error (visualiced output - true classification)
    
    Parameters
    ----------
    y_true : TYPE: Array
        DESCRIPTION: Reference manually digitised patch of size ice extent
    y_pred : TYPE: Array
        DESCRIPTION: ViSual_IceD output.

    Returns
    -------
    TYPE: scaler (double)
        DESCRIPTION: mean of error values.

    """
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append(
                    (symbolic_weights[i], weight_values[i]))
                K.batch_set_value(weight_value_tuples)


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements cross entropy loss function.
    
    Parameters
    ----------
    y_true : TYPE: Array
        DESCRIPTION: Reference manually digitised patch of size ice extent
    y_pred : TYPE: Array
        DESCRIPTION: ViSual_IceD output.

    Returns
    -------
    TYPE: scaler (double)
        DESCRIPTION: cross entropy loss function output value
    """
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    # This converts back to logits
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    # Calculate class balance cross entropy
    cost = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred, labels=y_true, pos_weight=0.8)

    return cost


def upsampleLayer(in_layer, concat_layer, input_size, i, upsample_method):
    """
    Upsampling (=Decoder) layer building block
    Parameters
    ----------
    in_layer: input layer
    concat_layer: layer with which to concatenate
    input_size: input size fot convolution
    """

    # 30 * i equates to 30 then 60, 120 and 240.
    upsample = tf.image.resize(in_layer, [30*i, 30*i], method=upsample_method)
    print('upsample', upsample)
    upsample = concatenate([upsample, concat_layer])
    print('upsample', upsample)
    uconv = Conv2D(input_size, (1, 1), activation='relu', 
                   kernel_initializer='he_normal', padding="same")(upsample)

    a = BatchNormalization()(uconv)
    drop = Dropout(0.2)(a)
    # for final layer use smaller input size
    if input_size > 65:
        uconv_2 = Conv2D(input_size/2, (1, 1), activation='relu',
                         kernel_initializer='he_normal', padding="same")(drop)
    else:
        uconv_2 = Conv2D(input_size, (1, 1), activation='relu', 
                         kernel_initializer='he_normal', padding="same")(drop)
    a = BatchNormalization()(uconv_2)
    return a


def hed_sar(img_in_sar):
    """
    ViSual_IceD encoder branch using SAR imagery.
    
    Parameters
    ----------
    y_true : TYPE: Array
        DESCRIPTION: input SAR training image

    Returns
    -------
    TYPE: Keras object
        DESCRIPTION: model architecture
    """
    # Input
    img_input = img_in_sar

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1_sar')(img_input)
    batch_sar = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv2_sar')(batch_sar)
    batch_sar = BatchNormalization()(x)

    b1_pool_sar = MaxPooling2D((3, 3), strides=(
        3, 3), padding='same', name='block1_pool_sar')(batch_sar)  # 240 240 64

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1_sar')(b1_pool_sar)
    batch_sar = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv2_sar')(batch_sar)
    batch_sar = BatchNormalization()(x)

    b2_pool_sar = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='block2_pool_sar')(batch_sar)  # 120 120 128

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1_sar')(b2_pool_sar)
    batch_sar = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv2_sar')(batch_sar)
    batch_sar = BatchNormalization()(x)


    b3_pool_sar = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='block3_pool_sar')(batch_sar)  # 60 60 256

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1_sar')(b3_pool_sar)
    batch_sar = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2_sar')(batch_sar)
    batch_sar = BatchNormalization()(x)

    b4_pool_sar = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='block4_pool_sar')(batch_sar)  # 30 30 512

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1_sar')(b4_pool_sar)
    batch_sar = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv2_sar')(batch_sar)



    output = Activation('sigmoid', name='out')(x)
    # model
    model = Model(inputs=[img_input], outputs=output)
    filepath = ''
    load_weights_from_hdf5_group_by_name(model, filepath)

    return model
    # return b1_pool_sar.output, b2_pool_sar.output, b3_pool_sar.output, b4_pool_sar.output, b4_final_conv_sar.output


def hed_modis(img_in_mod):
    """
    ViSual_IceD encoder branch using MODIS imagery.
    
    Parameters
    ----------
    y_true : TYPE: Array
        DESCRIPTION: input MODIS training image

    Returns
    -------
    TYPE: Keras object
        DESCRIPTION: model architecture
    """
    
    # Input
    img_input = img_in_mod

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    batch_mod = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv2')(batch_mod)
    batch_mod = BatchNormalization()(x)
    b1_pool_mod = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='block1_pool')(batch_mod)  # 240 240 64

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(b1_pool_mod)
    batch_mod = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv2')(batch_mod)
    batch_mod = BatchNormalization()(x)
    b2_pool_mod = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='block2_pool')(batch_mod)  # 120 120 128

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(b2_pool_mod)
    batch_mod = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv2')(batch_mod)
    batch_mod = BatchNormalization()(x)
    b3_pool_mod = MaxPooling2D((2, 2), strides=(
        2, 2), padding='same', name='block3_pool')(batch_mod)  # 60 60 256

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1')(b3_pool_mod)
    batch_mod = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2')(batch_mod)

    output = Activation('sigmoid', name='out')(x)
    # model
    model = Model(inputs=[img_input], outputs=output)

    filepath = '/data/hpcdata/users/marrog/fuse3/src/model/vgg16_weights.h5'
    load_weights_from_hdf5_group_by_name(model, filepath)

    return model


def visual_iced(classes, upsample, l_rate):
    """
    ViSual_IceD decoder branch.
    
    Parameters
    ----------
    classes : TYPE: int
        DESCRIPTION: number of classes to segment image into
    upsample : TYPE: string
        DESCRIPTION: upsample method
    l_rate : TYPE: double
        DESCRIPTION: model learning rate

    Returns
    -------
    TYPE: Keras object
        DESCRIPTION: model architecture
    """

    # seperate SAR and visible image input
    img_input_visible = Input(shape=(240, 240, 3), name='input_visible')
    img_input_sar = Input(shape=(720, 720, 3), name='input_SAR')

    model_sar = hed_sar(img_input_sar)
    model_mod = hed_modis(img_input_visible)

    # get outputs of relevant layers from modis encoder
    c1_mod = model_mod.get_layer(
        "block1_conv2").output  # (None, 240,240, 64)
    c2_mod = model_mod.get_layer(
        "block2_conv2").output  # (None, 120, 120, 128)
    c3_mod = model_mod.get_layer(
        "block3_conv2").output  # (None, 60, 60, 256)
    c4_mod = model_mod.get_layer(
        "block4_conv2").output  # (None, 30, 30, 512)

    # get outputs of relevant layers from sar encoder
    c1_sar = model_sar.get_layer(
        "block2_conv2_sar").output  # (None, 240,240,128)
    c2_sar = model_sar.get_layer(
        "block3_conv2_sar").output  # (None, 120, 120, 256)
    c3_sar = model_sar.get_layer(
        "block4_conv2_sar").output  # (None, 60, 60, 512)
    c4_sar = model_sar.get_layer(
        "block5_conv2_sar").output  # (None, 30, 30, 512)

    conv1_comb = concatenate(
        [c1_mod, c1_sar], name='concat_1')  # (240, 240, 192)
    conv2_comb = concatenate(
        [c2_mod, c2_sar], name='concat_2')  # (120, 120, 384)
    conv3_comb = concatenate(
        [c3_mod, c3_sar], name='concat_3')  # (60, 60, 768)
    conv4_comb = concatenate(
        [c4_mod, c4_sar], name='concat_4')  # (30, 30, 1024)

    c6 = upsampleLayer(in_layer=conv4_comb, concat_layer=conv3_comb,
                       input_size=256, i=2, upsample_method=upsample)
    c7 = upsampleLayer(in_layer=c6, concat_layer=conv2_comb,
                       input_size=128, i=4, upsample_method=upsample)
    c8 = upsampleLayer(in_layer=c7, concat_layer=conv1_comb,
                       input_size=64, i=8, upsample_method=upsample)

    output_layer = Conv2D(2, (1, 1), padding="same",
                          activation="sigmoid", name='output_layer')(c8)

    model = Model(inputs=[img_input_visible, img_input_SAR],
                  outputs=[output_layer, output_layer])
    opt = Adam(learning_rate=l_rate)
    model.compile(loss={'output_layer': cross_entropy_balanced},
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
