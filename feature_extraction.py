import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Multiply, Lambda
from tensorflow.keras.layers import Reshape, Activation, Input, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.layers import ReLU, concatenate, AveragePooling3D
from tensorflow.keras.optimizers import SGD

# channel attention module
def channel_attention(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]

    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer=initializer, use_bias=True)
    shared_dense_two = Dense(channel, kernel_initializer=initializer, use_bias=True)

    avg_pool = GlobalAveragePooling3D()(input_tensor)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2, 3], keepdims=True))(input_tensor)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_tensor, cbam_feature])

# spatial attention module
def spatial_attention(input_tensor):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_tensor)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    cbam_feature = Conv3D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

    return Multiply()([input_tensor, cbam_feature])

# a 3D Convolutional Block
def conv_block(x, growth_rate):
    x1 = BatchNormalization()(x)
    x1 = ReLU()(x1)
    x1 = Conv3D(4 * growth_rate, kernel_size=(1, 1, 1), padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv3D(growth_rate, kernel_size=(3, 3, 3), padding='same', use_bias=False)(x1)
    x = concatenate([x, x1])
    return x

# a Dense Block with a specified number of layers
def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        x = conv_block(x, growth_rate)
    return x

# a Transition Layer with bottleneck architecture
def transition_layer(x, reduction):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(int(x.shape[-1] * reduction), kernel_size=(1, 1, 1), padding='same', use_bias=False)(x)
    x = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(x)
    return x

input_shape = (17, 224, 224, 3)
input_tensor = Input(shape=input_shape)

conv3d = Conv3D(
    filters=64,
    kernel_size=(7, 7, 7),
    strides=(1, 2, 2),
    padding='same',
    activation='relu'
)(input_tensor)

max_pool3d = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv3d)
channel_attention_output = channel_attention(max_pool3d)
spatial_attention_output = spatial_attention(channel_attention_output)

# DenseNet architecture with dense blocks and transition layers
growth_rate = 32
reduction = 0.5

# First dense block with 6 layers
dense_block_1 = dense_block(spatial_attention_output, 6, growth_rate)
transition_1 = transition_layer(dense_block_1, reduction)

# Second dense block with 12 layers
dense_block_2 = dense_block(transition_1, 12, growth_rate)
transition_2 = transition_layer(dense_block_2, reduction)

# Third dense block with 24 layers
dense_block_3 = dense_block(transition_2, 24, growth_rate)
transition_3 = transition_layer(dense_block_3, reduction)

# Fourth dense block with 16 layers
dense_block_4 = dense_block(transition_3, 16, growth_rate)

# Global Average Pooling after DenseNet
global_avg_pool = GlobalAveragePooling3D()(dense_block_4)

# Final layer before classification
output_tensor = Dense(2, activation='softmax')(global_avg_pool)  
combined_model = Model(inputs=input_tensor, outputs=output_tensor)

sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
combined_model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


