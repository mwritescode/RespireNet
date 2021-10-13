from tensorflow.keras.layers import Conv2D, BatchNormalization,  MaxPooling2D, ReLU, Add, Input
from tensorflow.keras import Model

def resblock(inputs, channels, strides=1):
    x = Conv2D(channels, 3, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(channels, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    if strides > 1:
        inputs = Conv2D(channels, 1, strides=strides)(inputs)
        inputs = BatchNormalization()(inputs)
    x = Add()([inputs, x])
    x = ReLU()(x)
    return x

def resnet34(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, 2)(x)


    # First 3 resblocks, with 64 channels
    for _ in range(3):
        x = resblock(x, 64)
    
    # 4 resblocks with 128 channels, the first one with strides=2
    x = resblock(x, 128, strides=2)
    for _ in range(3):
        x = resblock(x, 128)
    
    # 6 resblocks with 256 channels, the first one with strides=2
    x = resblock(x, 256, strides=2)
    for _ in range(5):
        x = resblock(x, 256)
    
    # 3 resblocks with 512 channels, the first one with strices=2
    x = resblock(x, 512, strides=2)
    for _ in range(2):
        x = resblock(x, 512)
    
    return Model(inputs=inputs, outputs=x, name='ResNet34')

def resnet18(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, 2)(x)

    x = resblock(x, 64)
    x = resblock(x, 64)

    x = resblock(x, 128, strides=2)
    x = resblock(x, 128)

    x = resblock(x, 256, strides=2)
    x = resblock(x, 256)

    x = resblock(x, 512, strides=2)
    x = resblock(x, 512)

    return Model(inputs=inputs, outputs=x, name='ResNet18')
