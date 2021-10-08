from tensorflow.keras.layers import Dropout, Dense, ReLU, Input, GlobalAveragePooling2D
from tensorflow.keras import Model

from resnet import resnet34, resnet18

def respirenet(input_shape, num_classes=2, resnet_body='34'):
    inputs = Input(input_shape)
    if resnet_body == '34':
        backbone = resnet34(input_shape)
    else:
        backbone = resnet18(input_shape)
    
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_units=128)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_units=128)(x)
    x = ReLU()(x)
    x = Dense(num_units=num_classes)(x)

    return Model(inputs=inputs, outputs=x)

