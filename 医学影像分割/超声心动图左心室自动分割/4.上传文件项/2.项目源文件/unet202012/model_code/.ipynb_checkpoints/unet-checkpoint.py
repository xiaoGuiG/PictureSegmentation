from keras.models import *
from keras.layers import *
from keras.optimizers import *
from model_code.loss_function import dice_coefficient_loss, dice_coefficient

def unet(input_size=(256, 256, 1), axis=3):
    inputs = Input(input_size)
    kernel_initializer = 'he_normal'
    # kernel_initializer = 'zeros'
    origin_filters = 32
    conv1 = Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)#kernal=3:kernal=(3,3)
    conv1 = Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(origin_filters*16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(origin_filters*16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(origin_filters * 8, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=axis)

    conv6 = Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(origin_filters*8, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    up7 = Conv2D(origin_filters*4, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=axis)
    conv7 = Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(origin_filters*4, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    up8 = Conv2D(origin_filters*2, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=axis)
    conv8 = Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(origin_filters*2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    up9 = Conv2D(origin_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=axis)
    conv9 = Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(origin_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    conv10 = Conv2D(3, 3, activation='sigmoid', padding='same')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss = dice_coefficient_loss, metrics=['accuracy', dice_coefficient])
    print(model.summary())
    return model

if __name__=='__main__':
    unet(input_size=(224,224,3))