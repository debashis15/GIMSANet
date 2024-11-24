# import the libraries
import os
import numpy as np
import cv2
from conf import myConfig as config
from pathlib import Path
from conf import myConfig as config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow.data as tfdata
import tensorflow.image as tfimage
import tensorflow.nn as nn
import tensorflow.train as tftrain
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from numpy import *
import random
import os
from glob import glob
import datetime
import argparse
import PIL
import tensorflow.keras.backend as K
from PIL import Image
from PIL import Image, ImageOps
os.environ["CUDA_VISIBLE_DEVICES"]='2'
# import the libraries
import os
import numpy as np
import cv2
#shape = (3, 3, 3, 1)
from pathlib import Path

def my_Hfilter(shape, dtype=None):
    f = np.array([
        [[[-1], [0], [1]], [[-1], [0], [1]], [[-1], [0], [1]]],   # Red channel kernel
        [[[-2], [0], [2]], [[-2], [0], [2]], [[-2], [0], [2]]],   # Green channel kernel
        [[[-1], [0], [1]], [[-1], [0], [1]], [[-1], [0], [1]]]    # Blue channel kernel
    ], dtype=np.float32)
    
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_Vfilter(shape, dtype=None):
    f = np.array([
        [[[-1], [-2], [-1]], [[-1], [-2], [-1]], [[-1], [-2], [-1]]],   # Red channel kernel
        [[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]],      # Green channel kernel
        [[[1], [2], [1]], [[1], [2], [1]], [[1], [2], [1]]]       # Blue channel kernel
    ], dtype=np.float32)

    assert f.shape == shape
    return K.variable(f, dtype='float32')

def block21_layer(img_shape=(None,None,3)):
    input_img= tf.keras.Input(shape=img_shape)

    a= Conv2D(49,(3,3), padding="same")(input_img)
    b=LeakyReLU()(a)

    Gh=Conv2D(filters=1, kernel_size = 3, kernel_initializer=my_Hfilter, padding='same',trainable=False)(input_img)
    Gv=Conv2D(filters=1, kernel_size = 3, kernel_initializer=my_Vfilter, padding='same',trainable=False)(input_img)
    Gx=K.sqrt(Gh*Gh + Gv*Gv)

    c= tf.keras.layers.Concatenate()([Gx,b])

    return Model(input_img, c)

# create CNN model
input_img=Input(shape=(None,None,3))

x = block21_layer()(input_img)

x1=Conv2D(49,(3,3), dilation_rate=2,padding="same")(x)
x1=LeakyReLU()(x1)

x1=Conv2D(49,(3,3), dilation_rate=3,padding="same")(x1)
x1=LeakyReLU()(x1)

x1 = Conv2D(49,(3,3), padding="same")(x1)
x1=LeakyReLU()(x1)

x2 = Conv2D(49,(3,3), padding="same")(x1)

x1 = Add()([x1, x2])

x1=Conv2D(49,(3,3), dilation_rate=3,padding="same")(x1)
x1=LeakyReLU()(x1)

x1=Conv2D(49,(3,3), dilation_rate=2,padding="same")(x1)
x1=LeakyReLU()(x1)

x1=Conv2D(50,(3,3), dilation_rate=1,padding="same")(x1)
x1=LeakyReLU()(x1)

x1 = Add()([x1, x])
x1 = Concatenate()([x1, x])

x1 = Conv2D(49,(3,3), padding="same")(x1)

#MSAB 

# 2x2
x2 = Conv2D(49,(2,2), padding="same")(x1)
x3 = Conv2D(49,(2,2), padding="same")(x2)

x3=Conv2D(49,(2,2), dilation_rate=1,padding="same")(x3)
x3=LeakyReLU()(x3)

x3=Conv2D(49,(2,2), dilation_rate=2,padding="same")(x3)
x3=LeakyReLU()(x3)

x3=Conv2D(49,(2,2), dilation_rate=3,padding="same")(x3)
x3=LeakyReLU()(x3)

x3=Conv2D(49,(2,2), dilation_rate=2,padding="same")(x3)
x3=LeakyReLU()(x3)

x3=Conv2D(49,(2,2), dilation_rate=1,padding="same")(x3)
x3=LeakyReLU()(x3)

x3 = Conv2D(49,(2,2), padding="same")(x3)
x3=Activation('sigmoid')(x3)

x3 = Multiply()([x3,x2])

# 3x3
x2 = Conv2D(49,(3,3), padding="same")(x1)
x4 = Conv2D(49,(3,3), padding="same")(x2)

x4=Conv2D(49,(3,3), dilation_rate=1,padding="same")(x4)
x4=LeakyReLU()(x4)

x4=Conv2D(49,(3,3), dilation_rate=2,padding="same")(x4)
x4=LeakyReLU()(x4)

x4=Conv2D(49,(3,3), dilation_rate=3,padding="same")(x4)
x4=LeakyReLU()(x4)

x4=Conv2D(49,(3,3), dilation_rate=2,padding="same")(x4)
x4=LeakyReLU()(x4)

x4=Conv2D(49,(3,3), dilation_rate=1,padding="same")(x4)
x4=LeakyReLU()(x4)

x4 = Conv2D(49,(3,3), padding="same")(x4)
x4=Activation('sigmoid')(x4)

x4 = Multiply()([x4,x2])

# 4x4
x2 = Conv2D(49,(4,4), padding="same")(x1)
x5 = Conv2D(49,(4,4), padding="same")(x2)

x5=Conv2D(49,(4,4), dilation_rate=1,padding="same")(x5)
x5=LeakyReLU()(x5)

x5=Conv2D(49,(4,4), dilation_rate=2,padding="same")(x5)
x5=LeakyReLU()(x5)

x5=Conv2D(49,(4,4), dilation_rate=3,padding="same")(x5)
x5=LeakyReLU()(x5)

x5=Conv2D(49,(4,4), dilation_rate=2,padding="same")(x5)
x5=LeakyReLU()(x5)

x5=Conv2D(49,(4,4), dilation_rate=1,padding="same")(x5)
x5=LeakyReLU()(x5)

x5 = Conv2D(49,(4,4), padding="same")(x5)
x5=Activation('sigmoid')(x5)

x5 = Multiply()([x5,x2])

# 5x5
x2 = Conv2D(49,(5,5), padding="same")(x1)
x6 = Conv2D(49,(5,5), padding="same")(x2)

x6=Conv2D(49,(5,5), dilation_rate=1,padding="same")(x6)
x6=LeakyReLU()(x6)

x6=Conv2D(49,(5,5), dilation_rate=2,padding="same")(x6)
x6=LeakyReLU()(x6)

x6=Conv2D(49,(5,5), dilation_rate=3,padding="same")(x6)
x6=LeakyReLU()(x6)

x6=Conv2D(49,(5,5), dilation_rate=2,padding="same")(x6)
x6=LeakyReLU()(x6)

x6=Conv2D(49,(5,5), dilation_rate=1,padding="same")(x6)
x6=LeakyReLU()(x6)

x6 = Conv2D(49,(5,5), padding="same")(x6)
x6=Activation('sigmoid')(x6)

x6 = Multiply()([x6,x2])

x1 = Add()([x3, x4, x5, x6])


x1 = Conv2D(50,(3,3), padding="same")(x1)
x1 = Add()([x1, x])

x1 = Conv2D(50,(3,3), padding="same")(x1)
x1 = Add()([x1, x])

x1 = Conv2D(3,(3,3), padding="same")(x1)
x1 = Add()([x1, input_img])
mdl=Model(input_img,x1)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.0001
    factor=0.5
    dropEvery=25
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(learning_rate=0.0001)
def custom_loss(y_true,y_pred):
    diff=abs(y_true-y_pred)
    l1=K.sum(diff)/(config.batch_size)
    # l1=(diff)/(config.batch_size)
    return l1
mdl.compile(loss=custom_loss,optimizer=opt)
mdl.summary()

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest", validation_split=0.2)


def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(1,61)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield(noisyImagesBatch,batch)



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./training_checkpoints','ckpt_{epoch:03d}'), verbose=1,save_freq='epoch')
logdir = "./training_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
lr_callback = [LearningRateScheduler(lr_decay)]
# train
mdl.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=[lr_callback, cp_callback, tensorboard_callback], verbose=1)


