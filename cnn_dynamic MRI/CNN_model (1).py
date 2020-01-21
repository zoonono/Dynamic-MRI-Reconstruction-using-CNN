#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Conv2D,BatchNormalization,Activation,Flatten,Lambda
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import compressed_sensing as cs


#loading data
x_train=pickle.load(open("x_train.pickle","rb"))
x_kspace_train=pickle.load(open("x_ks_train.pickle","rb"))
mask=pickle.load(open("mask_ksp.pickle","rb"))
y_train=pickle.load(open("y_train.pickle","rb"))
mask_inv=np.invert(mask)
inv_mask=tf.convert_to_tensor(mask_inv,dtype='float32')
print(np.shape(inv_mask))
'''''''''
print(np.shape(mask_inv))
print(np.shape(mask))
plt.imshow(mask[20,:,:,0],cmap='gray')
plt.colorbar()
plt.show()
plt.imshow(mask_inv[20,:,:,0],cmap='gray')
plt.colorbar()
plt.show()
'''''''''


# In[2]:


def basic_model():
    input_img=Input(shape=(256,256,1), name='input_image')
    inv_mask=Input(shape=(256,256,1), name='input_mask')
    input_ksp=Input(shape=(256,256,1), name='input_ksp')
    cnn_1= Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal',data_format="channels_last")(input_img)
    cnn_1 = BatchNormalization()(cnn_1)
    cnn_1 = Activation('relu')(cnn_1)
    cnn_2= Conv2D(64, (3, 3), padding='same',name='conv2',kernel_initializer='he_normal',data_format="channels_last")(cnn_1)
    cnn_2= BatchNormalization()(cnn_2)
    cnn_2= Activation('relu')(cnn_2)
    cnn_3=Conv2D(64, (3, 3), padding='same', name='conv3',kernel_initializer='he_normal',data_format="channels_last")(cnn_2)
    cnn_3= BatchNormalization()(cnn_3)
    cnn_3= Activation('relu')(cnn_3)
    cnn_4=Conv2D(1, (3, 3), padding='same', name='conv7',kernel_initializer='he_normal',data_format="channels_last")(cnn_3)
    cnn_4= BatchNormalization()(cnn_4)
    cnn_4= Activation('relu')(cnn_4)
    #dc layer
    tf_fft=Lambda(lambda v: tf.spectral.fft2d(tf.cast(tf.transpose((v),perm=[0,2,1,3]),dtype=tf.complex64)))(cnn_4)
    real = Lambda(tf.real)(tf_fft)
    imag = Lambda(tf.imag)(tf_fft)
    dc_int_real=tf.keras.layers.multiply([inv_mask,real])
    orig_ksp_real =Lambda(tf.real)(input_ksp)
    print(dc_int_real)
    print(orig_ksp_real)
    dc_layer=Lambda(lambda a:a[0] + a[1])([dc_int_real,orig_ksp_real])
    model = Model(inputs=[input_img,inv_mask,input_ksp], outputs=dc_layer)
    return model


    


# In[3]:


model=basic_model();
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
#model.fit([x_train,inv_mask,x_kspace_train], y_train,batch_size=1,epochs=10,steps_per_epoch=1)


# In[ ]:




