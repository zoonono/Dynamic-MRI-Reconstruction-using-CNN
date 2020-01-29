#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Conv2D,BatchNormalization,Activation,Flatten,Lambda
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import compressed_sensing as cs
import random
from dltk.io.preprocessing import *
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
#loading data
x_train=pickle.load(open("x_train.pickle","rb"))
x_kspace_train=pickle.load(open("x_ks_train.pickle","rb"))
print(x_train.dtype)
mask=pickle.load(open("mask_data.pickle","rb"))
y_train=pickle.load(open("y_train.pickle","rb"))
mask_inv=1-mask
inv_mask=tf.convert_to_tensor(mask_inv,dtype='complex64',name='inverse_mask')
print(x_train.dtype)
print(x_kspace_train.dtype)
print(y_train.dtype)
print(np.shape(mask_inv))


X_train_inp,X_test_inp,X_train_ksp,X_test_ksp, y_train_img, y_test_img = train_test_split(x_train,x_kspace_train, y_train, test_size=0.2, random_state=0)


'''''
plt.imshow(np.abs(x_train[20,:,:,0]),cmap='gray')
plt.colorbar()
plt.show()
plt.imshow((np.abs(x_kspace_train[20,:,:,0])),cmap='gray')
plt.colorbar()
plt.show()
plt.imshow(y_train[20,:,:,0],cmap='gray')
plt.colorbar()
plt.show()
plt.imshow(mask_inv[0,:,:,0],cmap='gray')
plt.colorbar()
plt.show()

plt.imshow(np.squeeze(mask),cmap='gray')
plt.colorbar()
plt.show()
'''''


# In[2]:


def mean_square_error(a,b):
    mse=tf.keras.losses.mean_squared_error(a,b)
    return mse
    


# In[28]:


make_complex_array=lambda a:a[:,:,:,0]+1j*(a[:,:,:,1])
make_real_tensor=lambda s:tf.math.real(s)
make_complex_tensor=lambda d:tf.math.imag(d)
expand_dimns=lambda a:tf.concat((tf.expand_dims(make_real_tensor(a),axis=3),tf.expand_dims(make_complex_tensor(a),axis=3)),axis=3)


# In[5]:


def dc_layer(x,y,z):
    ex_img=make_complex_array(tf.cast(x,dtype=tf.complex64))
    ex_ksp=make_complex_array(tf.cast(z,dtype=tf.complex64))
    fft_2d=tf.signal.fft2d(ex_img)
    msk_ksp_mul=tf.math.multiply(fft_2d,y)
    ksp_add=tf.math.add(msk_ksp_mul,ex_ksp)
    ifft_2d=tf.signal.ifft2d(ksp_add)
    dc_op=expand_dimns(ifft_2d)
    return dc_op
    


# In[ ]:


datagen =tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    

def basic_model():
    input_img=Input(shape=(256,256,2), name='input_image')
    input_ksp=Input(shape=(256,256,2), name='input_ksp')
 
    cnn_11= Conv2D(64, (3, 3), padding='same', name='conv11', kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(input_img)
    cnn_11= Activation('tanh')(cnn_11)
    
    cnn_12= Conv2D(64, (3, 3), padding='same',name='conv12',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_11)
    cnn_12= Activation('tanh')(cnn_12)
    
    cnn_13=Conv2D(64, (3, 3), padding='same', name='conv13',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_12)
    cnn_13= Activation('tanh')(cnn_13)
    
    cnn_14=Conv2D(64, (3, 3), padding='same', name='conv14',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_13)
    cnn_14= Activation('tanh')(cnn_14)
    
    cnn_15=Conv2D(2, (3, 3), padding='same', name='conv15',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.00000011),data_format="channels_last")(cnn_14)
    cnn_15= Activation('tanh')(cnn_15)
    
    #residual connection
    cnn_op1=Lambda(lambda b:b[0] + b[1], name='residual_1')([input_img,cnn_15])
    #dc layer
    dc_conc_op=Lambda(lambda z:dc_layer(z[0],z[1],z[2]),name='dc_layer1')([cnn_op1,inv_mask,input_ksp])

    
    cnn_21= Conv2D(64, (3, 3), padding='same', name='conv21', kernel_initializer='he_normal', kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(dc_conc_op)
    cnn_21= Activation('tanh')(cnn_21)
    
    cnn_22= Conv2D(64, (3, 3), padding='same',name='conv22',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_21)
    cnn_22= Activation('tanh')(cnn_22)
    
    cnn_23=Conv2D(64, (3, 3), padding='same', name='conv23',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_22)
    cnn_23= Activation('tanh')(cnn_23)
    
    cnn_24=Conv2D(64, (3, 3), padding='same', name='conv24',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_23)
    cnn_24= Activation('tanh')(cnn_24)
    
    cnn_25=Conv2D(2, (3, 3), padding='same', name='conv25',kernel_initializer='he_normal',kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),data_format="channels_last")(cnn_24)
    cnn_25= Activation('tanh')(cnn_25)
    cnn_op2=Lambda(lambda x:x[0] + x[1], name='residual_2')([cnn_25,dc_conc_op])
    
    dc_conc_op1=Lambda(lambda z:dc_layer(z[0],z[1],z[2]),name='dc_layer2')([cnn_op2,inv_mask,input_ksp])


    model = Model(inputs=[input_img,input_ksp], outputs=dc_conc_op1)
    return model


    


# In[43]:



model=basic_model();
model.summary()
optimiser=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=optimiser,
              loss='mean_squared_error',
              metrics=['accuracy',mean_square_error])


# In[ ]:


H =model.fit([X_train_inp,X_train_ksp],y_train_img,batch_size=1,epochs=150)

x=model.predict([X_test_inp,X_test_ksp], batch_size=None)
x_op=make_complex_array(x)
plt.imshow(np.abs(x_op[5,:,:]),cmap='gray')
plt.colorbar()
plt.show()
plt.plot(H.history['loss'])
plt.show()





