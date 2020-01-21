#!/usr/bin/env python
# coding: utf-8

# In[6]:


import h5py
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from utils import compressed_sensing as cs
import tensorflow as tf
import random

#reading images
img1=sitk.ReadImage("F:/data sets/training set/f1.nii")
img2=sitk.ReadImage("F:/data sets/training set/f2.nii")
img3=sitk.ReadImage("F:/data sets/training set/f3.nii")
img4=sitk.ReadImage("F:/data sets/training set/m1.nii")
img5=sitk.ReadImage("F:/data sets/training set/m2.nii")
img6=sitk.ReadImage("F:/data sets/training set/m3.nii")

#creating array
arr1 = sitk.GetArrayFromImage(img1)
arr2 = sitk.GetArrayFromImage(img2)
arr3 = sitk.GetArrayFromImage(img3)
arr4 = sitk.GetArrayFromImage(img4)
arr5 = sitk.GetArrayFromImage(img5)
arr6 = sitk.GetArrayFromImage(img6)
#combining array
arr=np.concatenate((arr1,arr2,arr3,arr4,arr5,arr6),axis=0)
np.save('image_array',arr)


#fft to numpy array
fft_arr=np.fft.fft2(arr)
fshift_out = np.fft.fftshift(fft_arr)

#cartesian mask
mask_1=cs.cartesian_mask(arr.shape, 4, sample_n=10, centred=False)
im_und, k_und = cs.undersample(arr,mask_1, centred=False, norm='ortho')

#normalising the input and output data
features_IMG=tf.keras.utils.normalize(im_und,axis=1)
x_train=np.expand_dims(features_IMG, 3)
print(np.shape(x_train))
features_KS=tf.keras.utils.normalize(k_und,axis=1)
x_ks_train=np.expand_dims(features_KS, 3)
Target=tf.keras.utils.normalize(arr,axis=1)
y_train=np.expand_dims(Target, 3)

mask_data=np.expand_dims(mask_1[1], 3)
mask_ksp=mask_data.astype(int)
inv_mask=np.bitwise_not(mask_ksp)
print(np.shape(mask_data))
#storing data
import pickle
pickle_out=open("x_train.pickle","wb")
pickle.dump(x_train,pickle_out)
pickle_out.close()

pickle_out=open("x_ks_train.pickle","wb")
pickle.dump(x_ks_train,pickle_out)
pickle_out.close()

pickle_out=open("mask_ksp.pickle","wb")
pickle.dump(mask_ksp,pickle_out)
pickle_out.close()

pickle_out=open("y_train.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()


# In[3]:


print(mask_ksp[20,:,:,0])
print(inv_mask[20,:,:,:])
print(np.shape(inv_mask[20,:,:,:]))
plt.imshow(inv_mask[20,:,:,0],cmap='gray')
plt.colorbar()
plt.show()

