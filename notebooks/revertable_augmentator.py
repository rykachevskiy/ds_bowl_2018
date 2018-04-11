
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

import skimage.exposure 
import skimage.transform 



def resize_smooth(img, ratios):       
    resized_img = np.zeros((int(img.shape[0] * ratios[0]), int(img.shape[1] * ratios[1]), img.shape[2]))
    for i in range(1,img.shape[2]):
        layer = np.zeros((int(img.shape[0] * ratios[0]), int(img.shape[1] * ratios[1])))
        _, cnts_base, hierarchy = cv2.findContours(img[:,:,i].copy().astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        new_cnts = []
        for c in cnts_base:
            c = np.round_(c * np.array(ratios[::-1]).reshape((1,1,2))).astype(int)
            new_cnts.append(c)
        
        cv2.drawContours(layer,new_cnts,-1,1,-1)
        resized_img[:,:,i] = layer.astype(int)
      
    main_layer = cv2.resize(img[:,:,0], (resized_img.shape[1], resized_img.shape[0]),interpolation=cv2.INTER_NEAREST)
    resized_img[:,:,0] = main_layer
    return resized_img


# In[5]:

# In[116]:


def generate_affine(im, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    center_shift = np.array((im.shape[0], im.shape[1])) / 2
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    tform = tform_center + tform_augment + tform_uncenter

    return tform

def affine(im, tform): 
    new_im = skimage.transform.warp(im, tform, preserve_range=True)
    
    new_im[:,:,0] = new_im[:,:,0]
    for i in range(1, im.shape[2]):
        new_im[:,:,i] = (cv2.morphologyEx(new_im[:,:,i], cv2.MORPH_CLOSE, np.ones((3,3))) > 0).astype(int)
        #new_im[:,:,i] = (cv2.morphologyEx(new_im[:,:,i], cv2.MORPH_CLOSE, np.ones((3,3))) > 0).astype(int)

    return new_im


# In[43]:



class Transformation: 
    def __init__(self,
                 im,
                log=0,
                sigmoid=0,
                rotation=0,
                shear=0,
                ratios=np.array([1,1]),
                vertical_flip=0,
                horizontal_flip=0):
        self.log = log
        self.sigmoid = sigmoid
        self.affine = generate_affine(im, rotation=rotation,shear=shear)
        self.ratios = ratios
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
    
    def generate_random_transformation(im):
    	return Transformation(im.
    		np.random.randint(0,2),
    		np.random.randint(0,2),
        	np.random.randint(-20,20),
        	np.random.randint(-20,20),
        	(np.random.random(2) + 1) / 2,
        	np.random.randint(0,2),
        	np.random.randint(0,2))

        
    def generate_geometrical(im):
        return Transformation(im,
        	0,
    		0,
        	np.random.randint(-20,20),
        	np.random.randint(-20,20),
        	(np.random.random(2) + 4) / 4.5,
        	np.random.randint(0,2),
        	np.random.randint(0,2))
        
    def apply(self, im, normalize=True):
        ## BE CAREFUL APPLY ONLY ON THE SAME IMAGE
#         if self.log:
#             im[:,:,0] = skimage.exposure.adjust_log(im[:,:,0])
#         if self.sigmoid:
#             im[:,:,0] = skimage.exposure.adjust_sigmoid(im[:,:,0])
            
        im = affine(im, self.affine)
        im = resize_smooth(im, self.ratios)
        
        if self.vertical_flip:
            im = im[::-1,:,:]
                
        if self.horizontal_flip:
            im = im[:,::-1,:]
            
        if normalize:
            im[:,:,0] -= im[:,:,0].min()
            im[:,:,0] /= im[:,:,0].max()
        return im
    
    def apply_inverse(self, im, normalize=True):
        ##TODO apply log and sigmoid
        
        if self.vertical_flip:
            im = im[::-1,:,:]
                
        if self.horizontal_flip:
            im = im[:,::-1,:]
            
        im = resize_smooth(im, 1 / self.ratios)
        im = affine(im, self.affine.inverse)
        
        if normalize:
            im[:,:,0] -= im[:,:,0].min()
            im[:,:,0] /= im[:,:,0].max()
        return im
    

    

