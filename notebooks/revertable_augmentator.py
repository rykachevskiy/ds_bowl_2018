
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
    for i in range(1,3):
        layer = np.zeros((int(img.shape[0] * ratios[0]), int(img.shape[1] * ratios[1])))
        
        _, cnts_base, hierarchy = cv2.findContours(1 - img[:,:,i].copy().astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        new_cnts = []
        for c in cnts_base:
            c = np.round_(c * np.array(ratios[::-1]).reshape((1,1,2))).astype(int)
            new_cnts.append(c)
        
        cv2.drawContours(layer,new_cnts,-1,1,-1)
        resized_img[:,:,i] = 1 - layer.astype(int)
      
    for i in [0,-2,-1]:
        main_layer = cv2.resize(img[:,:,i], (resized_img.shape[1], resized_img.shape[0]),interpolation=cv2.INTER_NEAREST)
        resized_img[:,:,i] = main_layer
   
    return resized_img

def affine_smooth(im, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):       
    new_im = np.zeros(im.shape)
    #center_shift = np.array((im.shape[0], im.shape[1])) / 2
    center_shift = np.zeros(2)
    mat = np.array([[zoom * np.cos(np.deg2rad(-rotation)),- zoom * np.sin(np.deg2rad(-rotation-shear)), 0], [zoom * np.sin(np.deg2rad(-rotation)),  zoom  * np.cos(np.deg2rad(-rotation-shear)), 0]])    

    for i in range(1,3):
        layer = np.zeros((im.shape[0], im.shape[1]))
        _, cnts_base, hierarchy = cv2.findContours(1 - im[:,:,i].copy().astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_cnts = []
        for c in cnts_base:
            new_c = np.zeros_like(c)
            

#            for j, point in enumerate(c):
#                #[77 42]
#                new_c[j, 0, 0] = zoom * (point[0, 0] - center_shift[0]) * np.cos(np.deg2rad(-rotation)) - zoom * (point[0, 1] - center_shift[1]) * np.sin(np.deg2rad(-rotation+shear)) + center_shift[0]
#                new_c[j, 0, 1] = zoom * (point[0, 0] - center_shift[0]) * np.sin(np.deg2rad(-rotation)) + zoom * (point[0, 1] - center_shift[1]) * np.cos(np.deg2rad(-rotation+shear)) + center_shift[1]

            new_c[:, 0, 0] = mat[0,0] * (c[:, 0, 0] - center_shift[0])  + mat[0,1] * (c[:, 0, 1] - center_shift[1]) + center_shift[0]
            new_c[:, 0, 1] = mat[1,0] * (c[:, 0, 0] - center_shift[0])  + mat[1,1] * (c[:, 0, 1] - center_shift[1]) + center_shift[1]

            c = np.round_(new_c).astype(int)
            #print("cn\n", c)
            new_cnts.append(c)
        
        cv2.drawContours(layer,new_cnts,-1,1,-1)
        new_im[:,:,i] = 1 - layer.astype(int)
    
    for i in [0,-2,-1]:
        main_layer = cv2.warpAffine(im[:,:,i].copy(), mat, im[:,:,0].shape[::-1])
        new_im[:,:,i] = main_layer
    return new_im#, mat


def generate_affine(im, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    center_shift = np.array((im.shape[0], im.shape[1])) / 2
    tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    #tform = tform_center + tform_augment + tform_uncenter
    tform = tform_augment
    return tform

def affine(im, tform): 
    new_im = skimage.transform.warp(im, tform, order = 1,preserve_range=True)
    
    new_im[:,:,0] = new_im[:,:,0]
    circle3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    for i in range(2, im.shape[2]):
        new_im[:,:,i] = (cv2.morphologyEx(new_im[:,:,i], cv2.MORPH_CLOSE, circle3) > 0).astype(int)
        #new_im[:,:,i] = (cv2.morphologyEx(new_im[:,:,i], cv2.MORPH_CLOSE, np.ones((3,3))) > 0).astype(int)

    return new_im


# In[43]:



class Transformation: 
    def __init__(self,
                 im,
                inv=0,
                gamma=1,
                log=0,
                sigmoid=0,
                rotation=0,
                shear=0,
                ratios=np.array([1,1]),
                vertical_flip=0,
                horizontal_flip=0):
        self.inv = inv
        self.gamma = gamma
        self.log = log
        self.sigmoid = sigmoid
        self.rotation = rotation=rotation
        self.shear=shear
        self.ratios = ratios
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
    
    def generate_random_transformation(im):
    	return Transformation(im,
            np.random.randint(0,2),
            np.random.random() + 0.5,                  
    		np.random.randint(0,2),
    		np.random.randint(0,2),
        	np.random.randint(-20,20),
        	np.random.randint(-20,20),
        	(np.random.random(2) + 4) / 4.5,
        	np.random.randint(0,2),
        	np.random.randint(0,2))

        
    def generate_geometrical(im):
        return Transformation(im,
            0,
            1,
        	0,
    		0,
        	np.random.randint(-20,20),
        	np.random.randint(-20,20),
        	(np.random.random(2) + 4) / 4.5,
        	np.random.randint(0,2),
        	np.random.randint(0,2))
        
    def generate_mirror(im):
        return Transformation(im,
            0,
            1,
        	0,
    		0,
        	0,
        	0,
        	np.ones(2),
        	np.random.randint(0,2),
        	np.random.randint(0,2))
    
    def apply(self, im, normalize=True):
        ## BE CAREFUL APPLY ONLY ON THE SAME IMAGE
        
        im_c = im.copy()
        if self.inv:
            im_c[:,:,0] = 1 - im_c[:,:,0]
        if self.gamma:
            im_c[:,:,0] = skimage.exposure.adjust_gamma(im_c[:,:,0], )
        if self.log:
            im_c[:,:,0] = skimage.exposure.adjust_log(im_c[:,:,0])
        if self.sigmoid:
            im_c[:,:,0] = skimage.exposure.adjust_sigmoid(im_c[:,:,0])
            
        
        if not(self.rotation == 0 and self.shear == 0):
            im_c = affine_smooth(im_c, zoom=1.0, rotation=self.rotation, shear=self.shear, translation=(0, 0))
        if self.ratios[0] != 1 or self.ratios[1] != 1:
            im_c = resize_smooth(im_c, self.ratios)
        
        if self.vertical_flip:
            im_c = im_c[::-1,:,:]
                
        if self.horizontal_flip:
            im_c = im_c[:,::-1,:]
            
        if normalize:
            im_c[:,:,0] -= im_c[:,:,0].min()
            im_c[:,:,0] /= im_c[:,:,0].max()
        return im_c
    
    def apply_inverse(self, im, normalize=True):
        ##TODO REVERSE AFFINE!!!
        
        if self.vertical_flip:
            im = im[::-1,:,:]
                
        if self.horizontal_flip:
            im = im[:,::-1,:]
            
        im = resize_smooth(im, 1 / self.ratios)
        im = affine_smooth(im, self.affine.inverse)
        
        if normalize:
            im[:,:,0] -= im[:,:,0].min()
            im[:,:,0] /= im[:,:,0].max()
        return im
    

    

