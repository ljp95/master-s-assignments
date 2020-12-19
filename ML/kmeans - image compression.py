# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:38:22 2019

@author: Jp
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time

#Init image
im = plt.imread("1.jpg")
im_h,im_l,_= im.shape
pixels = im.reshape((im_h*im_l,3))
imnew = pixels.reshape((im_h,im_l,3))
plt.imshow(im)

pixels = pixels.astype(dtype='double')
k =5
n = len(pixels)

def init(pixels):
    indices = random.sample(list(range(n)),k)
    barys = np.zeros((k,3))
    for i in range(len(indices)):
        barys[i] = pixels[indices[i]]
    return barys

def affectation(pixels,barys):
    mat = np.zeros((k,n))
    for i in range(k):
        mat[i] = np.sum((barys[i] - pixels)**2,axis=1)
    Ck = np.argmin(mat,axis=0)
    return Ck
    
def final(pixels,barys,Ck):
    pixels2 = np.zeros((n,3),dtype='uint8')
    barys = barys.astype(dtype='uint8')
    for i in range(n):
        pixels2[i] = barys[Ck[i]]
    return pixels2.reshape((im_h,im_l,3))        

def mise_a_jour(pixels,barys,Ck):
    for i in range(k):
        barys[i] = np.mean(pixels[Ck==i],axis=0)
    return barys

def f_cout(pixels,barys,Ck):
    cout = 0 
    for i in range(k):
        cout += np.sum((pixels[Ck==i]-barys[i])**2)
    return cout
    
def kmeans(pixels,seuil=None,max_iter=None):
    barys = init(pixels)
    Ck = affectation(pixels,barys)
#    tmp = f_cout(pixels,barys,Ck)
#    cout = [tmp+seuil+1,tmp]
    cpt = 0
#    plt.figure()
#    img = final(pixels,barys,Ck)
#    plt.imshow(img)
#    while(cout[-2]-cout[-1]>seuil):
    while(cpt<max_iter):
        debut = time.time()
        barys = mise_a_jour(pixels,barys,Ck)
        Ck = affectation(pixels,barys)
#        cout.append(f_cout(pixels,barys,Ck))
#        plt.figure()
#        img = final(pixels,barys,Ck)
#        plt.imshow(img)
        cpt+=1
        print('iteration {} en {} secs'.format(cpt,time.time()-debut))
    img = final(pixels,barys,Ck)
    return img,barys,Ck
    
img,barys,Ck = kmeans(pixels,max_iter=5)
fig = plt.figure()
plt.imshow(img)
fig.savefig('kmeans.jpg')


