
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from tools import *

def compute_grad(I):
    ha,hb = np.array([-1,0,1]),np.array([1,2,1])
    Ix = conv_separable(I,ha,hb)/4
    Iy = conv_separable(I,hb,ha)/4
    return Ix, Iy

def compute_grad_mod_ori(I):
    Ix,Iy = compute_grad(I)
    Gm = np.sqrt(Ix**2+Iy**2)
    Go = compute_grad_ori(Ix,Iy,Gm)
    return Gm, Go

def compute_sift_region(Gm, Go, mask=None):
    # mask 
    if mask is not None:
        Gm = Gm*mask
    sift = []
    for i in range(4):
        x = i*4
        for j in range(4):
            region = np.zeros(8)
            y = j*4
            for ii in range(4):
                xx = x+ii 
                for jj in range(4):
                    yy = y+jj
                    region[Go[xx,yy]] += Gm[xx,yy] 
            sift += region.tolist()
    sift = np.array(sift)
    
    # post processing
    norm = np.sqrt(np.sum(sift**2))
    if norm<0.5:
        sift = np.zeros(128)
        return sift
    else:
        sift = sift/norm
        sift[sift>0.2] = 0.2
        norm = np.sqrt(np.sum(sift**2))
        sift = sift/norm
    return sift

def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    # TODO calculs communs aux patchs
    sifts = np.zeros((len(x), len(y), 128))
    mask = gaussian_mask()
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            patch = im[xi:xi+16,yj:yj+16]
            Gm,Go = compute_grad_mod_ori(patch)
            sifts[i, j, :] = compute_sift_region(Gm,Go,mask=mask) 
    return sifts
