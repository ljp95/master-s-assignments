# -*- coding: utf-8 -*

import math
import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage as ndi
import scipy.misc

import imageio

#########################################################
#%%


plt.close('all')

#im_obs=scipy.misc.imread('Iobservee.png')
im_obs=imageio.imread('Iobservee.png')
plt.figure()
plt.imshow(im_obs)
plt.set_cmap('gray')

#
# L'image 'IoriginaleBW.png' est codée sur 1 bit. 
#
# CAS PYTHON 2..3 et antérieur 
# scipy.misc.imread n'aime pas ce cas (du moins en python 2.7) 
# On utilise alors l'option flatten=1 qui renvoie une image en float
#  (en fait cette option est prévue pour passer en "gray" une image couleur)
# im_ori=np.uint8(scipy.misc.imread('IoriginaleBW.png',flatten=1))

#im_ori=scipy.misc.imread('IoriginaleBW.png')
im_ori=imageio.imread('IoriginaleBW.png')
plt.figure()
plt.imshow(im_ori)
plt.set_cmap('gray')
plt.show()

#%%
#etude des distributions des deux classes 
#affichage de l'histogramme de l'image 
import echan
plt.figure()
plt.hist(im_obs.ravel(),range=[0,np.max(im_obs)],bins=np.max(im_obs))
plt.show()
#%%


#selectionner une imagette dans une région sombre de l'image (classe 0)
#attention les ordonnées correspondent aux lignes et les abscisses aux colonnes
crop_classe0=im_obs[44:120,173:243]

#tracer son histogramme
plt.figure()
plt.hist(crop_classe0.ravel(),range=[0,np.max(crop_classe0)],bins=np.max(crop_classe0))

#calculer sa moyenne et sa variance
#
#  Par défaut, np.mean, np.zzz prend les deux axes de l'image. 
#   Il faut forcer l'option si l'on veut faire la moyenne en ligne ou en colonne
m0=np.mean(crop_classe0)
var0=np.var(crop_classe0)

#sélectionner une imagette dans une région claire de l'image (classe 1)
crop_classe1=im_obs[228:310,123:208]
#tracer son histogramme
plt.figure()
plt.hist(crop_classe1.ravel(),range=[0,np.max(crop_classe1)],bins=np.max(crop_classe1))
#calculer sa moyenne et sa variance 
m1=np.mean(crop_classe1)
var1=np.var(crop_classe1)
#%%
#definir le seuil pour faire une classification au sens du MV
seuil = (m0+m1)/2
im_bin=im_obs.copy()
im_mask=im_obs>seuil
im_bin[~im_mask]=0
im_bin[im_mask]=1
im_seuil=im_bin.copy()
plt.imshow(im_seuil);  
plt.title("im_seuil");
plt.show()  

#definir la valeur de beta_reg pour avoir une "bonne" regularisation
beta_reg = 3*var0

#autres initialisations
im_bin = np.zeros(im_bin.shape,dtype = 'uint8')
im_bin[0,0] = 0
#im_bin = np.random.randint(0,2,size=im_bin.shape)
plt.imshow(im_bin,cmap='gray')
plt.title("initialisation im_bin")
plt.show()
#test=plt.waitforbuttonpress()


#Iterations
for n in range(10):  
    echan.iter_icm(im_bin,im_obs,beta_reg,m0,m1) 
    #echan.iter_SA(im_bin,im_obs,beta_reg,m0,m1,100) 
    plt.imshow(im_bin);
    plt.title("icm iteration {}".format(n+1))        
    #mafigure.canvas.draw()
    plt.show(block=False)
    #test=plt.waitforbuttonpress()

plt.figure()
plt.imshow(im_bin);
plt.show()
#%%

