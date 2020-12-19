#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:23:50 2018

@author: said
"""


#%% SECTION 1 inclusion de packages externes 


import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio


# POUR LA MORPHO
import skimage.morphology as morpho  
import skimage.feature as skf
from scipy import ndimage as ndi

#%% SECTION 2 fonctions utiles pour le TP

def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    '''if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase=' ' 
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase= ' &'
    '''
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M
    
    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    plt.imshow(imt,cmap='gray')
    '''nomfichier=tempfile.mktemp('TPIMA.png')
    commande=prephrase +nomfichier+endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)'''

def viewimage_color(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI(defaut 0) et MAXI (defaut 255) seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    '''
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase= ' '
    else: #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase=' &'
        '''
    
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M
    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    plt.imshow(imt,cmap='gray')

    '''
    nomfichier=tempfile.mktemp('TPIMA.pgm')
    commande=prephrase +nomfichier+endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)
    '''


def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """

    if forme == 'diamond':
        return morpho.selem.diamond(taille)
    if forme == 'disk':
        return morpho.selem.disk(taille)
    if forme == 'square':
        return morpho.selem.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=np.float32(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x**2+y**2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc=morpho.selem.draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')

            

def couleurs_alea(im):
    """ 
    Donne des couleurs aleatoires a une image en niveau de gris.
    Cette fonction est utile lorsque le niveua de gris d'interprete comme un numero
      de region. Ou encore pour voir les leger degrades d'une teinte de gris.
      """
    sh=im.shape
    out=np.zeros((*sh,3),dtype=np.uint8)
    nbcoul=np.int32(im.max())
    tabcoul=np.random.randint(0,256,size=(nbcoul+1,3))
    tabcoul[0,:]=0
    for k in range(sh[0]):
        for l in range(sh[1]):
            out[k,l,:]=tabcoul[im[k,l]]
    return out
    

#%% SECTION 3 exemples de commandes pour effectuer ce qui est demande pendant le TP

im=skio.imread('retina2.gif')
viewimage(im)
se=strel('disk',3)
# dilatation
dil4=morpho.dilation(im,se)
viewimage(dil4)
# erosion
ero4=morpho.erosion(im,se)
viewimage(ero4)
#ouverture
open4=morpho.opening(im,se)
viewimage(open4)
#fermeture
close4=morpho.closing(im,se)
viewimage(close4)

#propriétés
im=skio.imread('retina2.gif')
viewimage(im)
se=strel('disk',5)

im2 = np.float32(im)
n,m=im2.shape

mask = (np.random.choice(100,n*m)-50).reshape(n,m)
im2= im2-mask
im2[im2<0]=0
viewimage(im2)

plt.imshow(im-im2);plt.colorbar()

cl1 = morpho.dilation(im,se)
viewimage(cl1)
cl2 = morpho.dilation(im2,se)
viewimage(cl2)

cl3 = np.minimum(cl1,cl2)
viewimage(cl3)
cl4 = morpho.dilation(np.minimum(im,im2))
viewimage(np.minimum(im,im2))
viewimage(cl4)
plt.imshow(cl3-cl4);plt.colorbar()

plt.imshow(cl1-cl2);plt.colorbar()
#iteration de dilatation
se3=strel('square',3)
dil3 = morpho.dilation(im,se3)
se5=strel('square',5)
dil35 = morpho.dilation(dil3,se5)
se7=strel('square',7)
dil7 = morpho.dilation(im,se7)

viewimage(dil3)
viewimage(dil35)
viewimage(dil7)

#iteration d'ouverture
se3=strel('square',3)
op3 = morpho.opening(im,se3)
viewimage(op3)

se5=strel('square',5)
op5 = morpho.opening(im,se5)
viewimage(op5)

op35 = morpho.opening(op3,se5)
viewimage(op35)

op53 = morpho.opening(op5,se3)
viewimage(op53)

se8=strel('square',8)
op8 = morpho.opening(im,se8)
viewimage(op8)

#%% Chapeau haut-de-forme
im=skio.imread('retina2.gif')
viewimage(im)
t=10
se1=strel('square',5,45)
ch1=im-morpho.opening(im,se1)
viewimage(ch1)

se2 = strel('disk',2)
ch2=im-morpho.opening(im,se2)
viewimage(ch2)

#dual top hat
im=skio.imread('laiton.bmp')
viewimage(im)
t=10
se1=strel('disk',t)
ch3 = morpho.closing(im,se1)-im
viewimage(ch3)
ch4 = im-morpho.opening(im,se1)
viewimage(ch4)


se1=strel('line',t,45)
ch1=morpho.opening(im,se1)
viewimage(ch1)
se2=strel('line',t,-45)
ch2=morpho.opening(im,se2)
viewimage(ch2)
se3=strel('line',t,-90)
ch3=morpho.opening(im,se3)
viewimage(ch3)

se4 = se1+se2
ch4 = morpho.opening(im,se4)
viewimage(ch4)
final = np.maximum(ch1,ch2,ch3)
viewimage(final)

#%%  Filtre alterne
im=skio.imread('retina2.gif')
imt=im.copy()
N=50
viewimage(im)

for k in range(N,75):
    se=strel('disk',k,45)
    imt=morpho.closing(morpho.opening(imt,se),se)
    plt.figure()
    viewimage(imt)
    
#%% reconstruction
im=skio.imread('retina2.gif')
se4=strel('line',8,45)
open4=morpho.opening(im,se4)
viewimage(open4)
reco=morpho.reconstruction(open4,im)
viewimage(reco)

#%%  Reconstruction fas
im=skio.imread('retina2.gif')
imt=im.copy()
N=8
viewimage(im)

for k in range(1,N):
    se=strel('disk',k,45)
    op = morpho.opening(imt,se)
    imt = morpho.reconstruction(op,imt)
    plt.figure()
    viewimage(imt)
    clo =morpho.closing(imt,se)
    imt = morpho.reconstruction(imt,clo)
    plt.figure()
    viewimage(imt)

#%% grad et partage des eaux
im = skio.imread('bat200.bmp')
viewimage(im)
se = strel('disk',1,0)
se=morpho.selem.disk(1)

grad=morpho.dilation(im,se)-morpho.erosion(im,se)
grad = morpho.closing(grad,se)
grad=np.int32(grad>10)*grad
viewimage(grad)

local_mini = skf.peak_local_max(255-grad, #il n'y a pas de fonction local_min...
                            indices=False)
markers = ndi.label(local_mini)[0]
labels = morpho.watershed(grad, markers,watershed_line=True)
viewimage_color(couleurs_alea(labels))
imt = labels==0
viewimage(imt)

plt.figure()
imt2 = skio.imread('bat200.bmp')
imt2[imt==True] = 255
viewimage(imt2)

grad2 = np.maximum(grad,markers)
im2 = morpho.reconstruction(grad,markers)
grad=im2

im = skio.imread('bulles.bmp')
se=morpho.selem.disk(1)
viewimage(255*im)

grad=morpho.dilation(im,se)-morpho.erosion(im,se)

#%% FIN  exemples TP MORPHO
markers = 255*np.ones(im.shape)
markers[:,0]=0;markers[:,-1]=0;markers[0,:]=0;markers[-1,:]=0
im2=morpho.opening(im,strel("disk",1))
viewimage(im2)
grad=morpho.dilation(im2,se)-morpho.erosion(im2,se)
