#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% SECTION 0 inclusion de packages externes 

from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import matplotlib.pyplot as plt

from nt_toolbox.general import *
from nt_toolbox.signal import *
#%pylab inline
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

from numpy import *
from matplotlib.pyplot import *

#%% SECTION 1a - Définition d'une courbe paramétrique

gamma0 = np.array([.78, .14, .42, .18, .32, .16, .75, .83, .57, .68, .46, .40, .72, .79, .91, .90]) + 1j*np.array([.87, .82, .75, .63, .34, .17, .08, .46, .50, .25, .27, .57, .73, .57, .75, .79])

periodize = lambda gamma: concatenate((gamma, [gamma[0]]))
def cplot(gamma,s='b',lw=1): 
    plot(imag(periodize(gamma)), real(periodize(gamma)), s, linewidth=lw)
    axis('equal')
    axis('off')
cplot(gamma0,'b.-');

# ré-échantillonage de la courbe pour un nombre de points p

p = 256
interpc = lambda x,xf,yf: interp(x,xf,real(yf)) + 1j * interp(x,xf,imag(yf))
curvabs = lambda gamma: concatenate( ([0], cumsum( 1e-5 + abs(gamma[:-1:]-gamma[1::]) ) ) )
resample1 = lambda gamma,d: interpc(arange(0,p)/float(p),  d/d[-1],gamma)
resample = lambda gamma: resample1( periodize(gamma), curvabs(periodize(gamma)) )
gamma1 = resample(gamma0)
cplot(gamma1, 'k')

# Calcul de dérivées par différences finies
shiftR = lambda c: concatenate( ([c[-1]],c[:-1:]) )
shiftL = lambda c: concatenate( (c[1::],[c[0]]) )
BwdDiff = lambda c: c - shiftR(c)
FwdDiff = lambda c: shiftL(c) - c

# tangente et normale
normalize = lambda v: v/maximum(abs(v),1e-10)
tangent = lambda gamma: normalize( FwdDiff(gamma) )
normal = lambda gamma: -1j*tangent(gamma)

#%% SECTION 1b - Déplacement de la courbe le long de la normale d'un pas delta

delta = .01
gamma2 = gamma1 + delta * normal(gamma1)
gamma3 = gamma1 - delta * normal(gamma1)

cplot(gamma1, 'k')
cplot(gamma2, 'r--')
cplot(gamma3, 'b--')
#axis('tight') 
#axis('off')

#%% SECTION 1c - Evolution de la courbe en fonction de la courbure

normalC = lambda gamma: BwdDiff(tangent(gamma)) / abs( FwdDiff(gamma) )

# paramètres de l'évolution
dt = 0.0005 / 100
Tmax = 3.8 / 100
niter = int(Tmax/dt)
gamma = gamma1
cplot(gamma,'b')

# Evolution de la courbe
for i in range(1,niter+1):
    gamma = resample(gamma + dt * normalC(gamma))
    if (i == 100) :#or ((i % 300) == 0) or (i == niter):
        cplot(gamma, 'r')
#axis('tight');  axis('off')

#%% SECTION 1d - Segmentation par contours actifs géodésiques

n = 256
name = 'brain.bmp'
f = load_image(name, n)
imageplot(f)

# initialisation par un cercle
r = .98*n/2 # radius
p = 128 # number of points on the curve
theta = transpose( linspace(0, 2*pi, p + 1) )
theta = theta[0:-1]
gamma0 = n/2 * (1 + 1j) +  r*(cos(theta) + 1j*sin(theta))
gamma = gamma0
cplot(gamma,'r')

# calcul de la métrique : fonction décroissante du gradient lissé
G = grad(f)
d0 = sqrt(sum(G**2, 2))
imageplot(d0)

a = 2
d = gaussian_blur(d0, a)
imageplot(d)

d = minimum(d, .4)
W = rescale(-d, .8, 1)
imageplot(W)

# paramètres de l'évolution
dt = 1
Tmax = 20000
niter = int(Tmax/ dt)

# Evolution de la courbe
G = grad(W)
G = G[:,:,0] + 1j*G[:,:,1]
imageplot(abs(G))
EvalG = lambda gamma: bilinear_interpolate(G, imag(gamma), real(gamma))
EvalW = lambda gamma: bilinear_interpolate(W, imag(gamma), real(gamma))
gamma = gamma0

imageplot(f)
cplot(gamma, 'b')
dotp = lambda c1,c2: real(c1)*real(c2) + imag(c1)*imag(c2)
for i in range(1, niter+1):
    N = normal(gamma)
    g = EvalW(gamma) * normalC(gamma) - dotp(EvalG(gamma), N) * N
    gamma = resample(gamma + dt*g)
    if (i == 1) or ((i % 500) == 0) or (i == niter):
        cplot(gamma, 'r')
#axis('ij'); axis('off')
        

## Open curve
#image
n = 256
f = load_image(name, n)
f = f[45:105, 60:120]
n = f.shape[0]
imageplot(f)

#initialisation
x0 = 4 + 55j
x1 = 53 + 4j
p = 128
t = transpose(linspace(0, 1, p))
gamma0 = t*x1 + (1-t)*x0
gamma = gamma0
cplot(gamma,'r')

# calcul de la métrique : fonction décroissante du gradient lissé
G = grad(f)
d0 = sqrt(sum(G**2,2))
imageplot(d0)

a = 2
d = gaussian_blur(d0,a)
d = minimum(d,.4)
W = rescale(-d,.4,1)
imageplot(W)  

# paramètres de l'évolution
dt = 0.15
Tmax = 2000
niter = int(Tmax/ dt)

# Evolution de la courbe
G = grad(W)
G = G[:,:,0] + 1j*G[:,:,1]
imageplot(abs(G))
EvalG = lambda gamma: bilinear_interpolate(G, imag(gamma), real(gamma))
EvalW = lambda gamma: bilinear_interpolate(W, imag(gamma), real(gamma))
gamma = gamma0

imageplot(f)
cplot(gamma, 'b')
dotp = lambda c1,c2: real(c1)*real(c2) + imag(c1)*imag(c2)
for i in arange(0,niter+1):
    N = normal(gamma)
    g = EvalW(gamma) * normalC(gamma) - dotp(EvalG(gamma), N) * N
    gamma = resample(gamma + dt*g)
    gamma[0] = x0
    gamma[-1] = x1
    if (i == 1) or ((i % 500) == 0) or (i == niter):
        cplot(gamma, 'r')
        
#%% SECTION 2a - Ensembles de niveaux

#discrétisation du domaine [0,1]^2
n = 200
Y,X = np.meshgrid(np.arange(1,n+1), np.arange(1,n+1))

#cercle et fonction distance
r = n/3.
c = np.array([r,r]) + 10
phi1 = np.sqrt((X-c[0])**2 + (Y-c[1])**2) - r

# cercle de centre différent (on peut aussi remplacer le cercle par une autre forme, par exemple un carré)
#do_fast = 1
r = n/3.
c = n  - 10 - np.array([r,r])
phi2 = np.sqrt((X-c[0])**2 + (Y-c[1])**2) - r

from nt_toolbox.plot_levelset import *
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
plot_levelset(phi1)

plt.subplot(1,2,2)
plot_levelset(phi2)

#%% SECTION 2b - Ensembles de niveaux - intersection et réunion
clf
plt.figure(figsize = (10,5))
phi0 = np.minimum(phi1, phi2)
plt.subplot(1,2,1)
plot_levelset(phi0)
title('Union')
plt.subplot(1,2,2)
plot_levelset(np.maximum(phi1, phi2))
title('Intersection')


#%% SECTION 2c - Mouvement de courbure moyenne

Tmax = 7
tau = 1
niter = int(Tmax/tau)
phi = np.copy(phi0)
k=1
from nt_toolbox.grad import *
from nt_toolbox.div import *
#from nt_toolbox.perform_redistancing import *

for i in range(1,niter+1):
# gradient de phi, norme du gradient et gradient normalisé
    g0 = grad(phi, order=2)
    eps = np.finfo(float).eps
    d = np.maximum(eps*np.ones([n,n]), np.sqrt(np.sum(g0**2, 2)))
    g = g0/np.repeat(d[:,:,np.newaxis], 2, 2)

# terme de courbure
    K = - d*div(g[:,:,0], g[:,:,1], order=2)

#descente de gradient
    phi = phi - tau*K
#    if ((i % 30) == 0):
#      phi = perform_redistancing(phi)
    k=k+1
    if (i == 1) or ((i % 30) == 0) or (i == niter):
        plot_levelset(phi)


#%% SECTION 2d - Segmentation par ensemble de niveaux et information de régions
n = 256
name = 'brain.bmp'
f = load_image(name, n)
imageplot(f)
#Initialisation par plusieurs cercles (images 256x256)
n=256
Y,X = np.meshgrid(np.arange(1,n+1), np.arange(1,n+1))
k = 4
r = .3*n/ k
phi0 = np.zeros((n, n)) +2*n
for i in range(1, k+1):
    for j in range(1, k+1):
        c = np.array([i-1,j-1])*(n/ k) + (n/ k)*.5
        phi1 = np.sqrt((X-c[0])**2 + (Y-c[1])**2) - r
        phi0 = np.minimum(phi0, phi1)
subplot(1, 2, 1)
plot_levelset(phi0)
subplot(1, 2, 2)
plot_levelset(phi0, 0, f)
plt.axis()


# paramètres de l'évolution
lambd = 2
c1 = 0
c2 = 0.1
tau = .5
Tmax = 200
niter = int(Tmax/ tau)

#initialisation
phi = np.copy(phi0)

#gradient de phi
gD = grad(phi, order=2)
eps = np.finfo(float).eps
d = np.maximum(eps*np.ones([n,n]), np.sqrt(np.sum(gD**2, 2)))
g = gD/np.repeat(d[:,:,np.newaxis], 2, 2)
G = d * div(g[:,:,0], g[:,:,1], order=2) - lambd*(f-c1)**2 + lambd*(f-c2)**2
#une étape de descente de gradient
phi = phi + tau*G

#update c1 and c2

        

#%% SECTION 2e descente de gradient complete

#n = 256
#name = 'brain2.bmp'
#name = 'retineOA.bmp'
#name = 'coeurIRM.bmp'

#f = load_image(name, n)
#imageplot(f)
#Initialisation par plusieurs cercles (images 256x256)
#n=256
#Y,X = np.meshgrid(np.arange(1,n+1), np.arange(1,n+1))
#k = 4
#r = .2*n/ k
#phi0 = np.zeros((n, n)) +2*n
#for i in [100,100,200,190]:
#    for j in [100,100,170,170]:
#        c = np.array([i,j])
#        phi1 = np.sqrt((X-c[0])**2 + (Y-c[1])**2) - r
#        phi0 = np.minimum(phi0, phi1)
#subplot(1, 2, 1)
#plot_levelset(phi0)
#subplot(1, 2, 2)
#plot_levelset(phi0, 0, f)
#plt.axis()

lambd = 2
c1 = 0.6
c2 = 0.
tau = .5
Tmax = 175
niter = int(Tmax/ tau)
phi = phi0
list_c1 = [c1]
list_c2 = [c2]
for i in range(1, niter+1):
    gD = grad(phi, order=2)
    d = np.maximum(eps*np.ones([n,n]), np.sqrt(np.sum(gD**2, 2)))
    g = gD/np.repeat(d[:,:,np.newaxis], 2, 2)
    G = d * div(g[:,:,0], g[:,:,1], order=2) - lambd*(f-c1)**2 + lambd*(f-c2)**2
    phi = phi + tau*G
#    c1 = c1-0.1*f[phi>0].mean()
#    c2 = c2-0.1*f[phi<0].mean()
#    list_c1.append(c1)
#    list_c2.append(c2)
#    if ((i % 30) == 0):
#       phi = perform_redistancing(phi)
subplot(1, 2, 1)
plot_levelset(phi)
subplot(1, 2, 2)
plot_levelset(phi, 0, f)

plt.plot(range(len(list_c1)),list_c1)
plt.plot(range(len(list_c2)),list_c2)

