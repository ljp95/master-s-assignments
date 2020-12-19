#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:39:07 2018

@author: 3602786
"""

import numpy as np
import matplotlib.pyplot as plt

#####Donnee jouets
a = 6.
b = -1.
N = 100
sig = .4 # écart type

x = np.random.rand(1,N)
epsilon = np.random.randn(1,N)*sig
y = a*x[0]+b+epsilon
plt.scatter(x[0],y[0])

#####Estimation parametre probabiliste
a2 = np.cov(x[0],y[0])[0][1]/np.var(x[0])
b2 = np.mean(y[0])-a2*np.mean(x[0])
print('proba',[a2,b2])
Y = a2*x[0]+b2
plt.plot(x[0],Y)

#####Formulation au sens des moindres carres
X = np.hstack((x.reshape(N,1),np.ones((N,1))))
A = np.dot(np.transpose(X),X)
B = np.dot(np.transpose(X),y[0])
w = np.linalg.solve(A,B)
print('moindres carres',w)

#####Optimisation par descente de gradient
wstar = np.linalg.solve(X.T.dot(X), X.T.dot(y[0])) # pour se rappeler du w optimal

eps = 5e-3
#eps = 5e-2
eps = 2.5e-3
nIterations = 50
w = np.zeros(X.shape[1]) # init à 0
#w = np.array([5,1])
allw = [w]
for i in range(nIterations):
    # A COMPLETER => calcul du gradient vu en TD
    grad = np.array([-2*(x*(y-w[0]*x-w[1])).sum(), -2*(y-w[0]*x-w[1]).sum()])
    w = w - eps* grad
    allw.append(w)
    print (w)

allw = np.array(allw)

# tracer de l'espace des couts
ngrid = 20
w1range = np.linspace(-0.5, 8, ngrid)
w2range = np.linspace(-1.5, 1.5, ngrid)
w1,w2 = np.meshgrid(w1range,w2range)

cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-y)**2).sum()) for w1i in w1range] for w2j in w2range])

plt.figure()
plt.contour(w1, w2, cost)
plt.scatter(wstar[0], wstar[1],c='r')
plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )

from mpl_toolkits.mplot3d import Axes3D

costPath = np.array([np.log(((X.dot(wtmp)-y)**2).sum()) for wtmp in allw])
costOpt  = np.log(((X.dot(wstar)-y)**2).sum())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w1, w2, cost, rstride = 1, cstride=1 )
ax.scatter(wstar[0], wstar[1],costOpt, c='r')
ax.plot(allw[:,0],allw[:,1],costPath, 'b+-' ,lw=2 )


#####Extension non-lineaire
c = 1
y = a*x[0]**2+b*x[0]+c+epsilon
plt.scatter(x[0],y[0])
Xe = np.hstack((x.reshape(N,1)**2,x.reshape(N,1),np.ones((N,1))))
A = np.dot(np.transpose(Xe),Xe)
B = np.dot(np.transpose(Xe),y[0])
w = np.linalg.solve(A,B)
Y = w[0]*(x[0]**2)+w[1]*x[0]+w[2]
arg_x = x[0].argsort()
x.sort()
plt.plot(x[0],Y[arg_x])


#####Donnees reelles  NON TERMINE
data = np.loadtxt("winequality-red.csv", delimiter=";", skiprows=1)
N,d = data.shape # extraction des dimensions
pcTrain  = 0.7 # 70% des données en apprentissage
allindex = np.random.permutation(N)
indTrain = allindex[:int(pcTrain*N)]
indTest = allindex[int(pcTrain*N):]
X = data[indTrain,:-1] # pas la dernière colonne (= note à prédire)
Y = data[indTrain,-1]  # dernière colonne (= note à prédire)
# Echantillon de test (pour la validation des résultats)
XT = data[indTest,:-1] # pas la dernière colonne (= note à prédire)
YT = data[indTest,-1]  # dernière colonne (= note à prédire)






















