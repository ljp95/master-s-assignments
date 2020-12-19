# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 02:09:57 2018

@author: HOME
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# old version = python 2
# data = pkl.load(file("ressources/lettres.pkl","rb"))
# new : 
with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return

#Q1 discretisation des valeurs du signal
def discretise(X,d):
    intervalle = 360/d
    Xd = []
    for signal in X:
        Xd.append(np.floor(signal/intervalle))
    return np.array(Xd)
    
d = 3
Xd = discretise(X, d)
#print(Xd)

#Q2 Regroupement des indices des signaux par classe
def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

#Q3 Apprentissage des modèles
    
def learnMarkovModel(Xc,d):
    A = np.zeros((d,d))  #Fin de l'ex remplacer np.zeros par np.ones
    Pi = np.zeros(d)
    for signal in Xc:
        signal = signal.astype(int)
        Pi[signal[0]]+=1
        for i in range(1,len(signal)):
            A[signal[i-1],signal[i]]+=1    
    A = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return [Pi,A]

d = 3
Xd = discretise(X,d)
C = groupByLabel(Y)
Xc = Xd[C[0]] 

Pi,A = learnMarkovModel(Xc,d)
#print(Pi,A)

#Q4
#d=3     # paramètre de discrétisation
d=20
Xd = discretise(X,d)  # application de la discrétisation
index = groupByLabel(Y)  # groupement des signaux par classe
models = []
for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
    models.append(learnMarkovModel(Xd[index[cl]], d))
    
#TEST
    
def probaSequence(s,Pi,A):
    s = s.astype(int)
    proba_sequence = np.log(Pi[s[0]])
    for i in range(len(s)-1):
        proba_sequence += np.log(A[s[i],s[i+1]])
    return proba_sequence

proba_models = []
s = Xc[0]

for model in models:
    proba_models.append(probaSequence(s,model[0],model[1]))
proba_models=np.array(proba_models)
#print(proba_models)
    
#Q2
proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in range(len(Xd))]for cl in range(len(np.unique(Y)))])
#print(proba)

#Q3
Ynum = np.zeros(Y.shape)
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num
    
pred = proba.argmax(0) # max colonne par colonne

print(np.where(pred != Ynum, 0.,1.).mean())

#Evaluation

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest
# exemple d'utilisation
itrain,itest = separeTrainTest(Y,0.8)

ia = []
for i in itrain:
    ia += i.tolist()    
it = []
for i in itest:
    it += i.tolist()

#print(itrain)
for i in range(20):
    d = 70
    Xd = discretise(X, d)
    itrain,itest = separeTrainTest(Y,0.8)
    models = []
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        models.append(learnMarkovModel(Xd[itrain[cl]], d))
    proba = np.array([[probaSequence(Xd[i], models[cl][0], models[cl][1]) for i in it]for cl in range(len(np.unique(Y)))])
    Y2 = Y[it]
    Ynum = np.zeros(Y2.shape)
    for num,char in enumerate(np.unique(Y)):
        Ynum[Y2==char] = num
        
    pred = proba.argmax(0) # max colonne par colonne
    
    print(np.where(pred != Ynum, 0.,1.).mean())



