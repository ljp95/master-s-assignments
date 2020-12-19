#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:43:32 2018

@author: 3602786
"""

################### OK jusqu'à la question 6. MetropolisHastings manque quelque chose##############""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import math

#ESTIMATION DE PI PAR MONTE CARLO
def tirage(m):
    return -m + 2*m*rd.random(),-m + 2*m*rd.random()

#print(tirage(1)) 

def monteCarlo(N):
    abscisses = []
    ordonnees = []
    pi = 0
    for n in range(N):
        x,y = tirage(1)
        abscisses.append(x)
        ordonnees.append(y)
        if((x**2 + y**2)**(1/2) <= 1):
            pi += 1
    pi = pi / N
    return 4*pi, np.array(abscisses), np.array(ordonnees)
    


#plt.figure()
#
## trace le carré
#plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')
#
## trace le cercle
#x = np.linspace(-1, 1, 100)
#y = np.sqrt(1- x*x)
#plt.plot(x, y, 'b')
#plt.plot(x, -y, 'b')
#
## estimation par Monte Carlo
#pi, x, y = monteCarlo(int(1e4))
#
## trace les points dans le cercle et hors du cercle
#dist = x*x + y*y
#plt.plot(x[dist <=1], y[dist <=1], "go")
#plt.plot(x[dist>1], y[dist>1], "ro")
#plt.show()
            

#DECODAGE PAR LA METHODE DE METROPOLIS HASTINGS

# si vos fichiers sont dans un repertoire "ressources"
with open("./ressources/countWar.pkl", 'rb') as f:
    (count, mu, A) = pkl.load(f, encoding='latin1')

with open("./ressources/secret.txt", 'r') as f:
    secret = f.read()[0:-1] # -1 pour supprimer le saut de ligne
    
#Q3
def swap(dict1):
    dict2 = dict1.copy()
    c1 = rd.choice(list(dict1.keys()))
    c2 = c1
    while(c2 == c1):
        c2 = rd.choice(list(dict1.keys()))
    dict2[c1],dict2[c2] = dict1[c2],dict1[c1]
    return dict2
#    
#tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
#print(swap(tau))
    

#Q4
def decrypt(mess, tau):
    traduction = str()
    for i in range(len(mess)):
        traduction += tau[mess[i]]
    return traduction
    
#tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
#print(decrypt ( "aabcd", tau ))
#print(decrypt ( "dcba", tau ))

#Q5
#chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))
with open("./ressources/fichierHash.pkl",'rb') as f:
    chars2index = pkl.load(f, encoding='latin1')

#Q6
    
def logLikelihood(mess, mu, A, chars2index):
    max_vraisemblance = math.log(mu[chars2index[mess[0]]])
    for i in range(0,len(mess)-1):
        max_vraisemblance += math.log(A[chars2index[mess[i]],chars2index[mess[i+1]]])
    return max_vraisemblance

#print(logLikelihood( "abcd", mu, A, chars2index))
#print(logLikelihood( "dcba", mu, A, chars2index))

#Q7
def MetropolisHastings(mess, mu, A, tau, N, chars2index):
    decodage_tau = decrypt(mess,tau)
    max_vraisemblance_tau = logLikelihood(decodage_tau, mu, A, chars2index)
#    print("Initialisation : ",decodage_tau, max_vraisemblance_tau)
    for i in range(N):
        new_tau = swap(tau)
        decodage_new_tau = decrypt(mess,new_tau)
        max_vraisemblance_new_tau = logLikelihood(decodage_new_tau, mu, A, chars2index)
        rapport = (max_vraisemblance_new_tau  )/(max_vraisemblance_tau  )
        u = rd.random()
        if(u < rapport):
            decodage_tau = decodage_new_tau
            max_vraisemblance_tau = max_vraisemblance_new_tau
#            print(decodage_tau, max_vraisemblance_tau) 
    return decodage_tau

def identityTau (count):
    tau = {}
    for k in list(count.keys ()):
        tau[k] = k
    return tau

with open("./ressources/secret2.txt", 'r') as f:
    secret2 = f.read()[0:-1] # -1 pour supprimer le saut de ligne

#print(MetropolisHastings( secret2, mu, A, identityTau (count), 10000, chars2index))
    
    
# ATTENTION: mu = proba des caractere init, pas la proba stationnaire
# => trouver les caractères fréquents = sort (count) !!
# distribution stationnaire des caracteres
freqKeys = np.array(list(count.keys()))
freqVal  = np.array(list(count.values()))
# indice des caracteres: +freq => - freq dans la references
rankFreq = (-freqVal).argsort()

# analyse mess. secret: indice les + freq => - freq
cles = np.array(list(set(secret2))) # tous les caracteres de secret2
rankSecret = np.argsort(-np.array([secret2.count(c) for c in cles]))
# ATTENTION: 37 cles dans secret, 77 en général... On ne code que les caractères les plus frequents de mu, tant pis pour les autres
# alignement des + freq dans mu VS + freq dans secret
tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])

print(MetropolisHastings(secret2, mu, A, tau_init, 100000, chars2index))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    