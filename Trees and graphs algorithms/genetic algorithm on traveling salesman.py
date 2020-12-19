#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:20:09 2019

@author: 3801964
"""
import numpy as np
import random
from TSP import *
import copy
import matplotlib.pyplot as plt

def ordinal(ind):
    non_visitee = list(range(1,len(ind)+1))
    ordre = []
    for i in range(len(ind)):
        ordre.append(non_visitee.ordre(ind[i])+1)
        non_visitee.remove(ind[i])
    return ordre

def croisement_point(x,y):
    z = copy.deepcopy(x)
    pos = random.randint(1,len(x)-1)
    z[pos:] = y[pos:]
    return z
    
def croisement_point_rang(x,y):
    pos = random.randint(1,len(x)-1)
    z1 = copy.deepcopy(x[:pos])
    z2 = copy.deepcopy(y[:pos])
    for i in y:
        if i not in z1:
            z1.append(i)
    for i in x:
        if i not in z2:
            z2.append(i)            
    return z1,z2

def generation_population(N,n):
    pop = []
    for i in range(N):
        ind = np.arange(1,n+1)
        np.random.shuffle(ind)
        ind = ind.tolist()
        pop.append(ind)
    return pop

def evaluation_population(pop,cities):
    f = []
    for ind in pop:
        f.append(evaluation(ind,cities))
    return f

def mutation(ind,pm):
    for i in range(len(ind)):
        r = random.random()
        if(r<=pm):
            pos = random.randint(0,len(ind)-1)
            ind[i],ind[pos] = ind[pos],ind[i]
    return ind
    
def selection(pop,probas):
    r1 = random.random()
    r2 = random.random()
    for i in range(len(pop)):
        if r1<probas[i]:
            break
    x = pop[i]
    for i in range(len(pop)):            
        if r2<probas[i]:
            break
    y = pop[i]
    return x,y

def methode_standard(f):
    probas = 1/np.array(f)
    probas = probas.cumsum()/probas.sum()
    return probas

def rangement_par_qualite(f,p):
    n = len(f)
    indices = list(range(n))
    sorted_d = sorted([valeur, indice] for (indice,valeur) in zip(indices,f))
    indices = np.array((sorted_d))[:,1]
    probas_tmp = np.zeros(n)
    for i in range(n):
        probas_tmp[i] = p*pow(1-p,i)
    probas = sorted([indice,proba] for (indice,proba) in zip(indices,probas_tmp))
    probas = np.array((probas))[:,1] 
    probas = probas.cumsum()/probas.sum()
    return probas

def algorithme_genetique(N,NbG,pm,ps,cities):
    #initialisation
    pop = generation_population(N,len(cities))
    res = []
    for k in range(NbG):
        #evaluation            
        f = evaluation_population(pop,cities)
        #proba selection
        probas = rangement_par_qualite(f,ps)
        #stockage meilleure distance
        res.append(min(f))
        pop2 = []
        for i in range(N//2):
            #selection
            x,y = selection(pop,probas)
            #croisement
            z1,z2 = croisement_point_rang(x,y)
            #mutation
            z1 = mutation(z1,pm)
            z2 = mutation(z2,pm)
            #insertion
            pop2.append(z1)
            pop2.append(z2)
        pop = pop2
    #stockage meilleure distance
    f = evaluation_population(pop,cities)
    res.append(min(f))
    return res

instance = "kroB100.tsp"
data = read_tsp_data(instance)
nbCities = int(detect_dimension(data))	
cities = read_tsp(nbCities,data)

N = 100
NbG = 100
pm = 0.05
ps = 0.4
res = algorithme_genetique(N,NbG,pm,ps,cities)
x = list(range(len(res)))
plt.plot(x,res)



