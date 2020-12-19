#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:50:20 2018

@author: 3602786
"""

import numpy as np
import pickle as pkl
data = pkl.load(open("genome_genes.pkl","rb"),encoding='latin1')

Xgenes  = data.get("genes") #Les genes, une array de arrays

Genome = data.get("genome") #le premier million de bp de Coli

Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]
 

Xgenes[0][:6]
Xgenes[0][-6:]

from markov_tools import *
a = 1/200

moy_seq = 0
for i in range(Xgenes.size):
    moy_seq += len(Xgenes[i])
moy_seq /= Xgenes.size

b = 1/moy_seq

Pi = np.array([1, 0, 0, 0])
A =  np.array([[1-a, a  , 0, 0],
              [0  , 0  , 1, 0],
              [0  , 0  , 0, 1],
              [b  , 1-b, 0, 0 ]])
B = ...


taille = Genome.size
proba_A = (Genome == 0).sum() / taille
proba_C = (Genome == 1).sum() / taille
proba_G = (Genome == 2).sum() / taille
proba_T = (Genome == 3).sum() / taille

Binter = np.array([proba_A,proba_C, proba_G, proba_T])

Bgene = np.zeros((3,4))

for gene in Xgenes:
    for i in range(1,int(len(gene)/3)-1):
        Bgene[0][gene[3*i]]+=1
        Bgene[1][gene[3*i+1]]+=1
        Bgene[2][gene[3*i+2]]+=1

        
Bgene = Bgene/np.maximum(Bgene.sum(1).reshape(3,1),1)

B_m1 = np.vstack((Binter, Bgene))

s, logp = viterbi(Genome,Pi,A,B_m1)
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp = s
sp[np.where(sp>=1)] = 1
percpred1 = float(np.sum(sp == Annotation) )/ len(Annotation)

percpred1

A = np.array(([1-a,a*0.83,a*0.14,a*0.03,0,0,0,0,0,0,0,0],
             [0,0,0,0,1,0,0,0,0,0,0,0], #ATG
             [0,0,0,0,1,0,0,0,0,0,0,0], #TTG
             [0,0,0,0,1,0,0,0,0,0,0,0], #GTG
             [0,0,0,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,0,0,0,0,0],
             [0,0,0,0,1-b,0,b,0,0,0,0,0], #position 2
             [0,0,0,0,0,0,0,0,0.5,0.5,0,0], #T
             [0,0,0,0,0,0,0,0,0,0,0.5,0.5], #T - A
             [0,0,0,0,0,0,0,0,0,0,1,0], #T - G
             [1,0,0,0,0,0,0,0,0,0,0,0],
             [1,0,0,0,0,0,0,0,0,0,0,0]))
             
s, logp = viterbi(Genome,Pi,A,B_m1)
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp = s
sp[np.where(sp>=1)] = 1
percpred2 = float(np.sum(sp == Annotation) )/ len(Annotation)

percpred2

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    