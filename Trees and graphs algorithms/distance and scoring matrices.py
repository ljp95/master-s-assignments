#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:06:07 2018

@author: 3602786


"""

import numpy as np

def matrice_distance(alphabet, match, mismatch):
    n = len(alphabet)
    #matrice avec match en diagonale et mismatch ailleurs
    D = np.zeros((n,n)) + mismatch + np.eye(n)*(match-mismatch)
    return D
    
#petit test
alphabet = [0,1,2,3]
match = 1
mismatch = -1
print(matrice_distance(alphabet, match, mismatch))

def matrice_score(A, B, D, gap):  #[1ere sequence, 2nde sequence, matrice_distance, penalite pour gap]
    #nombre de lignes et colonnes de la matrice de score
    n = len(A)+1 
    m = len(B)+1
    
    #creation de la matrice de score
    M = np.zeros((n,m))  
    M[0] += np.arange(0,-m,-1)
    M[:,0] += np.arange(0,-n,-1)
#    D1 = np.zeros((n-1,m-1))
#    D2 = np.array([np.arange(0,-m,-1)*(-gap)])
#    D3 = np.array([np.arange(-1,-n,-1)*(-gap)])
#    M = np.concatenate((D3.T,D1),axis=1)
#    M = np.concatenate((D2,M))
    
    #calcul des scores par max des valeurs ouest, nord et nord-ouest
    for i in list(range(n-1)):
        for j in list(range(m-1)):
            M[i+1,j+1] = max(M[i,j]+ D[A[i],B[j]],M[i+1,j]+gap,M[i,j+1]+gap)
    return M

#test sur les sequences du td1
#transformation en valeurs numeriques de la sequence
A = [2,1,2,3,0,0,1] #['T','C','T','G','A','A','C'] 
B = [0,1,2,3,0,1] #['C','A','T','G','A','C'] 
alphabet = ['A','T','C','G'] 
match = 1
mismatch = -2
gap = -1
D = matrice_distance(alphabet,match,mismatch);
print(matrice_score(A,B,D,gap))


def matrice_score2(A, B, D, gap):  #[1ere sequence, 2nde sequence, matrice_distance, penalite pour gap]
    #nombre de lignes et colonnes de la matrice de score
    n = len(A)+1 
    m = len(B)+1
    
    #creation de la matrice de score
    M = np.zeros((n,m))  
    M[0] += np.arange(0,-m,-1)
    M[:,0] += np.arange(0,-n,-1)
#    D1 = np.zeros((n-1,m-1))
#    D2 = np.array([np.arange(0,-m,-1)*(-gap)])
#    D3 = np.array([np.arange(-1,-n,-1)*(-gap)])
#    M = np.concatenate((D3.T,D1),axis=1)
#    M = np.concatenate((D2,M))
    
    #calcul des scores par max des valeurs ouest, nord et nord-ouest
    for i in list(range(n-1)):
        for j in list(range(m-1)):
            M[i+1,j+1] = max(M[i,j]+ D[A[i],B[j]],M[i+1,j]+gap,M[i,j+1]+gap)
    
    i = n-1 #compteur des lignes et colonnes pour l'arrêt
    j = m-1
    #score et alignement partant d'en bas a droite
    chemin1 = [i] 
    chemin2 = [j]
    score = M[i,j] 
    while((i != 0) and (j != 0)): #tant qu'on a pas atteint un bord
        # calcul des valeurs ouest,nord et nord-ouest
        Mj = M[i,j-1] #decalage en colonne
        Mi = M[i-1,j] #decalage en ligne
        Mij = M[i-1,j-1] #decalage en ligne et colonne
        if(Mj == max(Mj,Mi,Mij)):  #si le max est celui de l'ouest alors pas de decalage pour A
            chemin1.append(i)
            j-=1
            chemin2.append(j)        
        else:
            if(Mi == max(Mi,Mij)): #si le max est celui du nord alors pas de decalage pour B
                i-=1
                chemin1.append(i)
                chemin2.append(j)
            else:   #si le max est celui du nord-ouest alors decalage pour A et B
                i-=1
                chemin1.append(i)
                j-=1
                chemin2.append(j)
    chemin1.reverse()  #inversion du chemin
    chemin2.reverse()
    seq1 = ['_']
    seq2 = ['_']
#    print(chemin1,chemin2)
    if(chemin1[0]):
        seq1[0] = A[0]
    if(chemin2[0]):
        seq2[0] = B[0]
    for i in range(1,len(chemin1)):
        if(chemin1[i] == chemin1[i-1]):
            seq1.append('_')
        else:
            seq1.append(A[chemin1[i-1]])
        if(chemin2[i] == chemin2[i-1]):
            seq2.append('_')
        else:
            seq2.append(B[chemin2[i-1]])

    return seq1, seq2, score

#test avec les sequences du td1
A = [2,1,2,3,0,0,1] #['T','C','T','G','A','A','C'] 
B = [1,0,2,3,0,1] #['C','A','T','G','A','C'] 
alphabet = ['A','C','T','G']  #[0,1,2,3]
match = 1
mismatch = -2
gap = -1
D = matrice_distance(alphabet,match,mismatch);
print(matrice_score2(A,B,D,gap))
#
#def sequence_to_numbers(alphabet, sequence):
#    B = range(len(alphabet))
#    sequence = []
#    for i in range(len(sequence)):
#        sequence.append()








