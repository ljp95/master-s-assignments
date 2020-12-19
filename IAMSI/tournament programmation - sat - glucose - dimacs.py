#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:05:07 2019

@author: 3602786
"""

import numpy as np
import time
import os

########## Exercice 2 ###########

def nb_VarProp(ne,nj):
    return nj*ne**2

def K(ne,nj,j,x,y):
    return j*(ne**2)+x*ne+y+1

def resolution(k,ne):
    j = (k-1)//(ne**2)
    x = (k-1-j*(ne**2))//ne
    y = k-1-j*(ne**2)-x*ne
    return j,x,y


########### Exercice 3 ###########
    
#question1
def au_moins(liste):
    liste_clause = ""
    for i in liste:
        liste_clause = liste_clause + str(i) + ' ' 
    liste_clause += str(0)+'\n'
    return liste_clause
        
def au_plus(liste):
    liste_clause = ""
    clause = ""
    liste = np.array(liste)
    liste = -1 * liste
    for i in range(len(liste)):
        for j in range(i+1,len(liste)):
            clause = str(liste[i]) + ' ' + str(liste[j]) + ' ' + str(0) + '\n'
            liste_clause = liste_clause + clause
    return liste_clause
   
liste = [1,2,3,4]
print(au_moins(liste))
print(au_plus(liste))

#question2
def encoderC1(ne,nj):
    liste_clause = "" 
    liste_domicile = [] 
    liste_exterieur = [] 
    for j in range(nj):
        for x in range(ne):
            for y in range(ne):
                if x!=y:
                    liste_domicile.append(K(ne,nj,j,x,y))
                    liste_exterieur.append(K(ne,nj,j,y,x))
            liste = liste_domicile + liste_exterieur
            liste_domicile = []
            liste_exterieur = []
            liste_clause = liste_clause + au_plus(liste) 
    return liste_clause

ne,nj = 3,6
a = encoderC1(ne,nj)
print(a + '\n')
print(len(a.split('\n'))-1)
# 72 contraintes retournées
    


def encoderC2(ne,nj):
    liste_clause = ""
    liste_domicile = []
    liste_exterieur = []
    for x in range(ne):
        for y in range(ne):
            if(x == y):
                break
            for j in range(nj):
                liste_domicile.append(K(ne,nj,j,x,y))
                liste_exterieur.append(K(ne,nj,j,y,x))
            liste_clause += au_moins(liste_domicile) + au_plus(liste_domicile)
            liste_clause += au_moins(liste_exterieur) + au_plus(liste_exterieur)
            liste_domicile = []
            liste_exterieur = []
    return liste_clause

ne,nj = 3,6
a = encoderC2(ne,nj)
print(a + '\n')
print(len(a.split('\n'))-1)
#42 contraintes

def encoder(ne,nj):
    res = encoderC1(ne,nj) + encoderC2(ne,nj)
    f=open("encodage.cnf", "w")
    nb_var=nb_VarProp(ne,nj)
    nb_clause=len(res.split('\n'))-1
    ligne1="p cnf "+str(nb_var)+ " "+str(nb_clause) +'\n'
    f.write(ligne1)
    f.write(res)
    f.close()

ne,nj = 4,6
encoder(ne,nj)
#print(a + '\n')
#print(len(a.split('\n'))-1)
#114 contraintes

#question 3
def decoder(fichier,ne, f_equipe):
    f=open(fichier,"r+") #ouvre le fichier du resultat de glucose
    f2=open("match.txt","w")
    f3=open(f_equipe,"r")
    eq=f3.readlines()
    for k in range(len(eq)):
        eq[k]=eq[k][:-1]
    res=f.readline()
    res=res.split(' ')
    #on enleve "v" et "0\n"
    res.remove(res[0])
    #res.remove(res[-1])
    for i in res[1:]:
        i = int(i)
        if (i>0):
            j,x,y=resolution(i,ne)
            f2.write("Jour "+str(j)+ ": l'equipe " +eq[x] +" joue contre l'equipe " +eq[y] +'\n')
    f2.close()
    f3.close()
    f.close()
    
    
#decoder("resultat.txt", 4, "equipe.txt")


def assemblage(ne,nj):
    encoder(ne,nj)
    os.system("rm resultat.txt") #supprimer le fichier s'il existe deja
    cmd="./glucose_static -model encodage.cnf | tail -n 1 >> resultat.txt"
    os.system(cmd)
    decoder("resultat.txt",ne,"equipe.txt") #pour afficher le planning des matchs dans le fichier "match.txt"
    

assemblage(4,6)


########## Exercice 4 ###############   
def main():
    list_nj=[]
    for ne in range(3,10):
        debut=time.time()
        s=False
        nj=ne       
        while s!=True:
            encoder(ne,nj)
            os.system("rm satisfiable.txt") #supprimer le fichier s'il existe deja
            cmd="./glucose_static encodage.cnf | tail -n 1 >> satisfiable.txt"
            os.system(cmd)
            f=open("satisfiable.txt","r")
            res=f.readline()
            res=res[2:-1]
            if res=="SATISFIABLE":
                s=True
            nj+=1  
            if (time.time()-debut>10):
                print("reponse inconnue pour ne= "+str(ne))
                nj=0
                break            
 
        list_nj.append(nj-1)
    print(list_nj)
    
main()


    
    
            
        




