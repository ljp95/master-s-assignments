#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:49:47 2019

@author: 3602786
"""

# 1-bleu     2-vert   3-rouge
from constraint import *
import numpy as np
import copy
import time

def solve_p1():
    p = Problem();
    p.addVariable("a",[3,1])
    p.addVariable("b",[2,1])
    p.addVariable("c",[2,1])
    p.addVariable("d",[3])
    
    p.addConstraint(AllDifferentConstraint(),["a","b"])
    p.addConstraint(AllDifferentConstraint(),["a","c"])
    p.addConstraint(AllDifferentConstraint(),["a","d"])
    p.addConstraint(AllDifferentConstraint(),["b","c"])
    p.addConstraint(AllDifferentConstraint(),["c","d"])
    
    s = p.getSolutions()
    return s

#print(solve_p1())

def solve_p2():
    p = Problem();
    p.addVariable("a",[3,2])
    p.addVariable("b",[2,1])
    p.addVariable("c",[2,1])
    p.addVariable("d",[2])
    
    p.addConstraint(AllDifferentConstraint(),["a","b"])
    p.addConstraint(AllDifferentConstraint(),["a","c"])
    p.addConstraint(AllDifferentConstraint(),["a","d"])
    p.addConstraint(AllDifferentConstraint(),["b","c"])
    p.addConstraint(AllDifferentConstraint(),["c","d"])
    
    s = p.getSolutions()
    return s

#print(solve_p2())

class Zone(object):
    def __init__(self, voisins = [], utilite = {}):
        self.voisins = voisins
        self.utilite = utilite

z1 = Zone(); z2 = Zone(); z3 = Zone(); z4 = Zone()
z1.voisins = [z2,z3,z4]; z1.utilite = {'b':1, 'g':0.3, 'r':0.8}
z2.voisins = [z1,z3]; z2.utilite = {'b':0.2, 'g':0.6, 'r':1}
z3.voisins = [z1,z2,z4]; z3.utilite = {'b':1, 'g':0.7, 'r':0.4}
z4.voisins = [z1,z3]; z4.utilite = {'b':0.5, 'g':1, 'r':0.9}
    
zones = [z1,z2,z3,z4]
maxi = 0
current_value = 0
couleurs = []

def evaluation(zones,maxi):
    for zone in zones:
        maxi += max(zone.utilite.values())
    return maxi

def bb(zones,maxi,current_value,couleurs):
    #feuille atteinte
    if zones == []:
        return max(maxi,current_value),couleurs
    
    tmp_couleurs = copy.deepcopy(couleurs)

    #rangement des cles du noeud courant par leur valeur
    sorted_d = sorted((value, key) for (key,value) in zones[0].utilite.copy().items())
    sorted_d = sorted(sorted_d,reverse=True)

    for valeur,couleur in sorted_d:
        
        #copie des parametres qui seront modifies
        zones2 = copy.deepcopy(zones)
        noeud = zones2.pop(0)
        current_value2 = current_value + noeud.utilite[couleur]
        couleurs2 = copy.deepcopy(tmp_couleurs)
        couleurs2.append(couleur)
        
        #comparaison avec la borne superieure de la branche, passer si branche inutile   
        print(maxi,evaluation(zones2,current_value2))
        if(maxi > evaluation(zones2,current_value2)):
            print('break')
            break
        
        #enlever la couleur des voisins du noeud, passer si pas de solution possible
        for voisin in noeud.voisins:
            if voisin.utilite.pop(couleur,None) == None:
                break
        
        #recursion avec les nouveaux parametres
        utilite_branche, couleurs_branche = bb(zones2,maxi,current_value2,couleurs2)
        
        #stockage si meilleur
        if utilite_branche > maxi:
            maxi = utilite_branche
            couleurs = couleurs_branche
            
    return maxi,couleurs
debut=time.time()
a,b = bb(zones,maxi,maxi,couleurs)
print("Solution d'utilite {} et de couleurs {}".format(a,b))
print(time.time()-debut)

    
    
    
    
    
    
    
    
    
    
    
    
    