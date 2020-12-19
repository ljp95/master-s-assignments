#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:09:38 2018

@author: 3602786
"""

#https://www.uniprot.org/uniprot/P04637
file1 = open("reseau1.txt","r")
file2 = open("reseau2.txt","r")
#print(file2.read())
reseau = {}

line=file1.readline()
while line:
#        reseau[line].append()
    line = line.split(",",)
    line = [line[0],line[1].split("\n")[0]]
    if(int(line[0]) in reseau):
        reseau[int(line[0])].append(int(line[1]))
    else:
        reseau[int(line[0])]=[int(line[1])]
    if(int(line[1]) in reseau):
        reseau[int(line[1])].append(int(line[0]))
    else:
        reseau[int(line[1])]=[int(line[0])]    
    line=file1.readline()

file1.close()
file2.close()

def moy_clustering(reseau):
    cc = []
    for cle,valeur in reseau.items():
        E_N_cle = 0
        for i in range(len(valeur)):
            for j in range(i,len(valeur)):
                if(valeur[i] in reseau[j]):
                    E_N_cle += 1 
        cc.append([cle,(2*E_N_cle)/(len(valeur)*(len(valeur)-1))])
    return cc

print(moy_clustering(reseau))