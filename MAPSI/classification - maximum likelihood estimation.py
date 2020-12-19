#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:53:14 2018

@author: 3602786
"""

# -*- coding: utf-8 -*- 

import numpy as np
import matplotlib.pyplot as plt
import math

def read_file ( filename ):
    """
    Lit un fichier USPS et renvoie un tableau de tableaux d'images.
    Chaque image est un tableau de nombres réels.
    Chaque tableau d'images contient des images de la même classe.
    Ainsi, T = read_file ( "fichier" ) est tel que T[0] est le tableau
    des images de la classe 0, T[1] contient celui des images de la classe 1,
    et ainsi de suite.
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # creation de la structure de données pour sauver les images :
    # c'est un tableau de listes (1 par classe)
    data = np.empty ( 10, dtype=object )   
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    # lecture des images du fichier et tri, classe par classe
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # transformation des list en array
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()

training_data = read_file ( "2015_tme3_usps_train.txt" )

# affichage du 1er chiffre "2" de la base:
display_image ( training_data[2][0] )

# affichage du 5ème chiffre "3" de la base:
display_image ( training_data[3][4] )


#######Q2 et 3


def learnML_class_parameters(tabdimages):
    return (tabdimages.mean(0),tabdimages.var(0))

print("Q2.1 : \n", learnML_class_parameters ( training_data[0] ))
print("Q2.2 : \n", learnML_class_parameters ( training_data[1] ))
    
def learnML_all_parameters(data):
    parameters = []
    for i in range(len(data)):
        parameters.append(learnML_class_parameters(data[i]))
    return parameters

parameters = learnML_all_parameters(training_data)


#######Q4   


test_data = read_file("2015_tme3_usps_test.txt")

def log_likelihood(image, parameters_image):
    log_vrai = 0.
    moy = parameters_image[0]
    var = parameters_image[1]
    for i in range(len(image)):
        if(var[i] != 0):
            log_vrai += -0.5*(math.log(2*math.pi*var[i])+math.pow((image[i]-moy[i]),2)/var[i])
    return log_vrai

print("Q4.1 : \n", log_likelihood ( test_data[2][3], parameters[1] ))

print("Q4.2 : \n", [ log_likelihood ( test_data[0][0], parameters[i] ) for i in range ( 10 ) ])


#######Q5


def log_likelihoods(image, parameters):
    tab_log_vrai = np.zeros((1,10))
    for i in range(10):
        tab_log_vrai[0][i] = log_likelihood(image, parameters[i])
    return tab_log_vrai

print("Q5 : \n", log_likelihoods ( test_data[1][5], parameters ))


#######Q6


def classify_image(image, parameters):
    return log_likelihoods(image, parameters).argmax()

print(classify_image( test_data[1][5], parameters ))
print(classify_image( test_data[4][1], parameters ))


            

    
    
    
    
    
    
    
    
    
    