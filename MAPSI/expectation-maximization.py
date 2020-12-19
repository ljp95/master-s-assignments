# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# -*- coding: utf-8 -*-

import numpy as np
from math import *
from pylab import *

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data

data = read_file ( "2015_tme4_faithful.txt" )

#ou utiliser backslash /
def normale_bidim(x,z,p):
    a = 1/(2*math.pi*p[2]*p[3]*math.sqrt(1-math.pow(p[4],2)))
    b = -1/(2*(1-math.pow(p[4],2)))
    c = math.pow((x-p[0])/p[2],2)
    d = -2*p[4]*(x-p[0])*(z-p[1])/(p[2]*p[3])
    e = math.pow((z-p[1])/p[3],2)
    return a*math.exp(b*(c+d+e))


import matplotlib.pyplot as plt

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()

#print(normale_bidim ( 1, 2, (1.0,2.0,3.0,4.0,0) ))
#print(normale_bidim ( 1, 0, (1.0,2.0,1.0,2.0,0.7) ))

#dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.7) )
#dessine_1_normale ( (-3.0,-5.0,3.0,2.0,0.2) )

def dessine_normales ( data, params, weights, bounds, ax ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    # création de la grille
    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )# * weights[0]
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )# * weights[1]

    # affichages des normales et des points du dataset
    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    # récupération des paramètres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )


# affichage des données : calcul des moyennes et variances des 2 colonnes
mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

# les paramètres des 2 normales sont autour de ces moyennes
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [0.4, 0.6] )
bounds = find_bounds ( data, params )

# affichage de la figure
fig = plt.figure ()
ax = fig.add_subplot(111)
dessine_normales ( data, params, weights, bounds, ax )
plt.show ()



def Q_i(data, current_params, current_weights):
    T = np.zeros((size(data,0),2))
    for i in range(size(T,0)):
        a = current_weights[0]*normale_bidim(data[i][0],data[i][1],current_params[0])
        b = current_weights[1]*normale_bidim(data[i][0],data[i][1],current_params[1])
        T[i][0] = a/(a+b)
        T[i][1] = b/(a+b)
    return T

#current_params = np.array ( [(mu_x, mu_z, sigma_x, sigma_z, rho),   # params 1ère loi normale
#                             (mu_x, mu_z, sigma_x, sigma_z, rho)] ) # params 2ème loi normale
current_params = np.array([[ 3.28778309, 69.89705882, 1.13927121, 13.56996002, 0. ],
                           [ 3.68778309, 71.89705882, 1.13927121, 13.56996002, 0. ]])

# current_weights = np.array ( [ pi_0, pi_1 ] )
current_weights = np.array ( [ 0.5, 0.5 ] )

T = Q_i ( data, current_params, current_weights )

#print(T)

current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876,  0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )
T = Q_i ( data, current_params, current_weights )
#print(T)

def M_step(data,T,current_params,current_weights):
    
    a = sum(T[:,0])
    b = sum(T[:,1])
    pi0 = a/(a+b)
    pi1 = b/(a+b)
    ux0 = (sum(T[:,0]*data[:,0]))/a
    ux1 = (sum(T[:,1]*data[:,0]))/b
    uz0 = (sum(T[:,0]*data[:,1]))/a
    uz1 = (sum(T[:,1]*data[:,1]))/b
    sx0 = np.sqrt(sum(T[:,0]*(data[:,0]-ux0)*(data[:,0]-ux0))/a)
    sx1 = np.sqrt(sum(T[:,1]*(data[:,0]-ux1)*(data[:,0]-ux1))/b)
    sz0 = np.sqrt(sum(T[:,0]*(data[:,1]-uz0)*(data[:,1]-uz0))/a)
    sz1 = np.sqrt(sum(T[:,1]*(data[:,1]-uz1)*(data[:,1]-uz1))/b)
    p0 = sum(T[:,0]*(data[:,0]-ux0)*(data[:,1]-uz0)/(sx0*sz0))/a
    p1 = sum(T[:,1]*(data[:,0]-ux1)*(data[:,1]-uz1)/(sx1*sz1))/b
    return np.array([(ux0,uz0,sx0,sz0,p0),(ux1,uz1,sx1,sz1,p1)]),np.array([pi0,pi1])

current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
current_weights = array([ 0.45165145,  0.54834855])
Q = Q_i ( data, current_params, current_weights )
#print(M_step ( data, Q, current_params, current_weights ))

    
current_params = params
current_weights = weights
for i in range(20):
    T = Q_i ( data, current_params, current_weights )
    current_params,current_weights = M_step(data,T,current_params,current_weights)
    fig = plt.figure ()
    ax = fig.add_subplot(111)
    dessine_normales ( data, current_params, current_weights, bounds, ax )
    plt.show ()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    