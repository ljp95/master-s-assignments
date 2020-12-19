import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,step=20,data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y
    """
    if data is not None:
        xmax,xmin,ymax,ymin = np.max(data[:,0]),np.min(data[:,0]),\
                              np.max(data[:,1]),np.min(data[:,1])
    x,y = np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
                      np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


######ALGORITHME DE DESCENTE DE GRADIENT 
def optimize(fonc, dfonc, xinit, eps, max_iter):
    x_histo = [xinit]
    f_histo = [fonc(xinit)]
    grad_histo = [dfonc(xinit)]
    for i in range(max_iter):
        x = x_histo[-1] - eps*grad_histo[-1]
        x_histo.append(x)
        f_histo.append(fonc(x))
        grad_histo.append(dfonc(x))
    return np.array(x_histo), np.array(f_histo), np.array(grad_histo)

######OPTIMISATION DE FONCTIONS 
#Definition des fonctions et gradients
def f1(x):
    return x*np.cos(x)
def df1(x):
    return np.cos(x)-x*np.sin(x)

def f2(x):
    return -np.log(x)+x**2
def df2(x):
    return -1/x+2*x

def f3(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
def df3(x):
    return np.array([200*(x[1]-x[0]**2)*(-2)*x[0]-2*(1-x[0]), 200*(x[1]-x[0]**2)])

#f1
#Affichage fonction
x = np.linspace(-10,10,100)
plt.plot(x,f1(x))
plt.title('f1(x)'),plt.xlabel('x'),plt.ylabel('f1(x)')
plt.show()
#Parametres et calcul
max_iter = 200
eps = 0.01
xinit = 2
x_histo,f_histo,grad_histo = optimize(f1,df1,xinit,eps,max_iter)
iterations = list(range(len(x_histo)))
#Affichage f1 et gradient de f1 en fonction des iterations
plt.plot(iterations,f_histo);
plt.title('f1 selon nb iterations'),plt.xlabel('nb iterations'),plt.ylabel('f1')
plt.show()
plt.plot(iterations,grad_histo)
plt.title('gradient de f selon nb iterations'),plt.xlabel('nb iterations'),plt.ylabel('gradient de f1')
plt.show()
#Affichage f1 et trajectoire  de l'optimisation
x = np.linspace(x_histo.min(),x_histo.max(),100)
plt.plot(x,f1(x),'b',label = 'f1(x)')
plt.title('f1(x) et trajectoire'),plt.xlabel('x'),plt.ylabel('f1(x)')
plt.plot(x_histo,f_histo,'+r', label = 'trajectoire (init a {})'.format(xinit))
plt.legend(),plt.show()
#Affichage (t,log||x^t-x*||)
plt.plot(iterations[0:-1],np.log(abs(x_histo[0:-1]-x_histo[-1])))
plt.title('f1 : (t,log||x^t-x*||)'),plt.xlabel('t'),plt.ylabel('log||x^t-x*||')
plt.show()

#f2
#Affichage fonction
x = np.linspace(-10,10,100)
plt.plot(x,f2(x))
plt.title('f2(x)'),plt.xlabel('x'),plt.ylabel('f2(x)')
plt.show()
#Parametres et calcul
max_iter = 200
eps = 0.01
xinit = 8
x_histo,f_histo,grad_histo = optimize(f2,df2,xinit,eps,max_iter)
iterations = list(range(len(x_histo)))
#Affichage f et gradient de f en fonction des iterations
plt.plot(iterations,f_histo);
plt.title('f2 selon nb iterations'),plt.xlabel('nb iterations'),plt.ylabel('f2')
plt.show()
plt.plot(iterations,grad_histo)
plt.title('gradient de f selon nb iterations'),plt.xlabel('nb iterations'),plt.ylabel('gradient de f2')
plt.show()
#Affichage f et trajectoire  de l'optimisation
x = np.linspace(x_histo.min(),x_histo.max(),100)
plt.plot(x,f2(x),'b',label = 'f2(x)')
plt.plot(x_histo,f_histo,'+r', label = 'trajectoire (init a {})'.format(xinit))
plt.title('f2(x) et trajectoire'),plt.xlabel('x'),plt.ylabel('f2(x)')
plt.legend(),plt.show()
#Affichage (t,log||x^t-x*||)
plt.plot(iterations[0:-1],np.log(abs(x_histo[0:-1]-x_histo[-1])))
plt.title('f2 : (t,log||x^t-x*||)'),plt.xlabel('t'),plt.ylabel('log||x^t-x*||')
plt.show()

#f3
#Affichage 2d
grid,xx,yy = make_grid(-1,3,-1,3,20)
plt.contourf(xx,yy,f3([xx,yy]).reshape(xx.shape))
plt.title('f3(x)'),plt.xlabel('x'),plt.ylabel('y')
plt.colorbar()
fig = plt.figure()
#Affichage 3d
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, f3([xx,yy]).reshape(xx.shape),rstride=1,cstride=1,\
	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.show()
#Parametres et calcul
eps = 0.1
max_iter = 200
xinit = [0,0]
x_histo,f_histo,grad_histo = optimize(f3,df3,xinit,eps,max_iter)
iterations = list(range(len(x_histo)))
#Affichage f3 et gradient de f3 en fonction des iterations
plt.plot(iterations,f_histo);
plt.title('f3 selon nb iterations'),plt.xlabel('nb iterations'),plt.ylabel('f3')
plt.show()
plt.plot(iterations,grad_histo[:,0],'r',label = 'Derivee en x')
plt.plot(iterations,grad_histo[:,1],'b',label = 'Derivee en y')
plt.title('gradient de f3 selon nb iterations'),plt.xlabel('nb iterations'),plt.ylabel('gradient de f3')
plt.legend(),plt.show()
#Affichage f et trajectoire  de l'optimisation
grid,xx,yy = make_grid(-1,3,-1,3,20)
plt.contourf(xx,yy,f3([xx,yy]).reshape(xx.shape))
plt.scatter(x_histo[:,0],x_histo[:,1],c = 'r')
plt.title('Trajectoire'),plt.xlabel('x'),plt.ylabel('y')
fig = plt.figure()

#Affichage (t,log||x^t-x*||)
norme = np.sqrt(np.sum((x_histo[0:-1,:]-x_histo[-1,:])**2,1))
plt.plot(iterations[0:-1],np.log(norme))
plt.title('f3 : (t,log||point^t-point*||)'),plt.xlabel('t'),plt.ylabel('log||point^t-point*||')
plt.show()

###### REGRESSION LOGISTIQUE
class Logistique(object):
    def __init__(self,eps,max_iter):
        self.eps = eps
        self.max_iter = max_iter

    def fit(self,datax,datay):
        d = len(datax[0])
        self.w = np.zeros(d)      
        self.list_w = []
        for i in range(self.max_iter):
            gradient = np.dot(datax.T,sigmoide(np.dot(datax,self.w))-datay)/len(datay)
            w = self.w - self.eps*gradient
            self.w = w
            self.list_w.append(w)
        
    def predict(self,datax):
        return np.where(sigmoide(np.dot(datax,self.w))>=0.5,1,0)
    
    def score(self,datax,datay):
        return np.mean(self.predict(datax) == datay)
    
def sigmoide(z):
    return 1/(1+np.exp(-z))
    
#Chargement des donnees, choix de deux classes
classe1 = 6
classe2 = 9
#donnees train
datax, datay  = load_usps ( "USPS_train.txt" )
indices_train = np.logical_or(datay == classe1, datay == classe2)
datax_train = np.copy(datax[indices_train])
datay_train = np.copy(datay[indices_train])
indices_plus = datay_train == classe1 
indices_moins = np.logical_not(indices_plus)
datay_train[indices_plus] = 1
datay_train[indices_moins] = 0
#donnees test
datax, datay = load_usps ( "USPS_test.txt" )
indices_test = np.logical_or(datay == classe1, datay == classe2)
datax_test = np.copy(datax[indices_test])
datay_test = np.copy(datay[indices_test])
indices_plus = datay_test == classe1 
indices_moins = np.logical_not(indices_plus)
datay_test[indices_plus] = 1
datay_test[indices_moins] = 0

#Erreur train et test
eps = 0.05
max_iter = 100
reg = Logistique(eps,max_iter)
reg.fit(datax_train,datay_train)
print("Erreur : train %f, test %f"% (1-reg.score(datax_train,datay_train),1-reg.score(datax_test,datay_test)))
#Affichage poids
plt.figure()
plt.title('Poids {} vs {}'.format(classe1,classe2))
plt.imshow(reg.w.reshape((16,16)),interpolation="nearest",cmap="gray")

plt.colorbar()
plt.show()


