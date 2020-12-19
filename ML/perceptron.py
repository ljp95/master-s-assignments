from arftools import *
import numpy as np
import matplotlib.pyplot as plt
import time

def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    """ moy((Xw-Y)^2) """
    return np.mean((np.dot(datax,w)-datay)**2)

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    """ (2*X'(Xw-Y))/N """
    return np.dot(datax.T,2*((np.dot(datax,w)-datay)))/len(datax)

def hinge(datax,datay,w,l=None):
    """ retourne la moyenne de l'erreur hinge """
    """ moy(max(0,-Y.*Xw)) """
    if l!=None:
        return np.mean(np.maximum(0.,-np.dot(datax,w)*datay))+l*np.dot(w[1:].T,w[1:])
    return np.mean(np.maximum(0.,-np.dot(datax,w)*datay))

def hinge_g(datax,datay,w,l=None):
    """ retourne le gradient moyen de l'erreur hinge """
    """ a est de memes dimensions que datay : 0 si xw.*y>0, 1 sinon
        X'(-Y.*a)/N """
    a = np.dot(datax,w)*datay
    indices_plus = a<=0
    a[np.logical_not(indices_plus)] = 0
    a[indices_plus] = 1
    #np.dot(datax.T[1:] car datax de dimension (n,d+1) avec le biais)
    if l!=None:
        return np.row_stack((np.reshape(-np.mean(datay),(1,1)),np.dot(datax.T[1:],-datay*a)/len(datax))) + l*np.dot(w[1:].T,w[1:])
    return np.row_stack((np.reshape(-np.mean(datay),(1,1)),np.dot(datax.T[1:],-datay*a)/len(datax)))

class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01,proj=0,desc_type = 0,sigma=None,trainx=None,l=None):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
            :proj: type de proj, 0 pour aucune, 1 pour poly, 2 pour projection gaussienne
            :desc_type: type de descente, 0 pour batch, 1 pour stochastique, 2 pour mini-batch
            :sigma et trainx dans le cas de la projection gaussienne
            :l: parametre de regularisation
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.proj,self.desc_type = proj,desc_type
        self.sigma,self.trainx = sigma,trainx
        self.l = l

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        #on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        
        #on transforme datax selon la projection choisie
        if self.proj == 0:
            #ajout biais : colonne de 1
            datax = np.column_stack((np.ones(datax.shape[0]), datax))
        else:
            if self.proj == 1:
                #projection poly et ajout biais
                datax = projection_poly(datax)
                datax = np.column_stack((np.ones(datax.shape[0]), datax))
            else:
                if self.proj == 2:
                    #projection gaussienne
                    datax = projection_gauss(datax,datax,self.sigma)

        #w : matrice/vecteur (d+1,1) avec biais w[0] = 0
        d = datax.shape[1]
        self.w = (-1+2*np.random.random((d,1)))*self.eps
        self.w[0] = 0
        
        #stockage des w pour retracer toutes les frontieres
        self.list_w = np.zeros((self.max_iter+1,d))
        self.list_w[0] = self.w.T
        
        #descente selon le type choisi
        #batch
        if self.desc_type == 0:
            for i in range(self.max_iter):
                w = self.w -self.eps*self.loss_g(datax,datay,self.w,self.l)
                self.w = w
                self.list_w[i+1] = w.T
        #stochastique
        else:
            if self.desc_type == 1:
                for i in range(self.max_iter):
                    x = np.random.randint(len(datax))
                    if(np.dot(datax[x,:],self.w)*datay[x]<0):
                        w = self.w -self.eps*self.loss_g(datax[x,:].reshape(1,-1),datay[x].reshape(-1,1),self.w)
                        self.w = w
                        self.list_w[i+1] = w.T
                    else:
                        w = self.w
                        self.list_w[i+1] = w.T

        #mini-batch fixe à 20
            else:
                for i in range(self.max_iter):
                    X = np.random.randint(low = 0, high = len(datax), size = (1,20))
                    w = self.w -self.eps*self.loss_g(datax[X,:].reshape(20,d),datay[X].reshape(-1,1),self.w)
                    self.w = w
                    self.list_w[i+1] = w.T
                    
                            
    def predict(self,datax):
        #on transforme datax en vecteur ligne si ce n'est qu'une ligne
        if len(datax.shape) == 1:
            datax = datax.reshape(1,-1)
        
        #on transforme datax selon la projection choisie
        if self.proj == 0:
            # Ajout du biais : colonne de 1
            datax = np.column_stack((np.ones(datax.shape[0]), datax))
        else:
            if self.proj == 1:
                #projection poly et ajout biais
                datax = projection_poly(datax)
                datax = np.column_stack((np.ones(datax.shape[0]), datax))
            else:
                if self.proj == 2:
                    #projection gaussienne
                    datax = projection_gauss(datax,self.trainx,self.sigma)
        return np.where(np.dot(datax,self.w)>=0,1,-1).reshape(1,-1)

    def score(self,datax,datay):
        return np.mean(self.predict(datax) == datay)

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()

def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()

gauss2_train = gen_arti(nbex=1000,data_type=0,epsilon=1)
gauss2_test = gen_arti(nbex=1000,data_type=0,epsilon=1)

if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    plt.ion()
    trainx,trainy =  gauss2_train
    testx,testy =  gauss2_test
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    
    #Entrainement
    max_iter = 1000
    perceptron = Lineaire(hinge,hinge_g,max_iter=max_iter,eps=0.01,desc_type=2)
    debut = time.time()
    perceptron.fit(trainx,trainy)
    print("Entrainement en {} secs".format(time.time()-debut))
    print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))
    
    #Affichage
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plt.show()
    
    #Frontieres        
    x = np.linspace(-4,4,200)
    plt.figure()
    plt.title('Frontieres')
    for i in range(max_iter):
        perceptron.w = perceptron.list_w[i]
        #On laisse de côté les frontieres verticales a cause de la division par 0
        if(perceptron.w[2]!=0):
            y = (-perceptron.w[0] - perceptron.w[1]*x)/perceptron.w[2]
            plt.plot(x,y)
    plt.ylim((-4,4))
    plt.show()
    
    #Trajectoires des poids
    #pour obtenir des valeurs entre -1 et 1
    s = abs(perceptron.list_w).sum(axis=1)
    x = perceptron.list_w[:,1]/s
    y = perceptron.list_w[:,2]/s
    plt.figure()
    plt.title('Trajectoires des poids')
    plt.scatter(x,y,alpha=0.5)
    plt.show()
    
########## Donnees USPS 
# Choix de deux classes, extraction donnees train et test, changement des labels en 1 et -1
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
datay_train[indices_moins] = -1
#donnees test
datax, datay = load_usps ( "USPS_test.txt" )
indices_test = np.logical_or(datay == classe1, datay == classe2)
datax_test = np.copy(datax[indices_test])
datay_test = np.copy(datay[indices_test])
indices_plus = datay_test == classe1 
indices_moins = np.logical_not(indices_plus)
datay_test[indices_plus] = 1
datay_test[indices_moins] = -1

#Apprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.01,desc_type=1)
debut = time.time()
perceptron.fit(datax_train,datay_train)
print("Entrainement usps 2 classes en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(datax_train,datay_train),1-perceptron.score(datax_test,datay_test)))

#Affichage poids
plt.figure()
plt.title('Poids {} vs {}'.format(classe1,classe2))
show_usps(perceptron.w[1:])
plt.show()

#Courbes d'erreur train et test en fonction du nombre d'iterations
max_iter = 100
erreur_train = []
erreur_test = []
x = list(range(max_iter))    
for i in x:
    perceptron.w = perceptron.list_w[i]
    erreur_train.append(1-perceptron.score(datax_train,datay_train))
    erreur_test.append(1-perceptron.score(datax_test,datay_test))
plt.figure()
plt.title('Erreurs {} vs {}'.format(classe1,classe2))
#plt.title('Erreurs {} vs all'.format(classe))
plt.xlabel('iterations')
plt.ylabel('erreur')
plt.plot(x,erreur_train,'r',label='train')
plt.plot(x,erreur_test,'b',label='test')
plt.legend()
plt.show()

# 1 contre tous
# Choix d'une classe, extraction donnees train et test, changement des labels en 1 et -1
classe = 6
datax, datay  = load_usps ( "USPS_train.txt" )
#donnees train
datax_train = np.copy(datax)
datay_train = np.copy(datay)
indices_classe = datay_train == classe
indices_autres = np.logical_not(indices_classe)
datay_train[indices_classe] = 1
datay_train[indices_autres] = -1
#donnees test
datax, datay = load_usps ( "USPS_test.txt" )
datax_test = np.copy(datax)
datay_test = np.copy(datay)
indices_classe = datay_test==classe
indices_autres = np.logical_not(indices_classe)
datay_test[indices_classe] = 1
datay_test[indices_autres] = -1

#Apprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.01,desc_type=1)
debut = time.time()
perceptron.fit(datax_train,datay_train)
print("Entrainement usps {} contre tous en {} secs".format(classe,time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(datax_train,datay_train),1-perceptron.score(datax_test,datay_test)))

#Affichage
plt.title('Poids {} vs all'.format(classe))
show_usps(perceptron.w[1:])
plt.show()


########## Donnees 2D et projections 

gauss4_train = gen_arti(nbex=1000,data_type=1,epsilon=0.5)
gauss4_test = gen_arti(nbex=1000,data_type=1,epsilon=0.5)
echiquier_train = gen_arti(nbex=1000,data_type=2)
echiquier_test = gen_arti(nbex=1000,data_type=2)

# Sans projection
# melange de 4 gaussiennes
plt.ion()
trainx,trainy =  gauss4_train
testx,testy = gauss4_test
plt.figure()
plot_error(trainx,trainy,mse)
plt.figure()
plot_error(trainx,trainy,hinge)

#Apprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,desc_type=2)
debut = time.time()
perceptron.fit(trainx,trainy)
print("Entrainement 4 gaussiennes sans projection en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))

#Affichage
plt.figure()
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
plt.show()

# echiquier
plt.ion()
trainx,trainy =  echiquier_train
testx,testy =  echiquier_test

#apprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
debut = time.time()
perceptron.fit(trainx,trainy)
print("Entrainement echiquier sans projection en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))

#Affichage
plt.figure()
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
plt.show()

#Projection polynomiale
def projection_poly(datax):
    phi = np.zeros((len(datax),6))
    x1 = datax[:,0]
    x2 = datax[:,1]
    phi[:,0] = 1
    phi[:,1] = x1
    phi[:,2] = x2
    phi[:,3] = x1**2
    phi[:,4] = x2**2
    phi[:,5] = x1*x2
    return phi

# melange de 4 gaussiennes
plt.ion()
trainx,trainy =  gauss4_train
testx,testy =  gauss4_test

# Apprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,proj=1,desc_type=1)
debut = time.time()
perceptron.fit(trainx,trainy)
print("Entrainement 4 gaussiennes avec projection polynomiale en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))

#Affichage
plt.figure()
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
plt.show()

# echiquier
plt.ion()
trainx,trainy = echiquier_train
testx,testy =  echiquier_test

# Aprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,proj=1,desc_type=0)
debut = time.time()
perceptron.fit(trainx,trainy)
print("Entrainement echiquier avec projection polynomiale en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))

#Affichage
plt.figure()
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
plt.show()

#Projection gaussienne
def projection_gauss(datax,trainx,sigma):
    n1 = len(datax)
    n2 = len(trainx)
    phi = np.zeros((n1,n2))
    for i in range(n1):
        phi[i,:] = np.exp(-np.sum((datax[i]-trainx)**2,1)/sigma)
    return phi

# melange de 4 gaussiennes
sigma = 0.01
plt.ion()
trainx,trainy =  gauss4_train
testx,testy =  gauss4_test

# apprentissage 
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,proj=2,sigma=sigma,trainx=trainx,desc_type=2,l=0)
debut = time.time()
perceptron.fit(trainx,trainy)
print("Entrainement 4 gaussiennes avec projection gaussienne en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))

#Affichage
plt.figure()
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
plt.title('sigma = {}'.format(sigma))
plt.show()

# échiquier
sigma = 0.01
plt.ion()
trainx,trainy = echiquier_train
testx,testy =  echiquier_test

# apprentissage
perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1,proj=2,sigma=sigma,trainx=trainx,desc_type=1,l=100)
debut = time.time()
perceptron.fit(trainx,trainy)
print("Entrainement echiquier avec projection gaussienne en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(trainx,trainy),1-perceptron.score(testx,testy)))

#Affichage
plt.figure()
plot_frontiere(trainx,perceptron.predict,200)
plot_data(trainx,trainy)
plt.title('sigma = {}'.format(sigma))
plt.show() 

