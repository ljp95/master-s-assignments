from sklearn.tree import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from arftools import *
import numpy as np
import matplotlib.pyplot as plt
import time

######arbre de decision
#dt = tree.DecisionTreeClassifier(criterion='entropy',max_depth=k)
#dt.fit(datax,datay)
#print(dt.score(datax,datay))

######KNN
#knn = KNeighborsRegressor(n_neighbors = 5,weights='distance',p=2)
#knn.fit(data,notes)
#predictions = knn.predict(data_test)

######Donnees
gauss2_train = gen_arti(nbex=1000,data_type=0,epsilon=1)
gauss2_test =  gen_arti(nbex=1000,data_type=0,epsilon=1)
gauss4_train = gen_arti(nbex=1000,data_type=1,epsilon=0.5)
gauss4_test =  gen_arti(nbex=1000,data_type=1,epsilon=0.5)
echiquier_train = gen_arti(nbex=1000,data_type=2)
echiquier_test =  gen_arti(nbex=1000,data_type=2)

######perceptron
#choix données
trainx,trainy = gauss2_train
testx,testy =  gauss2_test

trainx,trainy = gauss4_train
testx,testy =  gauss4_test

trainx,trainy = echiquier_train
testx,testy =  echiquier_test

#apprentissage
clf = Perceptron(max_iter=1000,tol=0.001)
debut = time.time()
clf.fit(trainx,trainy)
print("Entrainement en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-clf.score(trainx,trainy),1-clf.score(testx,testy)))

#affichage
plt.figure()
plot_frontiere(trainx,clf.predict,200)
plot_data(trainx,trainy)
plt.show()

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()

######donnees usps
# choix de deux classes, extraction donnees train et test, changement des labels en 1 et -1
classe1 = 6
classe2 = 9
#donnees train
trainx,trainy  = load_usps ( "USPS_train.txt" )
indices_train = np.logical_or(trainy == classe1, trainy == classe2)
datax_train = trainx[indices_train]
datay_train = trainy[indices_train]
datay_train[datay_train == classe1 ] = 1
datay_train[datay_train == classe2 ] = -1
#donnees test
testx,testy = load_usps ( "USPS_test.txt" )
indices_test = np.logical_or(testy == classe1, testy == classe2)
datax_test = testx[indices_test]
datay_test = testy[indices_test]
datay_test[datay_test == classe1] = 1
datay_test[datay_test == classe2] = -1
#apprentissage
perceptron = Perceptron(max_iter=1000,tol=0.001)
debut = time.time()
perceptron.fit(datax_train,datay_train)
print("Entrainement usps {} vs {} en {} secs".format(classe1,classe2,time.time()-debut))
print("Erreur : train %f, test %f"% (1-perceptron.score(datax_train,datay_train),1-perceptron.score(datax_test,datay_test)))

#affichage poids
plt.figure()
plt.title('Poids {} vs {}'.format(classe1,classe2))
show_usps(perceptron.coef_[0])
plt.show()

#####SVM
def plot_frontiere_proba(data,f,step=20):
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),255)
    
#choix donnees
trainx,trainy = gauss2_train
testx,testy =  gauss2_test

trainx,trainy = gauss4_train
testx,testy =  gauss4_test

trainx,trainy = echiquier_train
testx,testy =  echiquier_test

#exemple avec un choix de noyaux
gamma = 10
clf = svm.SVC(probability=True,kernel='linear')
clf = svm.SVC(probability=True,kernel='poly',degree=3)
clf = svm.SVC(probability=True,kernel='rbf',gamma=gamma)

#apprentissage
debut = time.time()
clf.fit(trainx,trainy)
print("Entrainement en {} secs".format(time.time()-debut))
print("Erreur : train %f, test %f"% (1-clf.score(trainx,trainy),1-clf.score(testx,testy)))

#affichage
plt.figure()
plt.title('gamma : {}'.format(gamma))
plot_frontiere(trainx,clf.predict,200)
plot_data(trainx,trainy)
plt.show()
plt.figure()
plt.title('gamma : {}'.format(gamma))
plot_frontiere_proba(trainx,lambda x:clf.predict_proba(x)[:,0],step=50)
plt.colorbar()
plt.show()
#print(clf.support_vectors_)
print('nombre de supports : {}'.format(clf.n_support_))
#print(clf.dual_coef_)


######grid search noyau gaussien et choix des valeurs du parametre gamma
x = [1,3,5,10,20,30,40,50,60,70,80,90,100]
parametres = {'gamma': x}

#apprentissage
splits = [2,3,5]
erreur_test = []
erreur_train = []
for i in range(len(splits)):
    grid = GridSearchCV(svm.SVC(probability=True,kernel='rbf'),parametres,cv=splits[i])
    grid.fit(trainx,trainy)
    stockage = grid.cv_results_
    erreur_test.append(1-stockage['mean_test_score'])
    erreur_train.append(1-stockage['mean_train_score'])
  
#affichage
plt.figure()
#plt.title('erreur gauss sur 2 gaussiennes')
#plt.title('erreur gauss sur 4 gaussiennes')
plt.title('erreur gauss sur echiquier')
for i in range(len(splits)):
    plt.plot(x,erreur_test[i],label='test split {}'.format(splits[i]))
    plt.plot(x,erreur_train[i],label='train split {}'.format(splits[i]))
    plt.scatter(x,erreur_test[i])
    plt.scatter(x,erreur_train[i])
plt.xlabel('valeur parametre')
plt.ylabel('erreur')
plt.legend()
plt.show()


######apprentissage multi classe 
#donnees
trainx, trainy = load_usps ( "USPS_train.txt" )
testx, testy = load_usps ( "USPS_test.txt" )

######one versus one
#choix noyau
clf = svm.SVC(probability=True,kernel='linear',decision_function_shape='ovo')
clf = svm.SVC(probability=True,kernel='poly',degree=2,decision_function_shape='ovo')

#apprentissage 
debut = time.time()
clf.fit(trainx,trainy)
print("Entrainement en {} secs".format(time.time()-debut))
print("Score poly : train %f, test %f"% (clf.score(trainx,trainy),clf.score(testx,testy)))

######implementation manuelle
classes = list(range(10))
mat_poids = np.zeros((10,10,256))

#Apprentissage des k(k-1)/2 classifieurs
for i in range(10):
    for j in range(i+1,10):
        #donnees train
        indices_train = np.logical_or(trainy == classes[i], trainy == classes[j])
        datax_train = np.copy(trainx[indices_train])
        datay_train = np.copy(trainy[indices_train])
        indices_plus = datay_train == classes[i]
        indices_moins = np.logical_not(indices_plus)
        datay_train[indices_plus] = 1
        datay_train[indices_moins] = -1
        
        #donnees test
        indices_test = np.logical_or(testy == classes[i], testy == classes[j])
        datax_test = np.copy(testx[indices_test])
        datay_test = np.copy(testy[indices_test])
        indices_plus = datay_test == classes[i] 
        indices_moins = np.logical_not(indices_plus)
        datay_test[indices_plus] = 1
        datay_test[indices_moins] = -1
        
        clf= Perceptron(tol=0.001,max_iter=1000)
#        clf = LogisticRegression(max_iter=1000)
#        clf = svm.SVC(probability=True,kernel='linear')
        
        debut = time.time()
        clf.fit(datax_train,datay_train)
        print("Entrainement usps {} vs {} en {} secs".format(i,j,time.time()-debut))
        print("Erreur : train %f, test %f"% (1-clf.score(datax_train,datay_train),1-clf.score(datax_test,datay_test)))
        w = clf.coef_
        mat_poids[i,j] = w

#Prediction sur toute la base train et test
#datax = trainx
#datay = trainy

datax = testx
datay = testy

#prediction 
score = 0
stock = []
for k in range(len(datax)):
    data = datax[k]
    mat_votes = np.zeros(10)
    for i in range(10):
        for j in range(i+1,10):
            w = mat_poids[i,j]
            tmp = np.dot(data,w)
            if(tmp>=0):
#            if(tmp>=0.5):
                tmp = 1
            else:
                tmp = 0
            mat_votes[i] += tmp
            mat_votes[j] += 1-tmp
    prediction = mat_votes.argmax()
    score += prediction == datay[k]
    if(prediction!=datay[k]):
        stock.append(k)
        
#score_train = score/len(trainx)
#print("Score train {}".format(score_train))

score_test = score/len(testx)
print("Score test {}".format(score_test))

plt.figure()
plt.title('Repartition des mal classes (1vs1 test)')
plt.hist(datay[stock])
plt.xlabel('Chiffre')
plt.ylabel('nombre occurence')
plt.show()

#un exemple
number = 320
data = datax[number]
plt.figure()
show_usps(data)
plt.show()
mat_votes = np.zeros(10)
for i in range(10):
    for j in range(i+1,10):
        w = mat_poids[i,j]
        tmp = np.dot(data,w)
        if(tmp>=0):
#        if(tmp>=0.5):
            tmp = 1
        else:
            tmp = 0
        mat_votes[i] += tmp
        mat_votes[j] += 1-tmp
print('label : {}'.format(datay[number]))
print('votes : {}'.format(mat_votes))
print('prediction : {}'.format(mat_votes.argmax()))

#####one versus all
clf = svm.SVC(probability=True,kernel='linear',decision_function_shape='ovr')
clf = svm.SVC(probability=True,kernel='poly',degree=2,decision_function_shape='ovo')

debut = time.time()
clf.fit(trainx,trainy)
print("Entrainement en {} secs".format(time.time()-debut))
print("Score poly : train %f, test %f"% (clf.score(trainx,trainy),clf.score(testx,testy)))

#implementation manuelle
classes = list(range(10))
#mat_poids = np.zeros((10,257))
mat_poids = np.zeros((10,256))

#Apprentisage des 10 classifieurs
for i in range(10):
    indices_classe = trainy == classes[i]
    indices_autres = np.logical_not(indices_classe)
    datax_train = np.copy(trainx)
    datay_train = np.copy(trainy)
    datay_train[indices_classe] = 1
    datay_train[indices_autres] = -1
    
    indices_classe = testy == classes[i]
    indices_autres = np.logical_not(indices_classe)
    datax_test = np.copy(testx)
    datay_test = np.copy(testy)
    datay_test[indices_classe] = 1
    datay_test[indices_autres] = -1

    clf = Perceptron(max_iter=1000,tol=0.001)
#    clf = LogisticRegression(max_iter=1000)
#    clf = svm.SVC(probability=True,kernel='linear')
    
    debut = time.time()
    clf.fit(datax_train,datay_train)
    print("Entrainement usps {} vs all en {} secs".format(classes[i],time.time()-debut))
    print("Erreur : train %f, test %f"% (1-clf.score(datax_train,datay_train),1-clf.score(datax_test,datay_test)))
    w = clf.coef_
    mat_poids[i] = w

#Prediction sur toute la base train et test
#datax = trainx
#datay = trainy

datax = testx
datay = testy

score = 0
stock = []
#prediction 
for k in range(len(datax)):
    data = datax[k]
    mat_votes = np.zeros(10)
    for i in range(10):
        w = mat_poids[i]
        mat_votes[i] = np.dot(data,w)
    prediction = mat_votes.argmax()
    score += prediction == datay[k]
    if(prediction!=datay[k]):
        stock.append(k)

#score_train = score/len(trainx)
#print("Score train {}".format(score_train))

score_test = score/len(testx)
print("Score test {}".format(score_test))

plt.figure()
plt.title('Repartition des mal classes (1vs all train)')
plt.hist(datay[stock])
plt.xlabel('Chiffre')
plt.ylabel('nombre occurence')
plt.show()

#Un exemple
number = 10
data = datax[number]
plt.figure()
show_usps(data)
plt.show()

mat_votes = np.zeros(10)
for i in range(10):
    w = mat_poids[i]
    mat_votes[i] = np.dot(data,w)
print('label : {}'.format(datay[number]))
print('votes : {}'.format(np.round(mat_votes,0)))
print('prediction : {}'.format(mat_votes.argmax()))

########## String Kernel ##################
import itertools as iter
import math
import numpy as np

# trouver tous les indices d'une lettre dans un mot
def find_char_indices(char,mot):
    return [ind for ind, ltr in enumerate(mot) if ltr==char]

# trouver l'indice des lettres du sous mot dans le mot
def find_sequence_indices(sous_mot, mot):
    list_ind_char = [find_char_indices(ltr,mot) for ltr in sous_mot]
    
    #trouver toutes les sous-sequences d'indice
    def all_subsequence_indices(list_ind_char,borne_ind=-1):
        if not list_ind_char:
            return [[]]
        return [[idx]+suffix for idx in list_ind_char[0] for suffix in all_subsequence_indices(list_ind_char[1:],idx) if idx> borne_ind ]
    return all_subsequence_indices(list_ind_char)
    
def Kernel(s,t,n,l):
    K=0
    liste_mot = [i for i in iter.combinations(s,n)]
    tmp = [i for i in iter.combinations(t,n)]
    liste_mot += tmp
    
    liste_mot = list(set(liste_mot))
    
    for u in liste_mot:
        for i in find_sequence_indices(u,s):
            for j in find_sequence_indices(u,t):
                if(i!=[] and j!= []):
                    li = i[-1]-i[0]+1
                    lj = j[-1]-j[0]+1
                    K += math.pow(l,(li+lj))
    return K

s='alphabeta'
t='leopard'
#s = 'car'
#t = 'cat'
n=3
l=1
print(Kernel(s,t,n,l))
       
    
text1="Programmer un string kernel visualiser la matrice de similarité sur des exemples de textes de différents auteurs puis tester l’apprentissage"
text2="Pour les différents noyaux et différents nombre d’exemples d’apprentissage opérer un grid search afin de trouver les paramètres optimaux" 

text1=text1.split(' ')
text2=text2.split(' ')
mat=np.zeros((len(text1),len(text2)))

for i in range(len(text1)):
    for j in range(len(text2)):
        mat[i,j] = Kernel(text1[i],text2[j],min(len(text1[i]),len(text2[j]))//2,1)
print(mat)
