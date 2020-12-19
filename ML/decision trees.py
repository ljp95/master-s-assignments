from collections import Counter
import numpy as np
import pickle
import matplotlib.pyplot as plt

########## ENTROPIE
def entropie(vect):
    occ = np.array(list(Counter(vect).values()))/len(vect)
    return -np.sum(occ*np.log2(occ))

def entropie_cond(list_vect):
    h = 0.
    card = 0
    for vect in list_vect:
        h += len(vect)*entropie(vect)
        card += len(vect)
    return h/card

#####Base de données IMDB
# data : tableau (films , features ), id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[data,id2titles,fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la derniere colonne est le vote
datax = data [: ,:32]
datay = np.array ([1 if x [33] >6.5 else -1 for x in data ])

#####Q3)Calculs gain d'information
#lignes d'entropie, d'entropie conditionnelle et gain d'information
#H(X),H(X|Y),G(X,Y) = H(X)-H(X|Y)
l_entxy = np.zeros((3,28)) 
l_p = datay==1
l_m = datay==-1
for i in range(28):
    l_entxy[0,i] = entropie(datax[:,i]) #entropie
    l_entxy[1,i] = entropie_cond([datax[l_p,i],datax[l_m,i]]) #entropie conditionnelle
l_entxy[2] = l_entxy[0]-l_entxy[1] #gain d'information
print("Entropie X")
print(abs(l_entxy[0].round(2)))
print("Entropie conditionelle X|Y")
print(l_entxy[1].round(2))
print("Gain d'information")
print(abs(l_entxy[2].round(2)))

#H(Y), H(Y|X), G(X,Y)=H(Y)-H(Y|X)
enty = entropie(datay)
l_entyx = np.zeros((2,28)) 
for i in range(28):
    l_p = datax[:,i]==1
    l_m = datax[:,i]==0
    l_entyx[0,i] = entropie_cond([datay[l_p],datay[l_m]])
l_entyx[1,:] = enty-l_entyx[0]
print("Entropie Y")
print(enty.round(2))
print("Entropie conditionnelle Y|X")
print(l_entyx[0].round(2))
print("Gain d'information")
print(l_entyx[1].round(2))


########## QUELQUES EXPERIENCES PRELIMINAIRES
from decisiontree import DecisionTree
dt = DecisionTree ()
dt. max_depth = 4 #on fixe la taille de l’arbre 
dt. min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud
dt. fit(datax , datay )
dt. predict ( datax [:5 ,:])
print (dt. score (datax , datay ))
# dessine l’arbre dans un fichier pdf si pydot est installe .
dt. to_pdf ("tree4.pdf",fields )
# sinon utiliser http :// www. webgraphviz .com/
#dt. to_dot ( fields )
#ou dans la console
#print (dt. print_tree ( fields ))

#####Q5)Scores selon profondeur
scores =  []
for k in range(1,15):
    dt = DecisionTree ()
    dt. max_depth = k #on fixe la taille de l’arbre 
    dt. min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud
    dt. fit(datax , datay )
    dt. predict ( datax [:5 ,:])
    scores.append(dt. score (datax , datay ))
#Construction de la courbe
x = list(range(1,15))
plt.figure()
plt.title('Score selon profondeur')
plt.plot(x,scores,'r--')
plt.xlabel('Profondeur')
plt.ylabel('Score')
plt.legend()
plt.show()


########## SUR ET SOUS APPRENTISSAGE
######Q7
np.random.shuffle(data)
datax = data [: ,:32]
datay = np.array ([1 if x[33] >6.5  else  -1 for x in data])
taux = [0.2,0.5,0.8]
erreurs_app =  []
erreurs_test =  []
for i in range(len(taux)):
    #Separation des donnees selon taux
    s = int(taux[i]*len(datax))
    datax_app = datax[0:s,:]
    datay_app = datay[0:s]
    datax_test = datax[s:-1,:]
    datay_test = datay[s:-1]
    app = []
    test = []
    #Variation du score en fonction de la profondeur
    for k in range(1,15):
        dt = DecisionTree ()
        dt. max_depth = k #on fixe la taille de l’arbre 
        dt. min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud
        dt. fit(datax_app , datay_app ) # base d'apprentissage
        dt. predict ( datax[:5 ,:]) # prediction de toute la base
        app.append(1-dt.score(datax_app , datay_app )) #erreur d'apprentissage
        test.append(1-dt.score(datax_test, datay_test)) #erreur de test
    erreurs_app.append(app)
    erreurs_test.append(test)
#Construction des courbes
x = list(range(1,15))
plt.figure()
plt.title('Erreur selon profondeur et repartition')
plt.plot(x,erreurs_app[0],'b--',label='App 0.2')
plt.plot(x,erreurs_test[0],'b--',label='Test 0.2')
plt.plot(x,erreurs_app[1],'r--',label='App 0.5')
plt.plot(x,erreurs_test[1],'r--',label='Test 0.5')
plt.plot(x,erreurs_app[2],'g--',label='App 0.8')
plt.plot(x,erreurs_test[2],'g--',label='Test 0.8')
plt.xlabel('Profondeur')
plt.ylabel('Erreur')
plt.legend()
plt.show()


########## VALIDATION CROISEE
np.random.shuffle(data)
datax = data[:,:32]
datay = np.array([1 if x[33]>6.5 else -1 for x in data])
l = list(range(len(datax)))
erreurs_app = np.zeros((3,14))  
erreurs_test = np.zeros((3,14))

n = 5 #changer selon la repartition voulue
pas = int(len(datax)/n)
for i in range(n):
    #Separation des donnees en Ei et Eapp/Ei
    l_test = list(range(i*pas,(i+1)*pas))
    l_app = list(set(l)-set(l_test))
    datax_app = datax[l_test]
    datay_app = datay[l_test]
    datax_test = datax[l_app]
    datay_test = datay[l_app]
    #Variation du score en fonction de la profondeur
    for k in range(1,15):
        dt = DecisionTree ()
        dt. max_depth = k #on fixe la taille de l’arbre 
        dt. min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud
        dt. fit(datax_app , datay_app ) # base d'apprentissage
        dt. predict ( datax[:5 ,:]) # prediction de toute la base
        erreurs_app[0][k-1] += (1-dt.score(datax_app , datay_app )) #erreur d'apprentissage
        erreurs_test[0][k-1] += (1-dt.score(datax_test, datay_test)) #erreur de test
erreurs_app[0]/=n
erreurs_test[0]/=n
    
#Construction des courbes
x = list(range(1,15))
plt.figure()
plt.title('Validation croisee')
plt.plot(x,erreurs_app[0],'b--',label='App 0.2')
plt.plot(x,erreurs_test[0],'b--',label='Test 0.2')
plt.plot(x,erreurs_app[1],'r--',label='App 0.5')
plt.plot(x,erreurs_test[1],'r--',label='Test 0.5')
plt.plot(x,erreurs_app[2],'g--',label='App 0.8')
plt.plot(x,erreurs_test[2],'g--',label='Test 0.8')
plt.xlabel('Profondeur')
plt.ylabel('Erreur')
plt.legend()
plt.show()




