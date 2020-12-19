
pos(1..5). %position de la maison
maison(rouge;verte;blanche;jaune;bleue).
pers(britannique;suedois;danois;norvegien;allemand).
boisson(the;cafe;lait;biere;eau).
cigarette(pallMall;dunhill;bluemaster;princess;blend).
animal(chien;chat;chevaux;oiseau;poisson).

%% contraintes
1{couleur(M,I):pos(I)}1 :- maison(M).
1{vit(P,M):pers(P)}1 :- maison(M).
1{boit(P,B):pers(P)}1 :- boisson(B).
1{fume(P,C):pers(P)}1 :- cigarette(C).
1{possede(P,A):pers(P)}1 :- animal(A).

1{couleur(M,I):maison(M)}1 :- pos(I).
1{vit(P,M):maison(M)}1 :-pers(P).
1{boit(P,B): boisson(B)}1 :-pers(P).
1{fume(P,C):cigarette(C)}1 :- pers(P).
1{possede(P,A):animal(A)}1 :-pers(P).




%% 1/ Le Britannique vit dans la maison rouge
:-vit(britannique,M),M!=rouge.


%% 2/ Le Suedois garde un chien
:-possede(suedois,A),A!=chien.


%% 3/ Le Danois boit du the
:-boit(danois,B),B!=the.


%% 4/ La maison verte est a gauche de la maison blanche
:-couleur(verte,I1),couleur(blanche,I2),I1!=I2-1.


%% 5/  Le proprietaire de la maison verte boit du cafe
:- vit(P1,verte),boit(P2,cafe),P1!=P2.


%% 6/ La personne qui fume des PallMall possede des oiseaux
:- fume(P1,pallMall),possede(P2,oiseau),P1!=P2.


%% 7/ Le proprietaire de la maison jaune fume des Dunhill
:- vit(P1,jaune), fume(P2,dunhill),P1!=P2.


%% 8/ La personne qui vit dans la maison qui se trouve exactement au centre boit du lait.
:- vit(P,M),couleur(M,I),boit(P,lait),I!=3.


%% 9/ Le Norvegien vit dans la premiere maison
:- vit(norvegien,M),couleur(M,I),I!=1.


%% 10/ La personne qui fume des Blend vit a cote de celle qui possede des chats
:- fume(P1,blend),possede(P2,chat),vit(P1,M1),vit(P2,M2),couleur(M1,I1),couleur(M2,I2),|I1-I2|!=1.


%% 11/ La personne qui possede des chevaux vit a cote du fumeur de Dunhill
:- possede(P1,chevaux),fume(P2,dunhill),vit(P1,M1),vit(P2,M2),couleur(M1,I1),couleur(M2,I2),|I1-I2|!=1.


%% 12/ La personne qui fume des Bluemaster boit de la biere
:- fume(P,bluemaster),boit(P,B),B!=biere.


%% 13/ L'Allemand fume des Princess
:- fume(allemand,C),C!=princess.


%% 14/ Le Norvegien vit a cote de la maison bleue
:- vit(norvegien,M),couleur(M,I1),couleur(bleue,I2),|I1-I2|!=1.


%% 15/ La personne qui fume des Blend a un voisin qui boit de l'eau  
:- fume(P1,blend),boit(P2,eau),vit(P1,M1),vit(P2,M2),couleur(M1,I1),couleur(M2,I2),|I1-I2|!=1.


#show possede/2.

%qui possede le poisson ? 
%l'Allemand



