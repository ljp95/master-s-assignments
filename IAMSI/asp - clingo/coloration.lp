%nombre de couleur
#const n=4.
c(1..n).

%nombre de noeuds
#const m=7.
noeud(1..m).


%une couleur pour chaque noeud
1{couleur(X,I):c(I)}1:-noeud(X).

%contrainte couleur differents pour des noeuds adjacents
:- couleur(X,I), couleur(Y,I), adjacent(X,Y), c(I).



