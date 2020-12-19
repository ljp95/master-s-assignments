%position dans la grille [0,8]
#const d=8.
pos(0..d).

%les nombres dans la grille entre [1,9]
#const n=9.
num(1..n).


%on initialise la grille avec un chiffre dans chaque position
1{sudoku(X,Y,N):num(N)}1:-pos(X),pos(Y).

%chaque ligne ne doit contenir qu'une seule fois tous les chiffres
:-sudoku(X,Y1,N),sudoku(X,Y2,N), Y1!=Y2.

%chaque colonne ne doit contenir qu'une seule fois tous les chiffres
:-sudoku(X1,Y,N),sudoku(X2,Y,N), X1!=X2.

%chaque region ne doit contenir qu'une seule fois tous les chiffres
:-sudoku(X1,Y1,N),sudoku(X2,Y2,N),(X1/3)==(X2/3),(Y1/3)==(Y2/3),X1!=X2,Y1!=Y2.



