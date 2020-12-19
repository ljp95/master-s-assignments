#const n=8.
pos(1..n).


%une reine par colonne
1{reine(X,Y):pos(X)}1:-pos(Y).

%une reine par ligne
:-reine(X,Y1),reine(X,Y2),Y1!=Y2.

%contrainte en diagonale
:-reine(X1,Y1),reine(X2,Y2),X1-X2<0,Y1-Y2<0, X2-X1==Y2-Y1.
:-reine(X1,Y1),reine(X2,Y2),X1-X2>0,Y1-Y2>0, X1-X2==Y1-Y2.
:-reine(X1,Y1),reine(X2,Y2),X1-X2<0,Y1-Y2>0, X2-X1==Y1-Y2.
:-reine(X1,Y1),reine(X2,Y2),X1-X2>0,Y1-Y2<0, X1-X2==Y2-Y1.

%une reine en position (1,1)
%reine(1,1).


%SOLUTIONS

%Answer: 1
%pos(1) pos(2) pos(3) pos(4) pos(5) pos(6) pos(7) pos(8) reine(1,1) reine(2,7) reine(3,5) reine(4,8) reine(5,2) reine(6,4) reine(7,6) reine(8,3)
%Answer: 2
%pos(1) pos(2) pos(3) pos(4) pos(5) pos(6) pos(7) pos(8) reine(1,1) reine(2,6) reine(3,8) reine(4,3) reine(5,7) reine(6,4) reine(7,2) reine(8,5)
%Answer: 3
%pos(1) pos(2) pos(3) pos(4) pos(5) pos(6) pos(7) pos(8) reine(1,1) reine(2,5) reine(3,8) reine(4,6) reine(5,3) reine(6,7) reine(7,2) reine(8,4)
%Answer: 4
%pos(1) pos(2) pos(3) pos(4) pos(5) pos(6) pos(7) pos(8) reine(1,1) reine(2,7) reine(3,4) reine(4,6) reine(5,8) reine(6,2) reine(7,5) reine(8,3)
%SATISFIABLE



%aucune reine aux coordoonées (4,4) de l'echiquier
:-reine(4,4).
%ou not reine(4,4).

% 84 SOLUTIONS si on ne met pas de reine en position (1,1)

