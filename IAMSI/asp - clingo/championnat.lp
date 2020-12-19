
#const n=4. %nombre d'equipes
#const m=3. %nombre de semaines au maximum
#const j=2*m. %nombre de jours

eq(1..n).
sem(1..m).

jour(merc;dim).


%tous les matchs possibles du championnat
1{match(X,Y,J,S):jour(J),sem(S)}1:-eq(X),eq(Y),X!=Y.

% une equipe ne peut pas jouer a l'exterieur et a domicile le meme jour
:-match(X,_,J,S),match(_,X,J,S).

% une equipe ne peut jouer deux matchs a domicile le meme jour
:-match(X,Y,J,S),match(X,Y1,J,S),Y!=Y1.

% une equipe ne peut jouer deux matchs a l'exterieur le meme jour
:-match(X,Y,J,S),match(X1,Y,J,S),X!=X1.


#show match/4.










