#const n=4. 

time(0..n).

% Etat Initial 
holds(P,0) :- pred(P), init(P).


% Precondition
:- action(A), time(T), perform(A,T), pre(A,P), not holds(P,T).


% Effets positifs
holds(P,T+1) :- action(A), time(T), perform(A,T), pred(P), add(A,P).


%Inertie et effets negatifs
holds(P,T+1) :- holds(P,T), time(T),perform(A,T),not del (A,P), pred(P), action(A).


% Choix d'action
1{perform(A,T):action(A)}1 :- time(T), T!=n.


% Specification du but
:- pred(P), but(P), not holds(P,n). 

#show perform/2.

