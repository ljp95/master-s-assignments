%declaration des predicats
pred(on(A,B)) :- block(A),block(B).
pred(ontable(A)) :- block(A).
pred(clear(A)) :- block(A).
pred(empty).
pred(holding(A)) :- block(A).



%declaration des actions

% Action pick up 
action(pickup(A)) :- block(A).

%precondition
pre(pickup(A), clear(A)) :- action(pickup(A)).
pre(pickup(A), ontable(A)) :- action(pickup(A)).
pre(pickup(A), empty) :- action(pickup(A)).

%effect
del(pickup(A),ontable(A)) :- action(pickup(A)).
del(pickup(A),clear(A)) :- action(pickup(A)).
del(pickup(A),empty) :- action(pickup(A)).
add(pickup(A), holding(A)) :- action(pickup(A)).



% Action putdown 
action(putdown(A)):- block(A).

%precondition 
pre(putdown(A), holding(A)) :- action(putdown(A)).

%effect
del(putdown(A), holding(A)) :- action(putdown(A)).
add(putdown(A), ontable(A)) :- action(putdown(A)).
add(putdown(A), clear(A)) :- action(putdown(A)).
add(putdown(A), empty) :- action(putdown(A)).



% Action stack
action(stack(A,B)) :- block(A),block(B).

%precondition 
pre(stack(A,B), clear(B)) :- action(stack(A,B)).
pre(stack(A,B), holding(A)) :- action(stack(A,B)).

%effect
del(stack(A,B), clear(B)) :- action(stack(A,B)).
del(stack(A,B), holding(A)) :- action(stack(A,B)).
add(stack(A,B), on(A,B)) :- action(stack(A,B)).
add(stack(A,B), clear(A)) :- action(stack(A,B)).
add(stack(A,B), empty) :- action(stack(A,B)).



% Action unstack
action(unstack(A,B)) :- block(A),block(B).

%precondition
pre(unstack(A,B), clear(A)) :- action(unstack(A,B)).
pre(unstack(A,B), on(A,B)) :- action(unstack(A,B)).
pre(unstack(A,B), empty) :- action(unstack(A,B)).

%effect
del(unstack(A,B), clear(A)) :- action(unstack(A,B)).
del(unstack(A,B), on(A,B)) :- action(unstack(A,B)).
del(unstack(A,B), empty) :- action(unstack(A,B)).
add(unstack(A,B), clear(B)) :- action(unstack(A,B)).
add(unstack(A,B), holding(A)) :- action(unstack(A,B)).



block(a).
block(b).

init(ontable(a)).
init(on(b,a)).
init(clear(b)).
init(empty).

but(empty).
but(ontable(b)).
but(on(a,b)).
but(clear(a)).


