connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

valid_move(X,Y) :-
    connected(X,Y).
valid_move(X,Y) :- connected(X,Z), valid_move(Z,Y).
wet_road.
slippery :-
	wet_road.
reduce_speed :-
	slippery.
