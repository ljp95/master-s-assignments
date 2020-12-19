(define (domain exSinge)
	(:requirements :strips :typing)
	(:types animal - object
		obj - object
		place 
		level)
	(:predicates
		(situe ?x - object ?y - place)
		(niveau ?x - object ?y - level)
		(possede ?x - animal  ?y - obj)
		(empty))
	(:constants singe - animal
		   bananes - obj
		   caisse - obj
		   a - place
		   b - place 
		   c - place
		   bas - level
		   haut -level)

	(:action seDeplace
		;;; le singe se deplace de l'emplacement X a l'emplacement Y
		:parameters (?x - place ?y - place)
		:precondition (and (situe singe ?x) (niveau singe bas))
		:effect (and (not (situe singe ?x))
			     (situe singe ?y)))

	(:action prend
		;;; le singe prend l'objet X. Attention le singe ne peut prendre qu'un objet a la fois
		:parameters (?x - obj)
		:vars (?p - place ?l - level)
		:precondition (and (empty)
				(situe ?x ?p)
				(situe singe ?p)
				(niveau ?x ?l)
				(niveau singe ?l))
		:effect (and (not (empty))
			(not (situe ?x ?p))
			(not (niveau ?x ?l))
			(possede singe ?x)))

	(:action depose
		;;; le singe depose l'objet X
		:parameters (?x - obj)
		:vars (?p - place ?l - level)
		:precondition (and (possede singe ?x)
				(situe singe ?p)
				(niveau singe ?l))
		:effect (and (not (possede singe ?x))
			(situe ?x ?p)
			(niveau ?x ?l)
			(empty)))

	(:action monteCaisse
		;;; le singe monte sur la caisse 
		:vars (?p - place)
		:precondition (and (situe singe ?p)
				(niveau singe bas)
				(situe caisse ?p)
				(niveau caisse bas)
				(empty))
		:effect (and (not (niveau singe bas))
			(niveau singe haut)))


)


;;; contrainte var c'est comme parametre mais ce ne sont pas des arguments, ce sont des variables locales
;;; pas obligatoire d'utiliser var
