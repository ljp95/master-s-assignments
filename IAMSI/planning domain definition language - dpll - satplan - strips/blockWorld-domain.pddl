(define (domain blockWorld)
	(:requirements :strips :typing)
	(:types block)
	(:predicates
		(on ?x - block ?y - block)
		(ontable ?x - block)
		(clear ?x - block)
		(empty)
		(holding ?x - block))
	(:action pickup
		;;; action qui ramasse un bloc posé sur la table
		:parameters (?x - block)
		:precondition (and (clear ?x) (ontable ?x) (empty))
		:effect (and (not (ontable ?x))
			(not (clear ?x))
			(not (empty))
			(holding ?x)))
	(:action putdown
		;;; action qui pose un bloc sur la table
		:parameters (?x - block)
		:precondition (holding ?x)
		:effect (and (not (holding ?x))
			(ontable ?x)
			(clear ?x)
			(empty)))
	(:action stack
		;;; action qui met un bloc sur un autre bloc
		:parameters (?x - block ?y - block)
		:precondition (and (clear ?y) (holding ?x))
		:effect (and (not (clear ?y))
			(not (holding ?x))
			(on ?x ?y)
			(clear ?x)
			(empty)))
	(:action unstack
		;;; action qui enleve un bloc d'un autre
		:parameters (?x -block ?y - block)
		:precondition ( and (on ?x ?y) (clear ?x) (empty))
		:effect (and (not (on ?x ?y))
			(not (clear ?y))
			(not (empty))
			(clear ?y)	
			(holding ?x)))

)
