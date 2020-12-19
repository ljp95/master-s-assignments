(define (domain exercice3)
	(:requirements :strips :typing)
	(:types block - object
		support - object)
	(:constants table - support)

	(:predicates
		(on ?x - block ?y - object)
		(clear ?x - object))

	(:action moveTo 
		;;; action qui prend un bloc X qui est sur Y et le met sur Z
		:parameters (?x - block ?y - object ?z - block)
		:precondition (and (on ?x ?y) (clear ?x) (clear ?z))
		:effect (and (not (on ?x ?y))
			(not (clear ?z))
			(clear ?y)
			(on ?x ?z)))

	(:action moveToTable
		;;; action qui prend un bloc X qui est sur Y pour le mettre sur la table, X et Y sont tous les deux des blocs
		:parameters (?x - block ?y - block)
		:precondition (and (on ?x ?y) (clear ?x))
		:effect (and (not (on ?x ?y))
			(clear ?y)
			(on ?x table)))
	
)

