(define (problem exercice3Problem)
	(:domain exercice3)
	(:objects A B - block)

	(:init (on B A)
		(clear B)
		(on A table))
	(:goal (and (on A B)
		(clear A)
		(on B table))))	
