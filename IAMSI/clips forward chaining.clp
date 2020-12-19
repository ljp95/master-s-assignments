;;; IAMSI 2018 : séance TME 3



;; Faits initiaux
(defrule my_init
	(initial-fact)
=>
	(watch facts)
	(watch rules)

	(assert (taches_rouges))
	(assert (peu_bouton))
	(assert(sensation_froid))
	(assert(forte_fievre))
	(assert(mal_yeux))
	(assert(amydales_rouges))
	(assert(peau_pele))
	(assert(peau_seche))
)




; si le sujet a peu ou bcp de boutons on dit qu'il a comme symptome une eruption cutanee
(defrule R1
	(or (peu_bouton)
	    (bcp_bouton))
=>
	(assert(erruption_cutanee))
)


;on dit que le sujet a un exantheme s'il a des eruptions cutanee ou des rougeurs
;si eruption cutanne ou rougeur alors exantheme
(defrule R2
	(or (erruption_cutanee)
	    (rougeur))
=>
	(assert (exantheme))
)

;un sujet est dans un etat febrile si forte fievre ou sensation froid
(defrule R3 
	(or (forte_fievre)
	    (sensation_froid))
=>
	(assert(etat_febrile))
)


;si amydales_rouges, taches_rouge et peau qui pele donc signe suspect
(defrule R4
	(and (amydales_rouges)
	     (taches_rouges)
	     (peau_pele))
=> 
	(assert (signe_suspect))
)


;si etat febrile et yeux douleureux et exantheme, ou forte fievre et signe suspect alors rougeole
(defrule R5 
	(or (and (etat_febrile)
	         (yeux_douleureux)
	         (exantheme))
	    (and (forte_fievre)
	         (signe_suspect)))
=>
	(assert(rougeole))
)

;si peu_fievre et peu_bouton alors pas rougeole
(defrule R6
	(and (peu_fievre)
	     (peu_bouton))
=>
	(retract rougeole)
)


;si yeux_douleureux ou dos_douleureux alors douleur
(defrule R7
	(or (yeux_douleureux)
	    (dos_douleureux))
=>
	(assert (douleur))
)


;si dos douleureux et etat febrile alors grippe
(defrule R8
	(and (dos_douleureux)
	     (etat_febrile))
=>
	(assert (grippe))
)


;si pas rougeole alors rubéole et varicelle 
(defrule R9
	(not(rougeole))
=> 
	(assert (rubeole))
	(assert (varicelle))
)


;si forte demangeaisons et pustules alors varicelle
(defrule R10
	(and (forte_demangeaison)
	     (pustules))
=>
	(assert (varicelle))
)


;si peau seche, inflammation des ganglions mais ni pustules ni sensation_froid alors rubeole
(defrule R11
	(and (peau_seche)
	     (inflammation_ganglions)
	     (not(postules))
	     (not(sensation_froid)))
=>
	(assert(rubeole))
)





FIRE    1 R9: f-0,
==> f-1     (rubeole)
==> f-2     (varicelle)
FIRE    2 my_init: f-0
==> f-3     (taches_rouges)
==> f-4     (peu_bouton)
==> f-5     (sensation_froid)
==> f-6     (forte_fievre)
==> f-7     (mal_yeux)
==> f-8     (amydales_rouges)
==> f-9     (peau_pele)
==> f-10    (peau_seche)
FIRE    3 R4: f-8,f-3,f-9
==> f-11    (signe_suspect)
FIRE    4 R5: f-6,f-11
==> f-12    (rougeole)
FIRE    5 R3: f-6
==> f-13    (etat_febrile)
FIRE    6 R3: f-5
FIRE    7 R1: f-4
==> f-14    (erruption_cutanee)
FIRE    8 R2: f-14
==> f-15    (exantheme)

;la maladie trouvée est l'exantheme 
;suite des regles declenchée {R9,R4,R5,R3,R1,R2}
;faits ajoutée (rubeole, varicelle, (init), signe_suspect, rougeole, etat_febrile, erruption_cutanee, exantheme)














