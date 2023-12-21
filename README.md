# bdia
TP Centrale Nantes INFOIA GPGPU

Comment compiler : gcc *.c -lm

1. Introduction (enjeu TP)
2. Analyse du code séquentiel (prototype et algorithme des fonctions)
3. Performances du code séquentiel (nvprof, etc.)
    --> identifier les parties du code sur lesquelles travailler
4. Expliquer la stratégie d'optimisation adoptée
    --> Modification fonction par fonction avec cuBLAS : utilisation d'une librairie
    --> Modification fonction par fonction avec cuda : réécriture bas-niveau
5. Codage
    5.a. Rédaction des transferts mémoire (mémoire unifiée)
    5.b. Rédaction des prototypes nouvelles fonctions (entête + docstring + tests)
    5.c. Rédaction des nouvelles fonctions
    5.d. Passage des tests, débogage
6. Présenter et expliquer chaque modification apportée
7. Performances du code optimisé pour chaque stratégie
    --> analyser l'impact de chaque modification sur les performances globales
8. Performances comparées des différentes stratégies entre elles et avec le code séquentiel
9. Conclusion

# Réflexion au fur et à mesure du projet :

## Utiliser la mémoire unifiée
Cela permet de ne pas se faire chier avec les allocations de mémoire de partout