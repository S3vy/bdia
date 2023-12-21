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

# Performances du code séquentiel

## Résultats du profilage avec gprof

  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
 91.43    259.75   259.75   183681     0.00     0.00  matrix_dot
  2.37    266.49     6.75    93725     0.00     0.00  matrix_minus
  2.33    273.12     6.63    74980     0.00     0.00  matrix_scalar
  1.85    278.36     5.24    56235     0.00     0.00  matrix_transpose
  1.76    283.35     4.99    22489     0.00     0.00  populate_minibatch
  0.08    283.57     0.22    82468     0.00     0.00  matrix_function
  0.07    283.78     0.21 38386560     0.00     0.00  sigmoid
  0.07    283.99     0.21    44978     0.00     0.00  matrix_sum
  0.07    284.18     0.19    37490     0.00     0.00  hadamard_product
  0.02    284.24     0.06 11996800     0.00     0.00  dsigmoid
  0.01    284.27     0.03    18745     0.00     0.01  backward
  0.01    284.29     0.02   359890     0.00     0.00  alloc_matrix
  0.00    284.30     0.01    23820     0.00     0.00  normalRand
  0.00    284.31     0.01    22489     0.00     0.01  forward
  0.00    284.32     0.01        5     0.00     0.00  shuffle
  0.00    284.32     0.01   359875     0.00     0.00  destroy_matrix
  0.00    284.33     0.01        6     0.00     3.68  accuracy
  0.00    284.33     0.00       11     0.00     0.00  zero_to_n
  0.00    284.33     0.00        4     0.00     0.00  make_uint32
  0.00    284.33     0.00        3     0.00     0.00  create_layer
  0.00    284.33     0.00        2     0.00     0.01  init_weight
  0.00    284.33     0.00        2     0.00     0.00  read_images
  0.00    284.33     0.00        2     0.00     0.00  read_labels
  0.00    284.33     0.00        1     0.00     0.01  create_ann

## Identification des gisements de performance

Les résultats du profilage ci-dessous permettent d'établir une liste des gisements de performance par priorité.

Priorité n°1 : matrix_dot
    Le produit matriciel représente 91,4% du temps total d'exécution du programme. On retrouve souvent cette opération dans les gisements de performance et elle est facilement portable sur GPU.

Priorité n°2 : matrix_minus, matrix_scalar, matrix_transpose, populate_minibatch
    Ces opérations (principalement sur les matrices) représentent un total de 8,31% du temps total d'exécution du programme.

Nous allons porter nos efforts sur ces deux priorités mais il conviendra de vérifier que le gain en complexité est réel puisque le portage d'opérations sur GPU nécessite l'utilisation (a priori absente) de transferts mémoire coûteux.