# TP Centrale Nantes INFOIA GPGPU

Ce README.md est un document qui nous a servi de support originel pour notre réfléxion sur ce TP, il contient quelques débuts d'analyse, la manière dont nous avons envisagé la retranscription de nos avancées ainsi que les lignes de commandes utiles pour exécuter nos programmes.

Ce fichier est aussi l'occasion d'expliciter la structure du git en lui même.
Il y a 4 dossiers :
CODE_CUBLAS : qui fut utile au tout début du TP quand nous avons envisager d'utiliser la librairie CUBLAS avant de nous résigner face à nos difficultés de manier cette dernière
CODE_CUDA : il contient le code fonctionnant avec la première et la deuxième version de "matrix_dot_cuda"
CODE_MEM_UNI : il contient le code fonctionnant avec la mémoire unifiée
CODE_PROF : il contient le code fourni au début du TP et il nous a servi de référence pour adapter le notre
DONNEES_MNIST : il contient les données MNSIT afin de réaliser l'entrainement du réseaux de neurones
Chaque dossier (hormis DONNEES_MNSIT) contient un code fonctionnel qu'il suffit d'éxécuter avce la commande bash suivante (lorsque l'invite de commade se trouve dans le dossier en question) : nvcc *.cu -lm (ou gcc *.c -lm pour CODE_PROF)
Pourquoi cette structure ? La réponse est en partie dans la ligne précédente mais également parce que cela nous facilitait la mise à jour du repository en lui-même : chacun a pu travail dans un dossier sans contaminer ceux des autres, évitant ainsi de nombreux conflits de résolutions de ces derniers.

# Une partie de notre travail sur ce projet :

Comment compiler : gcc *.c -lm
Utiliser cublas : nvcc tests_antho.cu -lcublas
Le rapport overleaf latex : https://www.overleaf.com/7822131397tpkssvjjbshh#3edbfc

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

# Analyse du code séquentiel

## Structure du code séquentiel

La formulation algorithmique du problème résolu par l'ensemble du code est détaillée dans le sujet du TP.
On rappelle ici le prototype des fonctions qui composent le code et la structure du main.c.

### Liste des librairies requises

<stdlib.h>  : gestion mémoire dynamique, environnement et processus, fonctions de manipulation chaîne/nombre, tri, recherche
<stdio.h>   : fonctions lecture clavier, écriture console, manipulation de fichiers et opérations d'entrée/sortie standard
<assert.h>  : macros pour effectuer des assertions
<stdbool.h> : utilisation des variables booléennes et opérations logiques
<stdint.h>  : fournit des types entiers de largeurs fixes
<math.h>    : opérations mathématiques courantes
<string.h>  : fonctions pour la manipulation de chaînes de caractères
<time.h>    : manipulation, formatage et mesure du temps d'exécution des programmes

### Récapitulatif des prototype de l'ensemble des fonctions

%% Fonctions utilitaires pour le formatage du dataset train/test
uint32_t    make_uint32(byte buffer[]);
byte *      read_labels(const char filename[], unsigned* n );
image *     read_images(const char filename[], unsigned* n );

%% Fonctions utilitaires pour le calcul matriciel
matrix_t *  alloc_matrix(unsigned rows, unsigned columns);
void        destroy_matrix(matrix_t *m);
void        print_matrix(matrix_t *m, bool is_short);
void        hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);
void        matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);
void        matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);
void        matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);
void        matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res);
void        matrix_transpose(matrix_t *m1, matrix_t *res);
void        matrix_scalar(matrix_t *m1, double s, matrix_t *res);
void        matrix_memcpy(matrix_t *dest, const matrix_t *src);

%% Fonctions utilitaires pour la manipulation des réseaux de neurones
% Fonction de déclaration d'un objet réseau de neurones
ann_t *     create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer);
Fonction de définition d'un objet couche du réseau de neurones
layer_t *   create_layer(unsigned l, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size);
% Fonction de déclaration du vecteur d'entrée
void        set_input(ann_t *nn, matrix_t* input);
% Fonction d'affichage du réseau de neurones
void        print_nn(ann_t *nn);
% Fonction de propagation avant (calcul d'une prédiction)
void        forward(ann_t *nn, double (*activation_function)(double));
% Fonction de back-propagation (ajustement des paramètres)
void        backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double));

## Performances du code séquentiel

### Résultats du profilage avec gprof

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

### Identification des gisements de performance

Les résultats du profilage ci-dessus permettent d'établir une liste des gisements de performance par priorité.

Priorité n°1 : matrix_dot
    Le produit matriciel représente 91,4% du temps total d'exécution du programme. On retrouve souvent cette opération dans les gisements de performance et elle est facilement portable sur GPU.

Priorité n°2 : matrix_minus, matrix_scalar, matrix_transpose, populate_minibatch
    Ces opérations (principalement sur les matrices) représentent un total de 8,31% du temps total d'exécution du programme.

Nous allons porter nos efforts sur ces deux priorités mais il conviendra de vérifier que le gain en complexité est réel puisque le portage d'opérations sur GPU nécessite l'utilisation de transferts mémoire coûteux, absents du code séquentiel initial.
