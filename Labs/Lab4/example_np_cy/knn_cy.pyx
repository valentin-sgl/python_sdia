import numpy as np
cimport numpy as cnp
import bottleneck as bn

# Importation de la fonction racine carrée directement depuis la bibliothèque C
from libc.math cimport sqrt

# Importation pour les décorateurs finaux
cimport cython


def knn_cython_v1(double[:, ::1] x_train, double[:] class_train, double[:, ::1] x_test, int k):
    """
    Étape 1 : Typage des Memory Views et boucle explicite pour les distances.
    """
    cdef int n_train = x_train.shape[0]
    cdef int n_features = x_train.shape[1]
    cdef int n_test = x_test.shape[0]
    
    # Création du tableau de sortie via numpy, et accès via une memory view (type long = int64)
    predict_classe_np = np.zeros(n_test, dtype=int)
    cdef long long[:] predict_view = predict_classe_np
    
    # Tableau temporaire pour les distances
    distances_np = np.zeros(n_train, dtype=np.float64)
    cdef double[:] dist_view = distances_np
    
    # Déclaration des variables de boucle en C (fondamental pour la vitesse)
    cdef int i, j, d
    cdef double diff, dist_sq
    
    # On reconvertit class_train en tableau NumPy classique pour le "fancy indexing" 
    # (les memory views ne supportent pas l'indexation par un tableau comme class_train[knn])
    class_train_np = np.asarray(class_train)
    
    for i in range(n_test):
        # 1. Calcul des distances euclidiennes avec des boucles C pures
        for j in range(n_train):
            dist_sq = 0.0
            for d in range(n_features):
                diff = x_train[j, d] - x_test[i, d]
                dist_sq += diff * diff
            
            # Stockage dans la memory view pour éviter le surcoût Python
            dist_view[j] = sqrt(dist_sq)
        
        # 2. Appel aux fonctions NumPy/Bottleneck en utilisant les tableaux sous-jacents (_np)
        knn = bn.argpartition(distances_np, k)[:k]
        classes = class_train_np[knn]
        predict_view[i] = np.argmax(np.bincount(classes.astype(int)))
        
    return predict_classe_np


# Étape 2 : Désactivation des sécurités Python (boundscheck et wraparound)
@cython.boundscheck(False)
@cython.wraparound(False)
def knn_cython_v2(double[:, ::1] x_train, double[:] class_train, double[:, ::1] x_test, int k):
    """
    Étape 2 : Ajout des décorateurs finaux pour supprimer les vérifications 
    de dépassement d'index. Le code interne est identique à v1.
    """
    cdef int n_train = x_train.shape[0]
    cdef int n_features = x_train.shape[1]
    cdef int n_test = x_test.shape[0]
    
    predict_classe_np = np.zeros(n_test, dtype=int)
    cdef long long[:] predict_view = predict_classe_np
    
    distances_np = np.zeros(n_train, dtype=np.float64)
    cdef double[:] dist_view = distances_np
    
    cdef int i, j, d
    cdef double diff, dist_sq
    
    class_train_np = np.asarray(class_train)
    
    for i in range(n_test):
        for j in range(n_train):
            dist_sq = 0.0
            for d in range(n_features):
                diff = x_train[j, d] - x_test[i, d]
                dist_sq += diff * diff
            dist_view[j] = sqrt(dist_sq)
        
        knn = bn.argpartition(distances_np, k)[:k]
        classes = class_train_np[knn]
        predict_view[i] = np.argmax(np.bincount(classes.astype(int)))
        
    return predict_classe_np