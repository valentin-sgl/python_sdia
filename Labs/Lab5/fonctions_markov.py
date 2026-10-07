import numpy as np

def markov(rho, A, nmax, rng) :
    rho = np.asarray(rho)
    A = np.asarray(A)

    assert rho.ndim == 1, "rho doit être un vecteur à 1 dimension"
    N = rho.shape[0]

    assert A.ndim == 2, "A doit être une matrice à 2 dimensions"
    assert A.shape == (N, N), "A doit être une matrice carrée de taille N x N"

    assert np.all(rho >= 0), "rho doit contenir des valeurs non négatives"
    assert np.isclose(np.sum(rho), 1), "la somme des éléments de rho doit être égale à 1"

    assert np.all(A >= 0), "A doit contenir des valeurs non négatives"
    assert np.allclose(np.sum(A, axis=1), 1), "les lignes de A doivent sommer à 1"

    etat = np.arange(N)
    X = np.empty(nmax+1, dtype=int)

    X[0] = rng.choice(etat, p=rho)

    for i in range(nmax) :
        etat_actuel = X[i]
        transitions = A[etat_actuel]
        X[i+1] = rng.choice(etat, p=transitions)

    return X

def chaine_simple(rho, A, nmax, seed) :
    rng = np.random.default_rng(seed)
    return markov(rho, A, nmax, rng)