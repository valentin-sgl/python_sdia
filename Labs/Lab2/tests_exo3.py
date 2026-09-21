import numpy as np

def test_gradient2D(gradient2D_func):
    """
    Test unitaire pour valider le comportement de la fonction gradient2D,
    en vérifiant le format de sortie et le résultat sur des matrices constantes.
    """
    M, N = 3, 4
    X_random = np.random.rand(M, N)
    grad_random = gradient2D_func(X_random)

    assert grad_random.shape == (2, M, N), f"Erreur de format : attendu (2, {M}, {N}), obtenu {grad_random.shape}"

    X_square = np.full((3, 3), 4)
    grad_square = gradient2D_func(X_square)

    expected_square = np.zeros((2, 3, 3))
    np.testing.assert_array_equal(
        grad_square,
        expected_square,
        err_msg="Erreur : le gradient d'une matrice constante carrée doit être nul"
    )

    X_nonsquare = np.full((3, 2), 5)
    grad_nonsquare = gradient2D_func(X_nonsquare)

    expected_nonsquare = np.zeros((2, 3, 2))
    np.testing.assert_array_equal(
        grad_nonsquare,
        expected_nonsquare,
        err_msg="Erreur : le gradient d'une matrice constante non-carrée doit être nul"
    )

    print("Les tests ont été passés avec succès, la fonction gradient2D fonctionne correctement")


def test_tv(tv_func):
    X_const = np.full((3, 3), 4)
    tv_const = tv_func(X_const)
    assert np.isclose(tv_const, 0), f"Erreur : attendu 0, obtenu {tv_const}"

    X_reel = np.array([[1, 2],
                       [1, 2]])
    tv_reel = tv_func(X_reel)
    assert np.isclose(tv_reel, 2.0), f"Erreur : attendu 2, obtenu {tv_reel}"

    X_complex = np.array([[1j, 2j],
                          [1j, 2j]])
    tv_complex = tv_func(X_complex)

    assert isinstance(tv_complex, float) or isinstance(tv_complex, np.floating), "La TV doit renvoyer un nombre réel."
    assert np.isclose(tv_complex, 2.0), f"Erreur : attendu 2, obtenu {tv_complex}"

    print("Tous les tests de la fonction tv() ont réussi avec succès !")


def test_gradient2D_adjoint(gradient2D_func, gradient2D_adjoint_func):
    rng = np.random.default_rng(42)
    M, N = 4, 5

    X = rng.random((M, N)) + 1j * rng.random((M, N))
    Y = rng.random((2, M, N)) + 1j * rng.random((2, M, N))

    DX = gradient2D_func(X)
    DsY = gradient2D_adjoint_func(Y)

    assert DsY.shape == (M, N), \
        f"Erreur de dimension: attendu {(M, N)}, obtenu {DsY.shape}"

    produit_gauche = np.sum(np.conj(DX) * Y)
    produit_droit = np.sum(np.conj(X) * DsY)

    np.testing.assert_allclose(
        produit_gauche,
        produit_droit,
        err_msg="La propriété d'adjonction <D(X), Y> = <X, D*(Y)> n'est pas respectée."
    )

    print("Tous les tests pour gradient2D_adjoint ont réussi avec succès !")
