from fonctions_markov import markov, chaine_simple
from dask.distributed import Client

def dask_simulations(rho, A, nmax, nb_chaines) :
    """
    Lance la simulation de plusieurs chaînes de Markov en parallèle
    et renvoie la liste des trajectoires.
    """

    seeds = list(range(nb_chaines))
    liste_rho = [rho] * nb_chaines
    liste_A = [A] * nb_chaines
    liste_nmax = [nmax] * nb_chaines

    futures = client.map(chaine_simple, liste_rho, liste_A, liste_nmax, seeds)

    trajectoires = client.gather(futures)

    return trajectoires