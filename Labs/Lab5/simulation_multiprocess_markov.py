import multiprocessing as mp
from fonctions_markov import chaine_simple

def multiprocess_simulations(rho, A, nmax, nb_chaines):
    """
    Lance la simulation de plusieurs chaînes de Markov en parallèle
    et renvoie la liste des trajectoires.
    """
    __spec__ = None  # Sécurité pour Windows
    
    seeds = range(nb_chaines)
    parametres = [(rho, A, nmax, seed) for seed in seeds]

    with mp.Pool() as pool:
        trajectoires = pool.starmap(chaine_simple, parametres)
        
    return trajectoires