import multiprocessing as mp
from fonctions_markov import chaine_simple

def multiprocess_simulations(rho, A, nmax, nb_chaines):
    """
    Lance la simulation de plusieurs chaînes de Markov en parallèle
    et renvoie la liste des trajectoires.
    """
    __spec__ = None  # Comme on travaille sur Windows, on doit définir __spec__ à None pour éviter les erreurs de sécurité

    # On génère une suite de graines uniques pour chaque chaîne.
    seeds = range(nb_chaines)

    # starmap demande, en entrée, une liste de tuples où chacun contient tous les arguments.
    parametres = [(rho, A, nmax, seed) for seed in seeds]

    # with mp.Pool() crée un gestionnaire de contexte pour plusieurs processus
    # with assure que les processus soient fermés et nettoyés à la fin.
    with mp.Pool() as pool:
        trajectoires = pool.starmap(chaine_simple, parametres) # starmap distribue les tâches en parallèle et renvoie une liste.
        
    return trajectoires