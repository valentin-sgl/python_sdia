from fonctions_markov import markov, chaine_simple

def dask_simulations(client, rho, A, nmax, nb_chaines) :
    """
    Lance la simulation de plusieurs chaînes de Markov en parallèle
    et renvoie la liste des trajectoires.
    """

    seeds = list(range(nb_chaines)) # On crée une liste de graines pour chaque chaîne de Markov afin d'assurer la reproductibilité des résultats.
    
    # On duplique tous les paramètres pour chaque chaîne de Markov afin de pouvoir les passer à la fonction map de Dask.
    liste_rho = [rho] * nb_chaines
    liste_A = [A] * nb_chaines
    liste_nmax = [nmax] * nb_chaines

    # client.map permet de lancer la fonction chaine_simple en parallèle pour chaque ensemble de paramètres.
    futures = client.map(chaine_simple, liste_rho, liste_A, liste_nmax, seeds)

    # client.gather bloque l'exécution tant que toutes les tâches ne sont pas terminées.
    # Ensuite tous les résultats sont ajoutés dans une liste.
    trajectoires = client.gather(futures)

    return trajectoires