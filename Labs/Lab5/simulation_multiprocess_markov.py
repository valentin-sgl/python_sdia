import matplotlib.pyplot as plt
import multiprocessing as mp
from fonctions_markov import chaine_simple

rho = [0.2, 0.5, 0.3]
A = [[0.1, 0.6, 0.3],
     [0.4, 0.2, 0.4],
     [0.7, 0.1, 0.2]]
nmax = 20
nb_chaines = 5
seeds = range(nb_chaines)
parametres = [(rho, A, nmax, seed) for seed in seeds]

if __name__ == '__main__':
    __spec__ = None


    with mp.Pool() as pool :
        trajectoires = pool.starmap(chaine_simple, parametres)
    
    plt.figure(figsize=(8, 5))

    for i, trajectoire in enumerate(trajectoires) :
        plt.plot(trajectoire, marker='o', linestyle='-', label=f'Chaîne {i+1}')

    plt.xticks(range(0,nmax+1,2))
    plt.yticks([0,1,2])
    plt.xlabel('Itération')
    plt.ylabel('État')
    plt.title('Trajectoire de Markov')
    plt.legend()
    plt.show() 