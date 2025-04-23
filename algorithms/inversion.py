import numpy as np

def inversion(population, inversion_rate=0.1):
    for i in range(len(population)):
        if np.random.rand() < inversion_rate:
            start, end = sorted(np.random.randint(0, len(population[i]), size=2))
            population[i][start:end] = population[i][start:end][::-1]
    return population