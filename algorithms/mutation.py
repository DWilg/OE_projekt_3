import numpy as np

def bit_flip_mutation(offspring, mutation_rate):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if np.random.rand() < mutation_rate:
                offspring[i][j] = 1 - offspring[i][j]
    return offspring

def boundary_mutation(offspring, mutation_rate):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i][0] = 1 - offspring[i][0]  
            offspring[i][-1] = 1 - offspring[i][-1] 
    return offspring

def two_point_mutation(offspring, mutation_rate):
    for individual in offspring:
        if np.random.rand() < mutation_rate:
            point1, point2 = sorted(np.random.choice(len(individual), 2, replace=False))
            individual[point1:point2] = 1 - individual[point1:point2]
    return offspring