

import numpy as np

def one_point_crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, len(parents[0]))
            child1 = np.concatenate((parents[i][:point], parents[i+1][point:]))
            child2 = np.concatenate((parents[i+1][:point], parents[i][point:]))
            offspring.extend([child1, child2])
        else:
            offspring.extend([parents[i], parents[i+1]])
    return np.array(offspring)

def two_point_crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            point1 = np.random.randint(1, len(parents[0])-1)
            point2 = np.random.randint(point1+1, len(parents[0]))
            child1 = np.concatenate((parents[i][:point1], parents[i+1][point1:point2], parents[i][point2:]))
            child2 = np.concatenate((parents[i+1][:point1], parents[i][point1:point2], parents[i+1][point2:]))
            offspring.extend([child1, child2])
        else:
            offspring.extend([parents[i], parents[i+1]])
    return np.array(offspring)

def uniform_crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            mask = np.random.randint(0, 2, size=len(parents[0]))  
            child1 = np.where(mask == 1, parents[i], parents[i+1])
            child2 = np.where(mask == 1, parents[i+1], parents[i])
            offspring.extend([child1, child2])
        else:
            offspring.extend([parents[i], parents[i+1]])
    return np.array(offspring)

def granular_crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            parent1, parent2 = parents[i], parents[i + 1]
            mask = np.random.randint(0, 2, size=parent1.shape)
            child1 = mask * parent1 + (1 - mask) * parent2
            child2 = mask * parent2 + (1 - mask) * parent1
            offspring.extend([child1, child2])
        else:
            offspring.extend([parents[i], parents[i + 1]])
    return np.array(offspring)