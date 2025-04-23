
import numpy as np

def uniform_mutation(population, mutation_rate, variable_range):
    lower_bound, upper_bound = variable_range
    mutated_population = np.copy(population)
    
    for i in range(len(population)):
        for j in range(len(population[i])):
            if np.random.rand() < mutation_rate:
                mutated_population[i][j] = lower_bound + np.random.random() * (upper_bound - lower_bound)
    
    return mutated_population

def gaussian_mutation(population, mutation_rate, variable_range, sigma=None):

    lower_bound, upper_bound = variable_range
    range_size = upper_bound - lower_bound
    
    if sigma is None:
        sigma = 0.05 * range_size
    
    mutated_population = np.copy(population)
    
    for i in range(len(population)):
        for j in range(len(population[i])):
            if np.random.rand() < mutation_rate:
                delta = np.random.normal(0, sigma)
                mutated_population[i][j] += delta
                
                mutated_population[i][j] = np.clip(mutated_population[i][j], lower_bound, upper_bound)
    
    return mutated_population