
import numpy as np

def tournament_selection(population, fitness, tournament_size):
    fitness = np.array(fitness)
    new_population = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), tournament_size, replace=False)
        winner = tournament[np.argmin(fitness[tournament])]
        new_population.append(population[winner])
    return np.array(new_population)

def roulette_selection(population, fitness):
    fitness = [1/f for f in fitness] 
    fitness_sum = np.sum(fitness)
    probabilities = (fitness / fitness_sum) 
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected_indices]