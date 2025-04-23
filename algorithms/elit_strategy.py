def elitism(population, fitness, elite_percentage):
    elite_size = int(len(population) * elite_percentage)
    elite_indices = np.argsort(fitness)[-elite_size:] 
    elite_population = population[elite_indices]
    return elite_population