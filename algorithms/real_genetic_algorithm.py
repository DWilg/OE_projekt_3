import os
import time
import numpy as np

from algorithms.real_crossover import (
    arithmetic_crossover,
    linear_crossover,
    blend_crossover_alpha,
    blend_crossover_alpha_beta,
    average_crossover
)
from algorithms.real_mutation import (
    uniform_mutation,
    gaussian_mutation
)
from algorithms.selection import (
    tournament_selection,
    roulette_selection
)
from algorithms.inversion import inversion

class RealGeneticAlgorithm:
    def __init__(
            self,
            population_size,
            num_generations,
            mutation_rate,
            crossover_rate,
            num_variables,
            inversion_rate=0.1,
            selection_method="Turniejowa",
            crossover_method="Arytmetyczne",
            mutation_method="Równomierna",
            elitism_rate=0.01,
            optimization_goal="min",  
            variable_range=(-5.12, 5.12)
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_variables = num_variables
        self.inversion_rate = inversion_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.elitism_rate = elitism_rate
        self.optimization_goal = optimization_goal.lower()
        self.variable_range = variable_range
        self.population = self.initialize_population()
        self.best_values = []
        self.all_fitness_values = []

    def initialize_population(self):
        lower_bound, upper_bound = self.variable_range
        return lower_bound + np.random.random((self.population_size, self.num_variables)) * (upper_bound - lower_bound)

    def evaluate_population(self, function):
        return np.array([function(ind) for ind in self.population])

    def select_parents(self, fitness_scores):
        if self.selection_method == "Turniejowa":
            return tournament_selection(self.population, fitness_scores, 3)
        elif self.selection_method == "Koło ruletki":
            return roulette_selection(self.population, fitness_scores)
        else:
            raise ValueError("Unknown selection method: " + self.selection_method)

    def crossover(self, parents):
        if self.crossover_method == "Arytmetyczne":
            return arithmetic_crossover(parents, self.crossover_rate)
        elif self.crossover_method == "Liniowe":
            return linear_crossover(parents, self.crossover_rate)
        elif self.crossover_method == "Mieszające-alfa":
            return blend_crossover_alpha(parents, self.crossover_rate)
        elif self.crossover_method == "Mieszające-alfa-beta":
            return blend_crossover_alpha_beta(parents, self.crossover_rate)
        elif self.crossover_method == "Uśredniające":
            return average_crossover(parents, self.crossover_rate)
        else:
            raise ValueError("Unknown crossover method: " + self.crossover_method)

    def mutate(self, offspring):
        if self.mutation_method == "Równomierna":
            return uniform_mutation(offspring, self.mutation_rate, self.variable_range)
        elif self.mutation_method == "Gaussa":
            return gaussian_mutation(offspring, self.mutation_rate, self.variable_range)
        else:
            raise ValueError("Unknown mutation method: " + self.mutation_method)

    def evolve(self, function):
        if not os.path.exists('results'):
            os.makedirs('results')

        results = []
        start_time = time.time()

        for generation in range(self.num_generations):
            fitness_scores = self.evaluate_population(function)
            self.all_fitness_values.append(fitness_scores)

            if self.optimization_goal == "min":
                best_idx = np.argmin(fitness_scores)
                best_value = fitness_scores[best_idx]
                elite_indices = np.argsort(fitness_scores)[:max(1, int(self.elitism_rate * self.population_size))]
            elif self.optimization_goal == "max":
                best_idx = np.argmax(fitness_scores)
                best_value = fitness_scores[best_idx]
                elite_indices = np.argsort(fitness_scores)[::-1][:max(1, int(self.elitism_rate * self.population_size))]
            else:
                raise ValueError("Unknown optimization goal: use 'min' or 'max'")

            self.best_values.append(best_value)
            elites = self.population[elite_indices]

            print(f"Gen {generation + 1} | Best fitness: {best_value}")

            parents = self.select_parents(fitness_scores)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutate(offspring)
            
            if self.inversion_rate > 0:
                for i in range(len(mutated_offspring)):
                    if np.random.rand() < self.inversion_rate:
                        start, end = sorted(np.random.randint(0, self.num_variables, size=2))
                        mutated_offspring[i][start:end] = mutated_offspring[i][start:end][::-1]
            
            self.population = np.vstack((elites, mutated_offspring))
            self.population = self.population[:self.population_size]

            results.append({
                'generation': generation + 1,
                'best_value': best_value,
                'time': time.time() - start_time
            })

        csv_path = os.path.join('results', 'algorithm_results.csv')
        with open(csv_path, 'w') as file:
            file.write("Generation,Best Value,Time\n")
            for result in results:
                file.write(f"{result['generation']},{result['best_value']},{result['time']}\n")

        end_time = time.time()
        print(f"Optymalizacja zakończona. Czas obliczeń: {end_time - start_time:.2f} sekundy.")

        return self