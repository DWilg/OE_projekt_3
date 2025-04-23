import pygad
import numpy as np
import matplotlib.pyplot as plt
import time

def decode_ind(individual, bits_per_variable=20, variable_range=(-5.12, 5.12)):
    """
    Funkcja dekodująca binarnego osobnika na wartości rzeczywiste
    
    Args:
        individual: Binarny osobnik (np.array)
        bits_per_variable: Liczba bitów na zmienną
        variable_range: Zakres zmiennych (min, max)
        
    Returns:
        Zdekodowane wartości rzeczywiste (np.array)
    """
    min_val, max_val = variable_range
    chromosome_length = len(individual)
    num_variables = chromosome_length // bits_per_variable
    decoded = np.zeros(num_variables)
    
    for i in range(num_variables):
        start = i * bits_per_variable
        end = start + bits_per_variable
        
        # Konwersja z binarnej na dziesiętną
        binary_chunk = individual[start:end]
        decimal_value = 0
        for bit_idx, bit in enumerate(binary_chunk):
            decimal_value += bit * (2 ** (bits_per_variable - 1 - bit_idx))
        
        # Mapowanie na zakres rzeczywisty
        max_decimal = 2 ** bits_per_variable - 1
        decoded[i] = min_val + (decimal_value / max_decimal) * (max_val - min_val)
    
    return decoded

def fitness_function(ga_instance, individual, solution_idx):
    """
    Funkcja celu z przykładu:
    f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    """
    ind = decode_ind(individual, bits_per_variable=30)
    
    # Ograniczamy liczbę zmiennych do 2, ponieważ funkcja wymaga tylko x i y
    if len(ind) > 2:
        ind = ind[:2]
    elif len(ind) < 2:
        # W przypadku gdy mamy za mało zmiennych, uzupełniamy zerami
        ind = np.pad(ind, (0, 2 - len(ind)), 'constant')
    
    # Funkcja do minimalizacji: (x + 2y - 7)^2 + (2x + y - 5)^2
    result = (ind[0] + 2 * ind[1] - 7)**2 + (2 * ind[0] + ind[1] - 5)**2
    
    # PyGAD maksymalizuje funkcję fitness, więc musimy odwrócić znak
    return -result

def on_generation(ga_instance):
    """
    Funkcja wywoływana po każdej generacji
    """
    generation = ga_instance.generations_completed
    fitness = ga_instance.best_solution()[1]
    
    if generation % 10 == 0:
        print(f"Generacja {generation}, Najlepsza wartość: {-fitness}")  # Odwracamy znak
    
    return False  # Nie przerywaj algorytmu

def custom_single_point_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie jednopunktowe
    """
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    
    for k in range(offspring_size[0]):
        # Indeksy rodziców do krzyżowania
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        # Punkt krzyżowania
        crossover_point = np.random.randint(low=1, high=parents.shape[1]-1)
        
        # Dziedziczenie genów
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring

def custom_two_point_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie dwupunktowe
    """
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    
    for k in range(offspring_size[0]):
        # Indeksy rodziców do krzyżowania
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        # Punkty krzyżowania
        point1, point2 = sorted(np.random.choice(range(1, parents.shape[1]-1), 2, replace=False))
        
        # Dziedziczenie genów
        offspring[k, 0:point1] = parents[parent1_idx, 0:point1]
        offspring[k, point1:point2] = parents[parent2_idx, point1:point2]
        offspring[k, point2:] = parents[parent1_idx, point2:]
    
    return offspring

def custom_uniform_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie jednorodne
    """
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    
    for k in range(offspring_size[0]):
        # Indeksy rodziców do krzyżowania
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        # Maska losowa
        mask = np.random.randint(0, 2, size=parents.shape[1])
        
        # Dziedziczenie genów
        for j in range(parents.shape[1]):
            if mask[j] == 1:
                offspring[k, j] = parents[parent1_idx, j]
            else:
                offspring[k, j] = parents[parent2_idx, j]
    
    return offspring

def custom_gaussian_mutation(offspring, ga_instance):
    """
    Mutacja gaussowska dla reprezentacji binarnej - 
    wykorzystuje prawdopodobieństwo mutation_percent_genes, aby określić które geny ulęgną mutacji
    """
    for chromosome_idx in range(offspring.shape[0]):
        mutation_probability = ga_instance.mutation_percent_genes / 100
        mutation_indices = np.random.random(size=offspring.shape[1]) < mutation_probability
        
        # Zamień bity w wybranych pozycjach
        offspring[chromosome_idx, mutation_indices] = 1 - offspring[chromosome_idx, mutation_indices]
            
    return offspring

def run_example_function():
    """
    Uruchamia algorytm genetyczny z przykładową funkcją celu
    """
    num_generations = 100
    sol_per_pop = 50
    num_parents_mating = 20
    num_genes = 60  # 30 bitów na zmienną * 2 zmienne
    mutation_num_genes = 2  # Liczba genów do mutacji
    
    # Test różnych metod selekcji
    parent_selection_types = ["tournament", "rws", "random"]
    crossover_types = [custom_single_point_crossover, custom_two_point_crossover, custom_uniform_crossover]
    mutation_types = ["random", "swap", custom_gaussian_mutation]
    
    best_solutions = []
    
    # Test różnych kombinacji operatorów
    for parent_selection in parent_selection_types:
        for crossover in crossover_types:
            for mutation in mutation_types:
                print(f"\nTest kombinacji: Selekcja={parent_selection}, Krzyżowanie={crossover.__name__ if callable(crossover) else crossover}, Mutacja={mutation if isinstance(mutation, str) else mutation.__name__}")
                
                # Inicjalizacja instancji GA
                ga_instance = pygad.GA(
                    num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    num_genes=num_genes,
                    fitness_func=fitness_function,
                    init_range_low=0,
                    init_range_high=2,  # Generuje 0 i 1
                    gene_type=int,
                    parent_selection_type=parent_selection,
                    crossover_type=crossover,
                    mutation_type=mutation,
                    mutation_percent_genes=10,  # Procent genów do mutacji
                    keep_elitism=1,  # Zachowaj najlepszego osobnika
                    K_tournament=3,  # Rozmiar turnieju przy selekcji turniejowej
                    on_generation=on_generation
                )
                
                # Uruchomienie algorytmu
                start_time = time.time()
                ga_instance.run()
                end_time = time.time()
                
                # Wyniki
                solution, solution_fitness, solution_idx = ga_instance.best_solution()
                decoded_solution = decode_ind(solution, bits_per_variable=30)
                print(f"Czas wykonania: {end_time - start_time:.2f} sekund")
                print(f"Najlepsze rozwiązanie (x,y): ({decoded_solution[0]:.6f}, {decoded_solution[1]:.6f})")
                print(f"Najlepsza wartość funkcji: {-solution_fitness:.6f}")
                
                best_solutions.append({
                    "parent_selection": parent_selection,
                    "crossover": crossover.__name__ if callable(crossover) else crossover,
                    "mutation": mutation if isinstance(mutation, str) else mutation.__name__,
                    "solution": decoded_solution,
                    "fitness": -solution_fitness,
                    "time": end_time - start_time
                })
                
                # Wykres zbieżności
                plt.figure(figsize=(10, 6))
                plt.plot(range(num_generations), -np.array(ga_instance.best_solutions_fitness))
                plt.title(f"Zbieżność: {parent_selection}, {crossover.__name__ if callable(crossover) else crossover}, {mutation if isinstance(mutation, str) else mutation.__name__}")
                plt.xlabel("Generacja")
                plt.ylabel("Wartość funkcji")
                plt.grid(True)
                plt.savefig(f"results/example_function_{parent_selection}_{crossover.__name__ if callable(crossover) else crossover}_{mutation if isinstance(mutation, str) else mutation.__name__}.png")
                plt.close()
    
    # Podsumowanie wyników
    print("\n" + "=" * 80)
    print("PODSUMOWANIE WYNIKÓW DLA PRZYKŁADOWEJ FUNKCJI")
    print("=" * 80)
    print("{:<15} {:<25} {:<15} {:<20} {:<15}".format(
        "Selekcja", "Krzyżowanie", "Mutacja", "Rozwiązanie (x,y)", "Wartość"
    ))
    print("-" * 80)
    
    sorted_results = sorted(best_solutions, key=lambda x: x["fitness"])
    for result in sorted_results:
        print("{:<15} {:<25} {:<15} {:<20} {:<15.6f}".format(
            result["parent_selection"],
            result["crossover"],
            result["mutation"],
            f"({result['solution'][0]:.2f}, {result['solution'][1]:.2f})",
            result["fitness"]
        ))
    
    # Teoretyczne optimum funkcji
    # Można wykazać analitycznie, że minimum występuje w punkcie (1, 3)
    x_opt, y_opt = 1.0, 3.0
    opt_value = (x_opt + 2*y_opt - 7)**2 + (2*x_opt + y_opt - 5)**2
    print("\nTeorytyczne optimum funkcji:")
    print(f"x = {x_opt}, y = {y_opt}")
    print(f"f(x,y) = {opt_value}")
    
    return sorted_results

if __name__ == "__main__":
    # Stworzenie katalogu wyników, jeśli nie istnieje
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Uruchomienie przykładu z funkcją (x + 2y - 7)^2 + (2x + y - 5)^2
    run_example_function()