import pygad
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from optimization.benchmark_functions import Hypersphere, Rastrigin, Hyperellipsoid
from optimization.external_benchmark import create_benchmark_function

def decode_binary(binary_solution, num_variables, variable_range):
    """
    Funkcja dekodująca binarną reprezentację na wartości rzeczywiste
    
    Args:
        binary_solution: Binarny osobnik (np.array)
        num_variables: Liczba zmiennych decyzyjnych
        variable_range: Zakres zmiennych (min, max)
        
    Returns:
        Zdekodowane wartości rzeczywiste (np.array)
    """
    min_val, max_val = variable_range
    chromosome_length = len(binary_solution)
    bits_per_variable = chromosome_length // num_variables
    decoded = np.zeros(num_variables)
    
    for i in range(num_variables):
        start = i * bits_per_variable
        end = start + bits_per_variable
        
        # Konwersja z binarnej na dziesiętną
        binary_chunk = binary_solution[start:end]
        decimal_value = 0
        for bit_idx, bit in enumerate(binary_chunk):
            decimal_value += bit * (2 ** (bits_per_variable - 1 - bit_idx))
        
        # Mapowanie na zakres rzeczywisty
        max_decimal = 2 ** bits_per_variable - 1
        decoded[i] = min_val + (decimal_value / max_decimal) * (max_val - min_val)
    
    return decoded

# Funkcje fitness dla różnych typów benchmarków
def fitness_function_binary(ga_instance, solution, solution_idx):
    """
    Funkcja fitness dla reprezentacji binarnej
    
    Args:
        ga_instance: Instancja algorytmu genetycznego
        solution: Osobnik binarny
        solution_idx: Indeks osobnika w populacji
    
    Returns:
        Wartość funkcji celu
    """
    global function_to_optimize, num_variables, variable_range
    decoded_solution = decode_binary(solution, num_variables, variable_range)
    return -function_to_optimize(decoded_solution)  # Znak minus, ponieważ PyGAD maksymalizuje

def fitness_function_real(ga_instance, solution, solution_idx):
    """
    Funkcja fitness dla reprezentacji rzeczywistej
    
    Args:
        ga_instance: Instancja algorytmu genetycznego
        solution: Osobnik rzeczywisty
        solution_idx: Indeks osobnika w populacji
    
    Returns:
        Wartość funkcji celu
    """
    global function_to_optimize
    return -function_to_optimize(solution)  # Znak minus, ponieważ PyGAD maksymalizuje

# Implementacja własnych operatorów krzyżowania
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

# Implementacja operatorów krzyżowania dla reprezentacji rzeczywistej
def custom_arithmetic_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie arytmetyczne dla reprezentacji rzeczywistej
    """
    offspring = np.empty(offspring_size, dtype=float)
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        alpha = np.random.random()
        offspring[k] = alpha * parents[parent1_idx] + (1 - alpha) * parents[parent2_idx]
    
    return offspring

def custom_blend_crossover_alpha(parents, offspring_size, ga_instance):
    """
    Krzyżowanie mieszające alfa dla reprezentacji rzeczywistej
    """
    offspring = np.empty(offspring_size, dtype=float)
    alpha = 0.5
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        for j in range(parents.shape[1]):
            min_val = min(parents[parent1_idx, j], parents[parent2_idx, j])
            max_val = max(parents[parent1_idx, j], parents[parent2_idx, j])
            range_val = max_val - min_val
            
            lower_bound = min_val - alpha * range_val
            upper_bound = max_val + alpha * range_val
            
            offspring[k, j] = np.random.uniform(lower_bound, upper_bound)
    
    return offspring

def custom_blend_crossover_alpha_beta(parents, offspring_size, ga_instance):
    """
    Krzyżowanie mieszające alfa-beta dla reprezentacji rzeczywistej
    """
    offspring = np.empty(offspring_size, dtype=float)
    alpha = 0.5
    beta = 0.3
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        for j in range(parents.shape[1]):
            min_val = min(parents[parent1_idx, j], parents[parent2_idx, j])
            max_val = max(parents[parent1_idx, j], parents[parent2_idx, j])
            range_val = max_val - min_val
            
            if np.random.random() < 0.5:
                lower_bound = min_val - alpha * range_val
                upper_bound = max_val + beta * range_val
            else:
                lower_bound = min_val - beta * range_val
                upper_bound = max_val + alpha * range_val
                
            offspring[k, j] = np.random.uniform(lower_bound, upper_bound)
    
    return offspring

def custom_linear_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie liniowe dla reprezentacji rzeczywistej
    """
    offspring = np.empty(offspring_size, dtype=float)
    
    for k in range(0, offspring_size[0], 2):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        if k < offspring_size[0]:
            # Pierwsze dziecko: średnia rodziców
            offspring[k] = 0.5 * parents[parent1_idx] + 0.5 * parents[parent2_idx]
        
        if k+1 < offspring_size[0]:
            # Drugie dziecko: ekstrapolacja w kierunku pierwszego rodzica
            offspring[k+1] = 1.5 * parents[parent1_idx] - 0.5 * parents[parent2_idx]
            
            # Opcjonalnie trzecie dziecko: ekstrapolacja w kierunku drugiego rodzica
            # offspring[k+2] = -0.5 * parents[parent1_idx] + 1.5 * parents[parent2_idx]
    
    return offspring

def custom_average_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie uśredniające dla reprezentacji rzeczywistej
    """
    offspring = np.empty(offspring_size, dtype=float)
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        offspring[k] = (parents[parent1_idx] + parents[parent2_idx]) / 2
    
    return offspring

# Implementacja operatorów mutacji
def custom_gaussian_mutation(offspring, ga_instance):
    """
    Mutacja gaussa dla reprezentacji rzeczywistej
    """
    for chromosome_idx in range(offspring.shape[0]):
        # Wybierz losowe geny do mutacji na podstawie prawdopodobieństwa mutacji
        mutation_probability = ga_instance.mutation_percent_genes / 100
        mutation_indices = np.random.random(size=offspring.shape[1]) < mutation_probability
        
        if np.any(mutation_indices):
            # Zastosuj mutację Gaussa tylko dla wybranych genów
            sigma = 0.1 * (ga_instance.random_mutation_max_val - ga_instance.random_mutation_min_val)
            offspring[chromosome_idx, mutation_indices] += np.random.normal(0, sigma, size=np.sum(mutation_indices))
            
            # Ogranicz wartości do dozwolonego zakresu
            offspring[chromosome_idx] = np.clip(
                offspring[chromosome_idx],
                ga_instance.random_mutation_min_val,
                ga_instance.random_mutation_max_val
            )
            
    return offspring

# Funkcja pomocnicza do śledzenia postępu algorytmu
def on_generation(ga_instance):
    """
    Funkcja wywoływana po każdej generacji
    """
    generation = ga_instance.generations_completed
    
    # Bezpieczne odwołanie do best_solution tylko jeśli populacja została już oceniona
    if hasattr(ga_instance, 'best_solution_fitness') and ga_instance.best_solution_fitness is not None:
        fitness = ga_instance.best_solution_fitness
        if generation % 10 == 0:
            print(f"Generacja {generation}, Najlepsza wartość: {-fitness}")  # Odwracamy znak
    else:
        if generation % 10 == 0:
            print(f"Generacja {generation}, Wartość fitness jeszcze nie obliczona")
    
    return False  # Nie przerywaj algorytmu

# Funkcja do uruchamiania algorytmu genetycznego z reprezentacją binarną
def run_binary_ga(function_name, num_genes_per_variable, num_vars, var_range, 
                  population_size=50, num_generations=100, num_parents=20,
                  crossover_type="single_point", mutation_type="random", parent_selection="tournament"):
    """
    Uruchamia algorytm genetyczny z reprezentacją binarną
    
    Args:
        function_name: Nazwa funkcji testowej
        num_genes_per_variable: Liczba bitów na zmienną
        num_vars: Liczba zmiennych decyzyjnych
        var_range: Zakres zmiennych (min, max)
        population_size: Rozmiar populacji
        num_generations: Liczba generacji
        num_parents: Liczba rodziców do krzyżowania
        crossover_type: Typ krzyżowania
        mutation_type: Typ mutacji
        parent_selection: Metoda selekcji rodziców
        
    Returns:
        Instancja algorytmu genetycznego po zakończeniu ewolucji
    """
    global function_to_optimize, num_variables, variable_range
    
    # Skonfiguruj funkcję celu
    if function_name == "Rastrigin":
        function_obj = Rastrigin(n_dimensions=num_vars)
        function_to_optimize = function_obj._evaluate
    elif function_name == "Hypersphere":
        function_obj = Hypersphere(n_dimensions=num_vars)
        function_to_optimize = function_obj._evaluate
    elif function_name == "Hyperellipsoid":
        function_obj = Hyperellipsoid(n_dimensions=num_vars)
        function_to_optimize = function_obj._evaluate
    else:
        # Użyj zewnętrznej funkcji z biblioteki
        function_to_optimize, variable_range = create_benchmark_function(function_name, num_vars)
        var_range = variable_range
    
    num_variables = num_vars
    variable_range = var_range
    
    # Określ sposób krzyżowania
    if crossover_type == "single_point":
        crossover_func = custom_single_point_crossover
    elif crossover_type == "two_points":
        crossover_func = custom_two_point_crossover
    elif crossover_type == "uniform":
        crossover_func = custom_uniform_crossover
    else:
        crossover_func = None  # Użyj domyślnego z PyGAD
        
    # Stwórz instancję GA z reprezentacją binarną
    ga_instance = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=population_size,
        num_parents_mating=num_parents,
        num_genes=num_genes_per_variable * num_vars,
        fitness_func=fitness_function_binary,
        init_range_low=0,
        init_range_high=2,  # Generuje 0 i 1
        gene_type=int,
        parent_selection_type=parent_selection,
        crossover_type=crossover_func if crossover_func else crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=10,  # Procent genów do mutacji
        keep_elitism=1,  # Zachowaj najlepszego osobnika
        K_tournament=3,  # Rozmiar turnieju przy selekcji turniejowej
        random_mutation_max_val=1.0,
        random_mutation_min_val=0.0,
        on_generation=on_generation,
        save_best_solutions=True,
        save_solutions=True
    )
    
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    # Pobierz najlepsze rozwiązanie
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    decoded_solution = decode_binary(solution, num_variables, variable_range)
    print(f"Czas wykonania: {end_time - start_time:.2f} sekund")
    print(f"Najlepsze rozwiązanie: {decoded_solution}")
    print(f"Najlepsza wartość funkcji: {-solution_fitness}")  # Odwracamy znak
    
    return ga_instance

# Funkcja do uruchamiania algorytmu genetycznego z reprezentacją rzeczywistą
def run_real_ga(function_name, num_vars, var_range, 
                population_size=50, num_generations=100, num_parents=20,
                crossover_type="arithmetic", mutation_type="adaptive", parent_selection="tournament"):
    """
    Uruchamia algorytm genetyczny z reprezentacją rzeczywistą
    
    Args:
        function_name: Nazwa funkcji testowej
        num_vars: Liczba zmiennych decyzyjnych
        var_range: Zakres zmiennych (min, max)
        population_size: Rozmiar populacji
        num_generations: Liczba generacji
        num_parents: Liczba rodziców do krzyżowania
        crossover_type: Typ krzyżowania
        mutation_type: Typ mutacji
        parent_selection: Metoda selekcji rodziców
        
    Returns:
        Instancja algorytmu genetycznego po zakończeniu ewolucji
    """
    global function_to_optimize
    
    # Skonfiguruj funkcję celu
    if function_name == "Rastrigin":
        function_obj = Rastrigin(n_dimensions=num_vars)
        function_to_optimize = function_obj._evaluate
    elif function_name == "Hypersphere":
        function_obj = Hypersphere(n_dimensions=num_vars)
        function_to_optimize = function_obj._evaluate
    elif function_name == "Hyperellipsoid":
        function_obj = Hyperellipsoid(n_dimensions=num_vars)
        function_to_optimize = function_obj._evaluate
    else:
        # Użyj zewnętrznej funkcji z biblioteki
        function_to_optimize, var_range = create_benchmark_function(function_name, num_vars)
    
    # Określ sposób krzyżowania
    if crossover_type == "arithmetic":
        crossover_func = custom_arithmetic_crossover
    elif crossover_type == "blend_alpha":
        crossover_func = custom_blend_crossover_alpha
    elif crossover_type == "blend_alpha_beta":
        crossover_func = custom_blend_crossover_alpha_beta
    elif crossover_type == "linear":
        crossover_func = custom_linear_crossover
    elif crossover_type == "average":
        crossover_func = custom_average_crossover
    else:
        crossover_func = None  # Użyj domyślnego z PyGAD
    
    # Określ sposób mutacji
    if mutation_type == "gaussian":
        mutation_func = custom_gaussian_mutation
    else:
        mutation_func = None  # Użyj domyślnego z PyGAD
        
    # Stwórz instancję GA z reprezentacją rzeczywistą
    ga_instance = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=population_size,
        num_parents_mating=num_parents,
        num_genes=num_vars,
        fitness_func=fitness_function_real,
        init_range_low=var_range[0],
        init_range_high=var_range[1],
        gene_type=float,
        parent_selection_type=parent_selection,
        crossover_type=crossover_func,
        mutation_type=mutation_func if mutation_func else mutation_type,
        mutation_percent_genes=10,  # Procent genów do mutacji
        keep_elitism=1,  # Zachowaj najlepszego osobnika
        K_tournament=3,  # Rozmiar turnieju przy selekcji turniejowej
        random_mutation_max_val=var_range[1],
        random_mutation_min_val=var_range[0],
        on_generation=on_generation,
        save_best_solutions=True,
        save_solutions=True
    )
    
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    # Pobierz najlepsze rozwiązanie
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Czas wykonania: {end_time - start_time:.2f} sekund")
    print(f"Najlepsze rozwiązanie: {solution}")
    print(f"Najlepsza wartość funkcji: {-solution_fitness}")  # Odwracamy znak
    
    return ga_instance

# Funkcja do wizualizacji wyników
def plot_results(ga_instance, representation_type="Binary", title_prefix=""):
    """
    Wizualizuje wyniki algorytmu genetycznego
    
    Args:
        ga_instance: Instancja algorytmu genetycznego po zakończeniu ewolucji
        representation_type: Typ reprezentacji ("Binary" lub "Real")
        title_prefix: Prefiks do tytułu wykresu
    """
    # Przygotuj dane do wizualizacji
    generations = np.arange(ga_instance.num_generations)
    
    # Sprawdź czy mamy dostęp do najlepszego rozwiązania
    solution, solution_fitness, _ = ga_instance.best_solution()
    
    # Jeśli best_solutions_fitness nie istnieje lub jest puste, utwórz sztuczne dane
    if not hasattr(ga_instance, 'best_solutions_fitness') or len(ga_instance.best_solutions_fitness) == 0:
        print("Ostrzeżenie: Używam stałej wartości najlepszego rozwiązania do wizualizacji")
        # Tworzymy tablicę z powtórzoną wartością najlepszego rozwiązania
        best_solutions = np.ones(ga_instance.num_generations) * (-solution_fitness)
    else:
        # Upewnij się, że tablica ma dokładnie tyle elementów, ile wynosi liczba generacji
        best_solutions_array = np.array(ga_instance.best_solutions_fitness)
        if len(best_solutions_array) != ga_instance.num_generations:
            print(f"Ostrzeżenie: Tablica best_solutions_fitness ma rozmiar {len(best_solutions_array)}, "
                  f"dostosowuję do {ga_instance.num_generations}")
            # Przycięcie lub uzupełnienie tablicy
            if len(best_solutions_array) > ga_instance.num_generations:
                # Przytnij tablicę, jeśli jest za duża
                best_solutions = -best_solutions_array[:ga_instance.num_generations]
            else:
                # Uzupełnij tablicę, jeśli jest za mała
                best_solutions = np.ones(ga_instance.num_generations) * (-solution_fitness)
                best_solutions[:len(best_solutions_array)] = -best_solutions_array
        else:
            best_solutions = -best_solutions_array
    
    # Upewnij się, że obie tablice mają ten sam wymiar
    assert len(generations) == len(best_solutions), f"Niezgodność wymiarów: generations {len(generations)}, best_solutions {len(best_solutions)}"
    
    # Uproszczony wykres najlepszej wartości
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_solutions, 'b-', label='Najlepsza wartość')
    plt.title(f'{title_prefix}Zmiana najlepszej wartości funkcji celu')
    plt.xlabel('Generacja')
    plt.ylabel('Wartość funkcji')
    plt.grid(True)
    plt.legend()
    
    # Zapisz wykres
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/{title_prefix.replace(" ", "_")}_{representation_type}_results.png')
    # Zamykamy wykres by nie zużywać pamięci
    plt.close()
    
    print(f"Wykres zapisany do pliku: results/{title_prefix.replace(' ', '_')}_{representation_type}_results.png")

def compare_methods(function_name, num_vars, var_range, num_genes_per_variable=10,
                   population_size=50, num_generations=100):
    """
    Porównuje różne metody algorytmu genetycznego
    
    Args:
        function_name: Nazwa funkcji testowej
        num_vars: Liczba zmiennych decyzyjnych
        var_range: Zakres zmiennych (min, max)
        num_genes_per_variable: Liczba bitów na zmienną (dla reprezentacji binarnej)
        population_size: Rozmiar populacji
        num_generations: Liczba generacji
    """
    results = []
    
    print("\n=== PORÓWNANIE METOD SELEKCJI (REPREZENTACJA BINARNA) ===")
    
    # Porównanie metod selekcji (binarna)
    for selection in ["tournament", "rws", "random"]:
        print(f"\nTestowanie metody selekcji: {selection}")
        ga_instance = run_binary_ga(
            function_name=function_name,
            num_genes_per_variable=num_genes_per_variable,
            num_vars=num_vars,
            var_range=var_range,
            population_size=population_size,
            num_generations=num_generations,
            parent_selection=selection
        )
        solution, fitness, _ = ga_instance.best_solution()
        results.append({
            "representation": "Binary",
            "selection": selection,
            "crossover": "single_point",
            "mutation": "random",
            "fitness": -fitness,
            "instance": ga_instance
        })
        plot_results(ga_instance, "Binary", f"Selekcja {selection} - ")
    
    print("\n=== PORÓWNANIE METOD KRZYŻOWANIA (REPREZENTACJA BINARNA) ===")
    
    # Porównanie metod krzyżowania (binarna)
    for crossover in ["single_point", "two_points", "uniform"]:
        print(f"\nTestowanie metody krzyżowania: {crossover}")
        ga_instance = run_binary_ga(
            function_name=function_name,
            num_genes_per_variable=num_genes_per_variable,
            num_vars=num_vars,
            var_range=var_range,
            population_size=population_size,
            num_generations=num_generations,
            crossover_type=crossover
        )
        solution, fitness, _ = ga_instance.best_solution()
        results.append({
            "representation": "Binary",
            "selection": "tournament",
            "crossover": crossover,
            "mutation": "random",
            "fitness": -fitness,
            "instance": ga_instance
        })
        plot_results(ga_instance, "Binary", f"Krzyżowanie {crossover} - ")
    
    print("\n=== PORÓWNANIE METOD MUTACJI (REPREZENTACJA BINARNA) ===")
    
    # Porównanie metod mutacji (binarna)
    for mutation in ["random", "swap"]:
        print(f"\nTestowanie metody mutacji: {mutation}")
        ga_instance = run_binary_ga(
            function_name=function_name,
            num_genes_per_variable=num_genes_per_variable,
            num_vars=num_vars,
            var_range=var_range,
            population_size=population_size,
            num_generations=num_generations,
            mutation_type=mutation
        )
        solution, fitness, _ = ga_instance.best_solution()
        results.append({
            "representation": "Binary",
            "selection": "tournament",
            "crossover": "single_point",
            "mutation": mutation,
            "fitness": -fitness,
            "instance": ga_instance
        })
        plot_results(ga_instance, "Binary", f"Mutacja {mutation} - ")
    
    print("\n=== PORÓWNANIE METOD KRZYŻOWANIA (REPREZENTACJA RZECZYWISTA) ===")
    
    # Porównanie metod krzyżowania (rzeczywista)
    for crossover in ["arithmetic", "blend_alpha", "blend_alpha_beta", "linear", "average"]:
        print(f"\nTestowanie metody krzyżowania: {crossover}")
        ga_instance = run_real_ga(
            function_name=function_name,
            num_vars=num_vars,
            var_range=var_range,
            population_size=population_size,
            num_generations=num_generations,
            crossover_type=crossover
        )
        solution, fitness, _ = ga_instance.best_solution()
        results.append({
            "representation": "Real",
            "selection": "tournament",
            "crossover": crossover,
            "mutation": "adaptive",
            "fitness": -fitness,
            "instance": ga_instance
        })
        plot_results(ga_instance, "Real", f"Krzyżowanie {crossover} - ")
    
    print("\n=== PORÓWNANIE METOD MUTACJI (REPREZENTACJA RZECZYWISTA) ===")
    
    # Porównanie metod mutacji (rzeczywista)
    for mutation in ["adaptive", "gaussian"]:
        print(f"\nTestowanie metody mutacji: {mutation}")
        ga_instance = run_real_ga(
            function_name=function_name,
            num_vars=num_vars,
            var_range=var_range,
            population_size=population_size,
            num_generations=num_generations,
            mutation_type=mutation
        )
        solution, fitness, _ = ga_instance.best_solution()
        results.append({
            "representation": "Real",
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": mutation,
            "fitness": -fitness,
            "instance": ga_instance
        })
        plot_results(ga_instance, "Real", f"Mutacja {mutation} - ")
    
    # Podsumowanie wyników
    print("\n=== PODSUMOWANIE WYNIKÓW ===")
    print("{:<15} {:<15} {:<20} {:<15} {:<15}".format(
        "Reprezentacja", "Selekcja", "Krzyżowanie", "Mutacja", "Najlepsza wartość"
    ))
    print("-" * 80)
    
    sorted_results = sorted(results, key=lambda x: x["fitness"])
    for result in sorted_results:
        print("{:<15} {:<15} {:<20} {:<15} {:<15.6f}".format(
            result["representation"],
            result["selection"],
            result["crossover"],
            result["mutation"],
            result["fitness"]
        ))
    
    # Wygeneruj podsumowujący wykres porównawczy
    plt.figure(figsize=(12, 8))
    
    # Przygotuj dane do wykresu
    binary_results = [r for r in results if r["representation"] == "Binary"]
    real_results = [r for r in results if r["representation"] == "Real"]
    
    # Wykres dla reprezentacji binarnej
    labels_binary = [f"{r['selection']}/{r['crossover']}/{r['mutation']}" for r in binary_results]
    values_binary = [r["fitness"] for r in binary_results]
    
    # Wykres dla reprezentacji rzeczywistej
    labels_real = [f"{r['crossover']}/{r['mutation']}" for r in real_results]
    values_real = [r["fitness"] for r in real_results]
    
    # Rysuj wykres
    x_binary = np.arange(len(labels_binary))
    x_real = np.arange(len(labels_real)) + len(labels_binary) + 1
    
    plt.bar(x_binary, values_binary, color='blue', alpha=0.7, label='Binarna')
    plt.bar(x_real, values_real, color='green', alpha=0.7, label='Rzeczywista')
    
    plt.xlabel('Konfiguracja')
    plt.ylabel('Najlepsza wartość funkcji')
    plt.title(f'Porównanie konfiguracji dla funkcji {function_name}')
    plt.xticks(np.concatenate([x_binary, x_real]), 
               labels_binary + labels_real, 
               rotation=90)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/comparison_{function_name}.png')
    plt.show()
    
    return results

if __name__ == "__main__":
    # Przykład użycia
    print("Testowanie algorytmu genetycznego z reprezentacją binarną")
    binary_ga = run_binary_ga(
        function_name="Rastrigin",
        num_genes_per_variable=10,
        num_vars=5,
        var_range=(-5.12, 5.12),
        population_size=50,
        num_generations=100,
        crossover_type="single_point",
        mutation_type="random",
        parent_selection="tournament"
    )
    plot_results(binary_ga, "Binary")
    
    print("\nTestowanie algorytmu genetycznego z reprezentacją rzeczywistą")
    real_ga = run_real_ga(
        function_name="Rastrigin",
        num_vars=5,
        var_range=(-5.12, 5.12),
        population_size=50,
        num_generations=100,
        crossover_type="arithmetic",
        mutation_type="adaptive",
        parent_selection="tournament"
    )
    plot_results(real_ga, "Real")
    
    # Porównanie różnych metod
    compare_methods(
        function_name="Rastrigin",
        num_vars=5,
        var_range=(-5.12, 5.12),
        num_genes_per_variable=10,
        population_size=50,
        num_generations=100
    )