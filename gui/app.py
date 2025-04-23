import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.real_genetic_algorithm import RealGeneticAlgorithm
from optimization.benchmark_functions import \
    Hypersphere, Rastrigin, \
    Hyperellipsoid
import numpy as np
from utils import save_to_database

def run_algorithm(population_size, num_generations, mutation_rate, crossover_rate, inversion_rate,
                  selection_method, crossover_method, mutation_method, test_function_name,
                  num_variables, optimization_goal, representation_type):
    hypersphere = Hypersphere(n_dimensions=num_variables, opposite=False)
    rastrigin = Rastrigin(n_dimensions=num_variables, opposite=False)
    hyperellipsoid = Hyperellipsoid(n_dimensions=num_variables, opposite=False)
    
    def test_function(individual):
        if test_function_name == "Rastrigin":
            return rastrigin._evaluate(individual)
        elif test_function_name == "Hypersphere":
            return hypersphere._evaluate(individual)
        elif test_function_name == "Hyperellipsoid":
            return hyperellipsoid._evaluate(individual)

    if representation_type == "Binary":
        ga = GeneticAlgorithm(
            population_size=population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            num_variables=num_variables,
            inversion_rate=inversion_rate,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            optimization_goal=optimization_goal
        )
    else:  
        ga = RealGeneticAlgorithm(
            population_size=population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            num_variables=num_variables,
            inversion_rate=inversion_rate,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            optimization_goal=optimization_goal
        )

    ga.evolve(test_function)
    messagebox.showinfo("Info", "Optymalizacja zakończona")
    return ga

def plot_results(ga):
    generations = list(range(1, ga.num_generations + 1))
    best_values = ga.best_values

    all_fitness_values = ga.all_fitness_values

    mean_values = [np.mean(fitness) for fitness in all_fitness_values]
    std_deviation = [np.std(fitness) for fitness in all_fitness_values]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(generations, best_values, label="Najlepsza wartość funkcji celu", color='blue')
    plt.xlabel("Generacja")
    plt.ylabel("Wartość funkcji")
    plt.title("Wartości funkcji celu w czasie")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(generations, mean_values, label="Średnia wartość", color='green')
    plt.plot(generations, std_deviation, label="Odchylenie standardowe", color='red')
    plt.xlabel("Generacja")
    plt.ylabel("Wartość")
    plt.title("Statystyki populacji")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_gui():
    root = tk.Tk()
    root.title("Algorytm Genetyczny")
    
    left_frame = tk.Frame(root, padx=10, pady=10)
    left_frame.grid(row=0, column=0, sticky='n')
    
    right_frame = tk.Frame(root, padx=10, pady=10)
    right_frame.grid(row=0, column=1, sticky='n')
    
    tk.Label(left_frame, text="PARAMETRY PODSTAWOWE", font=("Helvetica", 10, "bold")).pack(pady=5)
    
    tk.Label(left_frame, text="Typ reprezentacji:").pack()
    representation_type = ttk.Combobox(left_frame, values=["Binary", "Real"])
    representation_type.set("Binary")
    representation_type.pack(pady=2)
    
    tk.Label(left_frame, text="Rozmiar populacji:").pack()
    population_size_entry = tk.Entry(left_frame)
    population_size_entry.insert(tk.END, "50")
    population_size_entry.pack(pady=2)

    tk.Label(left_frame, text="Liczba generacji:").pack()
    num_generations_entry = tk.Entry(left_frame)
    num_generations_entry.insert(tk.END, "100")
    num_generations_entry.pack(pady=2)

    tk.Label(left_frame, text="Prawdopodobieństwo mutacji:").pack()
    mutation_rate_entry = tk.Entry(left_frame)
    mutation_rate_entry.insert(tk.END, "0.05")
    mutation_rate_entry.pack(pady=2)

    tk.Label(left_frame, text="Prawdopodobieństwo krzyżowania:").pack()
    crossover_rate_entry = tk.Entry(left_frame)
    crossover_rate_entry.insert(tk.END, "0.8")
    crossover_rate_entry.pack(pady=2)

    tk.Label(left_frame, text="Prawdopodobieństwo inwersji:").pack()
    inversion_rate_entry = tk.Entry(left_frame)
    inversion_rate_entry.insert(tk.END, "0.1")
    inversion_rate_entry.pack(pady=2)

    tk.Label(left_frame, text="Liczba zmiennych:").pack()
    num_variables_entry = tk.Entry(left_frame)
    num_variables_entry.insert(tk.END, "10")
    num_variables_entry.pack(pady=2)

    tk.Label(left_frame, text="Cel optymalizacji:").pack()
    optimization_goal = ttk.Combobox(left_frame, values=["min", "max"])
    optimization_goal.set("min")
    optimization_goal.pack(pady=2)

    tk.Label(right_frame, text="OPERATORY GENETYCZNE", font=("Helvetica", 10, "bold")).pack(pady=5)
    
    tk.Label(right_frame, text="Wybierz funkcję testową:").pack()
    test_function_frame = tk.Frame(right_frame)
    test_function_frame.pack(pady=2)
    
    test_function = tk.StringVar(value="Rastrigin")
    tk.Radiobutton(test_function_frame, text="Rastrigin", variable=test_function, value="Rastrigin").pack(anchor='w')
    tk.Radiobutton(test_function_frame, text="Hypersphere", variable=test_function, value="Hypersphere").pack(anchor='w')
    tk.Radiobutton(test_function_frame, text="Hyperellipsoid", variable=test_function, value="Hyperellipsoid").pack(anchor='w')

    tk.Label(right_frame, text="Metoda selekcji:").pack()
    selection_method = ttk.Combobox(right_frame, values=["Turniejowa", "Koło ruletki"])
    selection_method.set("Turniejowa")
    selection_method.pack(pady=2)

    binary_crossover_values = ["Jednopunktowe", "Dwupunktowe", "Jednorodne", "Ziarniste"]
    real_crossover_values = ["Arytmetyczne", "Liniowe", "Mieszające-alfa", "Mieszające-alfa-beta", "Uśredniające"]
    
    binary_mutation_values = ["Bit Flip", "Brzegowa", "Dwupunktowa"]
    real_mutation_values = ["Równomierna", "Gaussa"]
    
    tk.Label(right_frame, text="Metoda krzyżowania:").pack()
    crossover_method = ttk.Combobox(right_frame, values=binary_crossover_values)
    crossover_method.set("Jednopunktowe")
    crossover_method.pack(pady=2)

    tk.Label(right_frame, text="Metoda mutacji:").pack()
    mutation_method = ttk.Combobox(right_frame, values=binary_mutation_values)
    mutation_method.set("Bit Flip")
    mutation_method.pack(pady=2)
    
    def update_operators(*args):
        if representation_type.get() == "Binary":
            crossover_method['values'] = binary_crossover_values
            if crossover_method.get() not in binary_crossover_values:
                crossover_method.set(binary_crossover_values[0])
                
            mutation_method['values'] = binary_mutation_values
            if mutation_method.get() not in binary_mutation_values:
                mutation_method.set(binary_mutation_values[0])
        else:   
            crossover_method['values'] = real_crossover_values
            if crossover_method.get() not in real_crossover_values:
                crossover_method.set(real_crossover_values[0])
                
            mutation_method['values'] = real_mutation_values
            if mutation_method.get() not in real_mutation_values:
                mutation_method.set(real_mutation_values[0])
    
    representation_type.bind('<<ComboboxSelected>>', update_operators)
    
    buttons_frame = tk.Frame(root, padx=10, pady=10)
    buttons_frame.grid(row=1, column=0, columnspan=2)
    
    def start_algorithm():
        try:
            population_size = int(population_size_entry.get())
            if population_size <= 0:
                raise ValueError("Rozmiar populacji musi być liczbą całkowitą większą od 0.")

            num_generations = int(num_generations_entry.get())
            if num_generations <= 0:
                raise ValueError("Liczba generacji musi być liczbą całkowitą większą od 0.")

            num_variables = int(num_variables_entry.get())
            if num_variables <= 0:
                raise ValueError("Liczba zmiennych musi być liczbą całkowitą większą od 0.")

            mutation_rate = float(mutation_rate_entry.get())
            if not (0 <= mutation_rate <= 1):
                raise ValueError("Prawdopodobieństwo mutacji musi być liczbą z zakresu [0, 1].")

            crossover_rate = float(crossover_rate_entry.get())
            if not (0 <= crossover_rate <= 1):
                raise ValueError("Prawdopodobieństwo krzyżowania musi być liczbą z zakresu [0, 1].")

            inversion_rate = float(inversion_rate_entry.get())
            if not (0 <= inversion_rate <= 1):
                raise ValueError("Prawdopodobieństwo inwersji musi być liczbą z zakresu [0, 1].")
                
            selected_representation = representation_type.get()
            selected_selection_method = selection_method.get()
            selected_crossover_method = crossover_method.get()
            selected_mutation_method = mutation_method.get()
            selected_test_function = test_function.get()
            selected_optimization_goal = optimization_goal.get()
            
            global ga
            ga = run_algorithm(
                population_size, num_generations, mutation_rate, crossover_rate, inversion_rate,
                selected_selection_method, selected_crossover_method, selected_mutation_method,
                selected_test_function, num_variables, selected_optimization_goal, selected_representation
            )
            plot_results(ga)
            save_to_database.save_results_to_db(ga)

        except ValueError as e:
            messagebox.showerror("Błąd danych wejściowych", str(e))
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")

    btn = tk.Button(buttons_frame, text="Start", command=start_algorithm, width=15)
    btn.pack(side=tk.LEFT, padx=5)

    plot_btn = tk.Button(buttons_frame, text="Pokaż wykres", command=lambda: plot_results(ga), width=15)
    plot_btn.pack(side=tk.LEFT, padx=5)
    
    def run_comparison_tests():
        from tests.test_comparison import compare_configurations
        compare_configurations()
    
    compare_btn = tk.Button(buttons_frame, text="Testy porównawcze", command=run_comparison_tests, width=15)
    compare_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    run_gui()