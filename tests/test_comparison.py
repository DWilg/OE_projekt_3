import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.real_genetic_algorithm import RealGeneticAlgorithm
from optimization.benchmark_functions import Rastrigin
from optimization.external_benchmark import create_benchmark_function, get_available_functions

def compare_configurations():

    if not os.path.exists('results'):
        os.makedirs('results')
        
    benchmark_functions = [
        "Rastrigin", 
        "Hyperellipsoid", 
        "CEC2014-F3"
    ]
    
    n_dimensions = 10
    
    binary_configurations = [
        {
            "name": "Binary: Konfiguracja bazowa",
            "representation": "Binary",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Jednopunktowe",
                "mutation": "Bit Flip",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        },
        {
            "name": "Binary: Wysoka mutacja",
            "representation": "Binary",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Jednopunktowe",
                "mutation": "Bit Flip",
                "mutation_rate": 0.2,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        },
        {
            "name": "Binary: Krzyżowanie dwupunktowe",
            "representation": "Binary",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Dwupunktowe",
                "mutation": "Bit Flip",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        }
    ]
    
    real_configurations = [
        {
            "name": "Real: Arytmetyczne + Równomierna",
            "representation": "Real",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Arytmetyczne",
                "mutation": "Równomierna",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        },
        {
            "name": "Real: Liniowe + Równomierna",
            "representation": "Real",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Liniowe",
                "mutation": "Równomierna",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        },
        {
            "name": "Real: Mieszające-alfa + Gaussa",
            "representation": "Real",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Mieszające-alfa",
                "mutation": "Gaussa",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        },
        {
            "name": "Real: Mieszające-alfa-beta + Gaussa",
            "representation": "Real",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Mieszające-alfa-beta",
                "mutation": "Gaussa",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        },
        {
            "name": "Real: Uśredniające + Gaussa",
            "representation": "Real",
            "params": {
                "selection": "Turniejowa",
                "crossover": "Uśredniające",
                "mutation": "Gaussa",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9,
                "population_size": 50,
                "num_generations": 100
            }
        }
    ]
    
    configurations = binary_configurations + real_configurations
    
    all_results = []
    
    for function_name in benchmark_functions:
        print(f"\n--- Testowanie funkcji: {function_name} ---\n")
        
        func, variable_range = create_benchmark_function(function_name, n_dimensions)
        
        results = []
        best_runs = []
        
        for config in configurations:
            config_best_values = []
            config_times = []
            all_runs_data = []
            
            print(f"Testowanie konfiguracji: {config['name']}...")
            
            for run in range(3):  
                start_time = time.time()
                
                if config["representation"] == "Binary":
                    ga = GeneticAlgorithm(
                        population_size=config["params"]["population_size"],
                        num_generations=config["params"]["num_generations"],
                        mutation_rate=config["params"]["mutation_rate"],
                        crossover_rate=config["params"]["crossover_rate"],
                        num_variables=n_dimensions,
                        selection_method=config["params"]["selection"],
                        crossover_method=config["params"]["crossover"],
                        mutation_method=config["params"]["mutation"],
                        variable_range=variable_range
                    )
                else: 
                    ga = RealGeneticAlgorithm(
                        population_size=config["params"]["population_size"],
                        num_generations=config["params"]["num_generations"],
                        mutation_rate=config["params"]["mutation_rate"],
                        crossover_rate=config["params"]["crossover_rate"],
                        num_variables=n_dimensions,
                        selection_method=config["params"]["selection"],
                        crossover_method=config["params"]["crossover"],
                        mutation_method=config["params"]["mutation"],
                        variable_range=variable_range
                    )
                    
                ga.evolve(func)
                final_best = min(ga.best_values)
                config_best_values.append(final_best)
                elapsed_time = time.time() - start_time
                config_times.append(elapsed_time)
                
                all_runs_data.append({
                    "best_values": ga.best_values.copy(),
                    "std_deviation": [np.std(fitness) for fitness in ga.all_fitness_values],
                    "time": elapsed_time
                })
            
            mean_value = np.mean(config_best_values)
            best_value = min(config_best_values)
            worst_value = max(config_best_values)
            mean_time = np.mean(config_times)
            std_time = np.std(config_times)
            
            best_run_idx = np.argmin(config_best_values)
            best_runs.append({
                "config_name": config["name"],
                "representation": config["representation"],
                "best_values": all_runs_data[best_run_idx]["best_values"],
                "std_deviation": all_runs_data[best_run_idx]["std_deviation"],
                "time": all_runs_data[best_run_idx]["time"]
            })
            
            result = {
                "function": function_name,
                "config_name": config["name"],
                "representation": config["representation"],
                "mean_value": mean_value,
                "best_value": best_value,
                "worst_value": worst_value,
                "mean_time": mean_time,
                "std_time": std_time,
                "all_values": config_best_values
            }
            
            results.append(result)
            all_results.append(result)
        
        generate_comparison_plots(results, best_runs, function_name)
    
    save_results_to_csv(all_results)
    
    generate_binary_vs_real_comparison(all_results)
    
    return all_results

def generate_comparison_plots(results, best_runs, function_name):

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    bar_colors = ['blue' if res["representation"] == "Binary" else 'green' for res in results]
    bars = plt.bar([res["config_name"] for res in results], [res["mean_value"] for res in results], color=bar_colors)
    plt.title(f"Średnie wartości funkcji celu - {function_name}", fontsize=12)
    plt.ylabel("Wartość funkcji", fontsize=10)
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 2, 2)
    bars = plt.bar([res["config_name"] for res in results], [res["mean_time"] for res in results],
            yerr=[res["std_time"] for res in results], capsize=5, color=bar_colors)
    plt.title("Średni czas wykonania", fontsize=12)
    plt.ylabel("Czas [s]", fontsize=10)
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}s',
                 ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 2, 3)
    binary_runs = [run for run in best_runs if run["representation"] == "Binary"]
    real_runs = [run for run in best_runs if run["representation"] == "Real"]
    
    for run in binary_runs:
        plt.plot(run["best_values"], label=run["config_name"], linestyle='-')
    
    for run in real_runs:
        plt.plot(run["best_values"], label=run["config_name"], linestyle='--')
    
    plt.title("Zbieżność algorytmu (najlepsze uruchomienie)", fontsize=12)
    plt.xlabel("Generacja", fontsize=10)
    plt.ylabel("Wartość funkcji", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8, loc='upper right')
    
    plt.subplot(2, 2, 4)
    for run in binary_runs:
        plt.plot(run["std_deviation"], label=run["config_name"], linestyle='-')
    
    for run in real_runs:
        plt.plot(run["std_deviation"], label=run["config_name"], linestyle='--')
        
    plt.title("Odchylenie standardowe populacji (najlepsze uruchomienie)", fontsize=12)
    plt.xlabel("Generacja", fontsize=10)
    plt.ylabel("Odchylenie standardowe", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    plot_path = os.path.join('results', f'comparison_results_{function_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv(all_results):

    csv_path = os.path.join('results', "comparison_results.csv")
    with open(csv_path, "w", encoding='utf-8') as f:
        f.write("Funkcja,Konfiguracja,Reprezentacja,Średnia wartość,Najlepsza wartość,Najgorsza wartość,Średni czas [s],Odchylenie czasu [s]\n")
        for res in all_results:
            f.write(f"{res['function']},{res['config_name']},{res['representation']},{res['mean_value']:.4f},{res['best_value']:.4f},{res['worst_value']:.4f},{res['mean_time']:.4f},{res['std_time']:.4f}\n")

def generate_binary_vs_real_comparison(all_results):

    functions = list(set([res["function"] for res in all_results]))
    
    comparison_data = []
    for function in functions:
        function_results = [res for res in all_results if res["function"] == function]
        
        binary_results = [res for res in function_results if res["representation"] == "Binary"]
        real_results = [res for res in function_results if res["representation"] == "Real"]
        
        binary_mean = np.mean([res["mean_value"] for res in binary_results])
        real_mean = np.mean([res["mean_value"] for res in real_results])
        
        binary_best = np.min([res["best_value"] for res in binary_results])
        real_best = np.min([res["best_value"] for res in real_results])
        
        binary_time = np.mean([res["mean_time"] for res in binary_results])
        real_time = np.mean([res["mean_time"] for res in real_results])
        
        comparison_data.append({
            "function": function,
            "binary_mean": binary_mean,
            "real_mean": real_mean,
            "binary_best": binary_best,
            "real_best": real_best,
            "binary_time": binary_time,
            "real_time": real_time
        })
    
    fig, axes = plt.subplots(len(functions), 3, figsize=(15, 5 * len(functions)))
    
    for i, data in enumerate(comparison_data):
        axes[i, 0].bar(['Binary', 'Real'], [data["binary_mean"], data["real_mean"]], color=['blue', 'green'])
        axes[i, 0].set_title(f"{data['function']} - Średnie wartości")
        axes[i, 0].set_ylabel("Wartość funkcji")
        
        axes[i, 1].bar(['Binary', 'Real'], [data["binary_best"], data["real_best"]], color=['blue', 'green'])
        axes[i, 1].set_title(f"{data['function']} - Najlepsze wartości")
        
        axes[i, 2].bar(['Binary', 'Real'], [data["binary_time"], data["real_time"]], color=['blue', 'green'])
        axes[i, 2].set_title(f"{data['function']} - Średni czas wykonania")
        axes[i, 2].set_ylabel("Czas [s]")
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'binary_vs_real_comparison_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df["mean_improvement"] = (comparison_df["binary_mean"] - comparison_df["real_mean"]) / comparison_df["binary_mean"] * 100
    comparison_df["best_improvement"] = (comparison_df["binary_best"] - comparison_df["real_best"]) / comparison_df["binary_best"] * 100
    comparison_df["time_ratio"] = comparison_df["real_time"] / comparison_df["binary_time"]
    
    comparison_summary = comparison_df[["function", "mean_improvement", "best_improvement", "time_ratio"]]
    print("\n--- Podsumowanie porównania reprezentacji binarnej vs rzeczywistej ---")
    print(comparison_summary)
    
    comparison_summary.to_csv(os.path.join('results', 'binary_vs_real_summary.csv'), index=False)

if __name__ == "__main__":
    compare_configurations()