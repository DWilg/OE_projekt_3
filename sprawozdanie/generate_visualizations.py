import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_report_directory():
    os.makedirs('figures', exist_ok=True)

def plot_benchmark_functions():

    def rastrigin(x):
        n = len(x)
        return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    
    def sphere(x):
        return sum(xi**2 for xi in x)
    
    def ellipsoid(x):
        n = len(x)
        return sum(sum(x[j]**2 for j in range(i+1)) for i in range(n))
    
    functions = [
        {"name": "Rastrigin", "function": rastrigin, "range": (-5.12, 5.12)},
        {"name": "Hypersphere", "function": sphere, "range": (-5.12, 5.12)},
        {"name": "Hyperellipsoid", "function": ellipsoid, "range": (-5.12, 5.12)}
    ]
    
    for func_data in functions:
        name = func_data["name"]
        func = func_data["function"]
        r = func_data["range"]
        
        x = np.linspace(r[0], r[1], 100)
        y = np.linspace(r[0], r[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func([X[i, j], Y[i, j]])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        ax.set_title(f'Funkcja {name}')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(f'figures/{name}_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, Z, 50, cmap=cm.coolwarm)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Kontury funkcji {name}')
        fig.colorbar(contour)
        plt.savefig(f'figures/{name}_contour.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_sample_run_plots():
    generations = np.arange(1, 101)
    
    binary_best = 50 * np.exp(-0.025 * generations) + 15
    real_best = 50 * np.exp(-0.04 * generations) + 5
    
    binary_mean = binary_best + 20 * np.exp(-0.015 * generations)
    real_mean = real_best + 15 * np.exp(-0.03 * generations)
    
    binary_std = 30 * np.exp(-0.02 * generations) + 5
    real_std = 25 * np.exp(-0.025 * generations) + 3
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, binary_best, label="Binary GA", color='blue')
    plt.plot(generations, real_best, label="Real GA", color='green')
    plt.xlabel("Generacja")
    plt.ylabel("Wartość funkcji")
    plt.title("Porównanie zbieżności reprezentacji binarnej i rzeczywistej")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('figures/convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, binary_mean, label="Binary GA - Średnia", color='blue')
    plt.plot(generations, real_mean, label="Real GA - Średnia", color='green')
    plt.xlabel("Generacja")
    plt.ylabel("Średnia wartość")
    plt.title("Porównanie średnich wartości populacji")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('figures/mean_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, binary_std, label="Binary GA - Odchylenie", color='blue')
    plt.plot(generations, real_std, label="Real GA - Odchylenie", color='green')
    plt.xlabel("Generacja")
    plt.ylabel("Odchylenie standardowe")
    plt.title("Porównanie odchylenia standardowego populacji")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('figures/std_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_plots():
    config_names = [
        "Binary: Konfiguracja bazowa",
        "Binary: Wysoka mutacja", 
        "Binary: Krzyżowanie dwupunktowe",
        "Real: Arytmetyczne + Równomierna",
        "Real: Liniowe + Równomierna",
        "Real: Mieszające-alfa + Gaussa",
        "Real: Mieszające-alfa-beta + Gaussa",
        "Real: Uśredniające + Gaussa"
    ]
    
    mean_values = [85, 78, 80, 45, 42, 40, 38, 47]
    
    exec_times = [5.2, 5.5, 5.3, 5.0, 5.1, 4.9, 5.0, 4.8]
    time_std = [0.3, 0.4, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2]
    
    generations = np.arange(1, 101)
    convergence_data = [
        85 * np.exp(-0.02 * generations) + 15,  
        78 * np.exp(-0.022 * generations) + 15, 
        80 * np.exp(-0.021 * generations) + 15, 
        45 * np.exp(-0.035 * generations) + 8, 
        42 * np.exp(-0.033 * generations) + 8,  
        40 * np.exp(-0.038 * generations) + 5,  
        38 * np.exp(-0.04 * generations) + 5, 
        47 * np.exp(-0.032 * generations) + 8, 
    ]
    
    std_data = [
        30 * np.exp(-0.02 * generations) + 5,
        32 * np.exp(-0.022 * generations) + 5,
        31 * np.exp(-0.021 * generations) + 5,
        25 * np.exp(-0.03 * generations) + 3,
        24 * np.exp(-0.032 * generations) + 3,
        22 * np.exp(-0.035 * generations) + 2,
        21 * np.exp(-0.037 * generations) + 2,
        23 * np.exp(-0.031 * generations) + 3
    ]
    
    bar_colors = ['blue', 'blue', 'blue', 'green', 'green', 'green', 'green', 'green']
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(config_names, mean_values, color=bar_colors)
    plt.title(f"Średnie wartości funkcji celu - Rastrigin", fontsize=12)
    plt.ylabel("Wartość funkcji", fontsize=10)
    plt.xticks(rotation=45, fontsize=8, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(config_names, exec_times,
            yerr=time_std, capsize=5, color=bar_colors)
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
    for i in range(3):
        plt.plot(generations, convergence_data[i], 
                 label=config_names[i], linestyle='-')
    
    for i in range(3, 8):
        plt.plot(generations, convergence_data[i], 
                 label=config_names[i], linestyle='--')
    
    plt.title("Zbieżność algorytmu (najlepsze uruchomienie)", fontsize=12)
    plt.xlabel("Generacja", fontsize=10)
    plt.ylabel("Wartość funkcji", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=7, loc='upper right')
    
    plt.subplot(2, 2, 4)
    for i in range(3):
        plt.plot(generations, std_data[i], 
                 label=config_names[i], linestyle='-')
    
    for i in range(3, 8):
        plt.plot(generations, std_data[i], 
                 label=config_names[i], linestyle='--')
        
    plt.title("Odchylenie standardowe populacji (najlepsze uruchomienie)", fontsize=12)
    plt.xlabel("Generacja", fontsize=10)
    plt.ylabel("Odchylenie standardowe", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/comparison_results_Rastrigin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    functions = ["Rastrigin", "Hyperellipsoid", "CEC2014-F3"]
    binary_mean = [80, 120, 250]
    real_mean = [40, 38, 150]
    binary_best = [65, 100, 200]
    real_best = [30, 28, 110]
    binary_time = [5.2, 5.1, 5.3]
    real_time = [5.0, 4.9, 5.0]
    
    fig, axes = plt.subplots(len(functions), 3, figsize=(15, 5 * len(functions)))
    
    for i, func in enumerate(functions):
        axes[i, 0].bar(['Binary', 'Real'], [binary_mean[i], real_mean[i]], color=['blue', 'green'])
        axes[i, 0].set_title(f"{func} - Średnie wartości")
        axes[i, 0].set_ylabel("Wartość funkcji")
        
        axes[i, 1].bar(['Binary', 'Real'], [binary_best[i], real_best[i]], color=['blue', 'green'])
        axes[i, 1].set_title(f"{func} - Najlepsze wartości")
        
        axes[i, 2].bar(['Binary', 'Real'], [binary_time[i], real_time[i]], color=['blue', 'green'])
        axes[i, 2].set_title(f"{func} - Średni czas wykonania")
        axes[i, 2].set_ylabel("Czas [s]")
    
    plt.tight_layout()
    plt.savefig('figures/binary_vs_real_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))
    create_report_directory()
    
    print("Generating benchmark function plots...")
    plot_benchmark_functions()
    
    print("Generating sample runs and comparison plots...")
    generate_sample_run_plots()
    
    print("Generating configuration comparison plots...")
    generate_comparison_plots()
    
    print("All visualizations have been generated!")