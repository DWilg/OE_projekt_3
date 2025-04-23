import argparse
import os
import sys
import numpy as np
from pygad_implementation import (
    run_binary_ga,
    run_real_ga,
    compare_methods,
    plot_results
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Algorytm Genetyczny z PyGAD')
    
    parser.add_argument('--mode', type=str, default='binary',
                      choices=['binary', 'real', 'compare', 'all'],
                      help='Tryb działania: binary, real, compare lub all')
    
    parser.add_argument('--function', type=str, default='Rastrigin',
                      choices=['Rastrigin', 'Hypersphere', 'Hyperellipsoid', 'CEC2014-F3', 'CEC2014-F5'],
                      help='Funkcja testowa')
    
    parser.add_argument('--dims', type=int, default=10,
                      help='Liczba wymiarów')
    
    parser.add_argument('--pop-size', type=int, default=50,
                      help='Rozmiar populacji')
    
    parser.add_argument('--generations', type=int, default=100,
                      help='Liczba generacji')
    
    parser.add_argument('--bits', type=int, default=20,
                      help='Liczba bitów na zmienną (dla reprezentacji binarnej)')
    
    parser.add_argument('--parent-selection', type=str, default='tournament',
                      choices=['tournament', 'rws', 'random'],
                      help='Metoda selekcji rodziców')
    
    parser.add_argument('--binary-crossover', type=str, default='single_point',
                      choices=['single_point', 'two_points', 'uniform'],
                      help='Metoda krzyżowania dla reprezentacji binarnej')
    
    parser.add_argument('--real-crossover', type=str, default='arithmetic',
                      choices=['arithmetic', 'blend_alpha', 'blend_alpha_beta', 'linear', 'average'],
                      help='Metoda krzyżowania dla reprezentacji rzeczywistej')
    
    parser.add_argument('--binary-mutation', type=str, default='random',
                      choices=['random', 'swap'],
                      help='Metoda mutacji dla reprezentacji binarnej')
    
    parser.add_argument('--real-mutation', type=str, default='adaptive',
                      choices=['adaptive', 'gaussian'],
                      help='Metoda mutacji dla reprezentacji rzeczywistej')
    
    return parser.parse_args()

def print_separator():
    print("\n" + "=" * 80 + "\n")

def print_configuration(args, representation):
    print_separator()
    print(f"KONFIGURACJA ALGORYTMU GENETYCZNEGO ({representation.upper()})")
    print(f"Funkcja testowa: {args.function}")
    print(f"Liczba wymiarów: {args.dims}")
    print(f"Rozmiar populacji: {args.pop_size}")
    print(f"Liczba generacji: {args.generations}")
    
    if representation == "binary":
        print(f"Liczba bitów na zmienną: {args.bits}")
        print(f"Metoda selekcji: {args.parent_selection}")
        print(f"Metoda krzyżowania: {args.binary_crossover}")
        print(f"Metoda mutacji: {args.binary_mutation}")
    else:
        print(f"Metoda selekcji: {args.parent_selection}")
        print(f"Metoda krzyżowania: {args.real_crossover}")
        print(f"Metoda mutacji: {args.real_mutation}")
    print_separator()

def run_binary_experiment(args):
    print_configuration(args, "binary")
    
    # Określenie zakresu zmiennych na podstawie funkcji testowej
    if args.function == "Rastrigin":
        var_range = (-5.12, 5.12)
    elif args.function == "Hypersphere":
        var_range = (-5.12, 5.12)
    elif args.function == "Hyperellipsoid":
        var_range = (-5.12, 5.12)
    else:  # Funkcje CEC
        var_range = (-100.0, 100.0)
    
    # Uruchomienie algorytmu genetycznego z reprezentacją binarną
    ga_instance = run_binary_ga(
        function_name=args.function,
        num_genes_per_variable=args.bits,
        num_vars=args.dims,
        var_range=var_range,
        population_size=args.pop_size,
        num_generations=args.generations,
        crossover_type=args.binary_crossover,
        mutation_type=args.binary_mutation,
        parent_selection=args.parent_selection
    )
    
    # Wizualizacja wyników
    plot_results(ga_instance, "Binary", f"{args.function} - ")
    
    return ga_instance

def run_real_experiment(args):
    print_configuration(args, "real")
    
    # Określenie zakresu zmiennych na podstawie funkcji testowej
    if args.function == "Rastrigin":
        var_range = (-5.12, 5.12)
    elif args.function == "Hypersphere":
        var_range = (-5.12, 5.12)
    elif args.function == "Hyperellipsoid":
        var_range = (-5.12, 5.12)
    else:  # Funkcje CEC
        var_range = (-100.0, 100.0)
    
    # Uruchomienie algorytmu genetycznego z reprezentacją rzeczywistą
    ga_instance = run_real_ga(
        function_name=args.function,
        num_vars=args.dims,
        var_range=var_range,
        population_size=args.pop_size,
        num_generations=args.generations,
        crossover_type=args.real_crossover,
        mutation_type=args.real_mutation,
        parent_selection=args.parent_selection
    )
    
    # Wizualizacja wyników
    plot_results(ga_instance, "Real", f"{args.function} - ")
    
    return ga_instance

def run_comparison(args):
    print_separator()
    print(f"PORÓWNANIE METOD DLA FUNKCJI {args.function}")
    print(f"Liczba wymiarów: {args.dims}")
    print(f"Rozmiar populacji: {args.pop_size}")
    print(f"Liczba generacji: {args.generations}")
    print_separator()
    
    # Określenie zakresu zmiennych na podstawie funkcji testowej
    if args.function == "Rastrigin":
        var_range = (-5.12, 5.12)
    elif args.function == "Hypersphere":
        var_range = (-5.12, 5.12)
    elif args.function == "Hyperellipsoid":
        var_range = (-5.12, 5.12)
    else:  # Funkcje CEC
        var_range = (-100.0, 100.0)
    
    # Uruchomienie porównania metod
    results = compare_methods(
        function_name=args.function,
        num_vars=args.dims,
        var_range=var_range,
        num_genes_per_variable=args.bits,
        population_size=args.pop_size,
        num_generations=args.generations
    )
    
    return results

def main():
    args = parse_arguments()
    
    # Tworzenie katalogu wyników, jeśli nie istnieje
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if args.mode == 'binary':
        run_binary_experiment(args)
    elif args.mode == 'real':
        run_real_experiment(args)
    elif args.mode == 'compare':
        run_comparison(args)
    elif args.mode == 'all':
        # Najpierw uruchom eksperymenty z pojedynczą konfiguracją
        binary_ga = run_binary_experiment(args)
        real_ga = run_real_experiment(args)
        
        # Następnie uruchom porównanie metod
        run_comparison(args)
    
    print("\nWszystkie eksperymenty zakończone. Wyniki zapisano w katalogu 'results'.")

if __name__ == "__main__":
    main()