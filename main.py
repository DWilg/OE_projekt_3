import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projekt 3 - Algorytmy Genetyczne z PyGAD")
    parser.add_argument("--mode", choices=["example", "pygad", "gui", "testy"], default="example",
                        help="Tryb uruchomienia programu")
    parser.add_argument("--representation", choices=["binary", "real", "compare", "all"], default="all",
                        help="Typ reprezentacji (tylko dla trybu 'pygad')")
    parser.add_argument("--function", choices=["Rastrigin", "Hypersphere", "Hyperellipsoid", "CEC2014-F3", "CEC2014-F5"], 
                        default="Rastrigin", help="Funkcja testowa (tylko dla trybu 'pygad')")
    parser.add_argument("--dims", type=int, default=10, help="Liczba wymiarów (tylko dla trybu 'pygad')")
    
    args = parser.parse_args()

    if args.mode == "gui":
        from gui.app import run_gui  
        run_gui()
    elif args.mode == "testy":
        from tests.test_comparison import compare_configurations 
        compare_configurations()
    elif args.mode == "example":
        # Uruchom przykład z funkcją (x + 2y - 7)^2 + (2x + y - 5)^2
        from example_function import run_example_function
        run_example_function()
    elif args.mode == "pygad":
        # Uruchom eksperymenty z PyGAD
        import sys
        from run_pygad_experiments import main as run_experiments
        sys.argv = [sys.argv[0]] + ["--mode", args.representation, "--function", args.function, "--dims", str(args.dims)]
        run_experiments()
    else:
        print(f"Nieznany tryb: {args.mode}")