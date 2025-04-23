import benchmark_functions as bf
from opfunu import cec_based
import numpy as np

def create_benchmark_function(function_name, n_dimensions=10):
    """
    Tworzy funkcję testową z zewnętrznej biblioteki benchmark_functions lub opfunu
    
    Args:
        function_name (str): Nazwa funkcji testowej
        n_dimensions (int): Liczba wymiarów
        
    Returns:
        function: Funkcja testowa
        tuple: Zakres zmiennych (lb, ub)
    """
    if function_name == "Hyperellipsoid":
        func = bf.Hyperellipsoid(n_dimensions=n_dimensions)
        bounds = func.suggested_bounds()
        return func._evaluate, bounds
    elif function_name == "Rastrigin":
        func = bf.Rastrigin(n_dimensions=n_dimensions)
        bounds = func.suggested_bounds()
        return func._evaluate, bounds
    elif function_name == "Sphere":
        func = bf.Hypersphere(n_dimensions=n_dimensions)
        bounds = func.suggested_bounds()
        return func._evaluate, bounds
    elif function_name == "CEC2014-F3":
        func = cec_based.cec2014.F32014(ndim=n_dimensions)
        return func.evaluate, func.bounds
    elif function_name == "CEC2014-F5":
        func = cec_based.cec2014.F52014(ndim=n_dimensions)
        return func.evaluate, func.bounds
    elif function_name == "CEC2014-F10":
        func = cec_based.cec2014.F102014(ndim=n_dimensions)
        return func.evaluate, func.bounds
    else:
        raise ValueError(f"Nieznana funkcja testowa: {function_name}")

def get_available_functions():

    return [
        {
            "name": "Hyperellipsoid",
            "source": "benchmark_functions",
            "description": "Funkcja Hyperellipsoid (obracany hyperellipsoid)",
            "dimensions": [2, 5, 10, 30, 50, 100]
        },
        {
            "name": "Rastrigin",
            "source": "benchmark_functions",
            "description": "Funkcja Rastrigin - wielomodalna z wieloma minimami lokalnymi",
            "dimensions": [2, 5, 10, 30, 50, 100]
        },
        {
            "name": "Sphere",
            "source": "benchmark_functions",
            "description": "Funkcja Sphere - najprostsza funkcja do optymalizacji",
            "dimensions": [2, 5, 10, 30, 50, 100]
        },
        {
            "name": "CEC2014-F3",
            "source": "opfunu",
            "description": "CEC 2014 Benchmark Function F3 - Rotated Discus Function",
            "dimensions": [10, 30, 50, 100]
        },
        {
            "name": "CEC2014-F5",
            "source": "opfunu",
            "description": "CEC 2014 Benchmark Function F5 - Rotated Katsuura Function",
            "dimensions": [10, 30, 50, 100]
        },
        {
            "name": "CEC2014-F10",
            "source": "opfunu",
            "description": "CEC 2014 Benchmark Function F10 - Hybrid Function 1",
            "dimensions": [10, 30, 50, 100]
        }
    ]

def test_functions():

    print("Testowanie dostępnych funkcji:")
    
    func = bf.Hyperellipsoid(n_dimensions=10)
    print(f"Hyperellipsoid bounds: {func.suggested_bounds()}")
    print(f"Hyperellipsoid minimum: {func.minimum()}")
    
    func = cec_based.cec2014.F32014(ndim=10)
    print(f"CEC2014-F3 bounds: {func.bounds}")
    print(f"CEC2014-F3 global optimum: x={func.x_global}, f={func.f_global}")
    
    return get_available_functions()

if __name__ == "__main__":
    test_functions()