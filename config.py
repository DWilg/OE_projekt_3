
DEFAULT_CONFIG = {
    "population_size": 50,
    "num_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "inversion_rate": 0.1,
    "elitism_rate": 0.01,
    "optimization_goal": "min",
    
    "precision": 10, 
    
    "variable_range": (-5.12, 5.12),
    
    "benchmark_functions": {
        "Rastrigin": {
            "description": "Funkcja Rastrigina (minimum globalne w 0, wartość 0)",
            "dimensions": [2, 5, 10, 30],
            "default_dimension": 10
        },
        "Hypersphere": {
            "description": "Funkcja hipersfery (minimum globalne w 0, wartość 0)",
            "dimensions": [2, 5, 10, 30],
            "default_dimension": 10
        },
        "Hyperellipsoid": {
            "description": "Funkcja hiperelipsoidy (minimum globalne w 0, wartość 0)",
            "dimensions": [2, 5, 10, 30],
            "default_dimension": 10
        }
    },

    "binary_operators": {
        "selection": ["Turniejowa", "Koło ruletki"],
        "crossover": ["Jednopunktowe", "Dwupunktowe", "Jednorodne", "Ziarniste"],
        "mutation": ["Bit Flip", "Brzegowa", "Dwupunktowa"]
    },
    
    "real_operators": {
        "selection": ["Turniejowa", "Koło ruletki"],
        "crossover": ["Arytmetyczne", "Liniowe", "Mieszające-alfa", "Mieszające-alfa-beta", "Uśredniające"],
        "mutation": ["Równomierna", "Gaussa"]
    },
    
    "comparison_configs": {
        "binary": [
            {
                "name": "Binary: Konfiguracja bazowa",
                "representation": "Binary",
                "selection": "Turniejowa",
                "crossover": "Jednopunktowe",
                "mutation": "Bit Flip",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9
            },
            {
                "name": "Binary: Wysoka mutacja",
                "representation": "Binary",
                "selection": "Turniejowa",
                "crossover": "Jednopunktowe",
                "mutation": "Bit Flip",
                "mutation_rate": 0.2,
                "crossover_rate": 0.9
            },
            {
                "name": "Binary: Krzyżowanie dwupunktowe",
                "representation": "Binary",
                "selection": "Turniejowa",
                "crossover": "Dwupunktowe",
                "mutation": "Bit Flip",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9
            }
        ],
        "real": [
            {
                "name": "Real: Arytmetyczne + Równomierna",
                "representation": "Real",
                "selection": "Turniejowa",
                "crossover": "Arytmetyczne",
                "mutation": "Równomierna",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9
            },
            {
                "name": "Real: Mieszające-alfa + Gaussa",
                "representation": "Real",
                "selection": "Turniejowa",
                "crossover": "Mieszające-alfa",
                "mutation": "Gaussa",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9
            },
            {
                "name": "Real: Liniowe + Równomierna",
                "representation": "Real",
                "selection": "Turniejowa",
                "crossover": "Liniowe",
                "mutation": "Równomierna",
                "mutation_rate": 0.1,
                "crossover_rate": 0.9
            }
        ]
    }
}

def load_config():
    return DEFAULT_CONFIG