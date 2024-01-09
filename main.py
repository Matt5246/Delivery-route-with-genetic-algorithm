import random
import numpy as np
import pygad
import folium

# Założenia
clients = {
    "K1": (52.237, 21.017),
    "K2": (52.247, 21.027),
    "K3": (52.257, 21.037),
    "K4": (52.267, 21.047),
    "K5": (52.277, 21.057),
    "K6": (52.287, 21.067),
    "K7": (52.297, 21.077),
    "K8": (52.307, 21.087),
    "K9": (52.317, 21.097),
    "K10": (52.327, 21.107),
}

# Convert clients to numerical indices
clients_indices = {k: i for i, k in enumerate(clients.keys())}
clients_coordinates = list(clients.values())

cost_matrix = np.array([
    [0, 5, 8, 3, 2, 7, 6, 4, 9, 1],
    [5, 0, 6, 4, 8, 3, 2, 9, 1, 7],
    [8, 6, 0, 2, 5, 9, 1, 7, 4, 3],
    [3, 4, 2, 0, 6, 8, 5, 7, 9, 1],
    [2, 8, 5, 6, 0, 3, 7, 9, 1, 4],
    [7, 3, 9, 8, 3, 0, 6, 4, 2, 1],
    [6, 2, 1, 5, 7, 6, 0, 8, 3, 9],
    [4, 9, 7, 7, 9, 4, 8, 0, 5, 2],
    [9, 1, 4, 9, 1, 2, 3, 5, 0, 8],
    [1, 7, 3, 1, 4, 1, 9, 2, 8, 0]
])

# 1. Zdefiniuj kodowanie (permutacyjne)
def generate_individual():
    clients_list = list(clients.keys())
    random.shuffle(clients_list)
    return [clients_indices[client] for client in clients_list]

# 2. Inicjalizacja populacji
population_size = 100
population = [generate_individual() for _ in range(population_size)]

# 3. Ocena funkcji celu
def fitness_function(solution, solution_index, ga_instance):
    total_cost = 0
    for i in range(len(solution) - 1):
        total_cost += cost_matrix[solution[i], solution[i + 1]]
    return 1 / total_cost  # Cel: minimalizacja kosztu, więc maksymalizujemy odwrotność kosztu

# Przykład użycia pyGAD
ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=50,
    sol_per_pop=population_size,
    num_genes=len(clients),
    fitness_func=fitness_function,
    initial_population=population,
)

ga_instance.run()

# 4. Operatory genetyczne, 5. Proces ewolucji, 6. Warunki zatrzymania, 7. Prezentacja wyników
for generation in range(ga_instance.num_generations):
    # Wybór rodziców
    parents = ga_instance.select_parents()
    
    # Krzyżowanie rodziców
    crossovered_population = ga_instance.crossover(parents)

    # Mutacja potomstwa
    mutated_population = ga_instance.mutation(crossovered_population)

    # Ocena potomstwa
    fitness_values = np.array([ga_instance.fitness_func(solution, idx, ga_instance) for idx, solution in enumerate(mutated_population)])

    # Zastąpienie starej populacji potomstwem
    ga_instance.update_population(mutated_population, fitness_values)

    # Wydruk informacji o postępie
    print("Generation:", generation + 1, "Best Fitness:", ga_instance.best_solution()[1])

    # 6. Warunki zatrzymania
    if ga_instance.best_solution()[1] == 1.0:
        print("Solution found in generation", generation + 1)
        break

# 7. Prezentacja wyników
best_solution, best_solution_fitness = ga_instance.best_solution()
print("Best Solution:", best_solution)
print("Best Fitness:", best_solution_fitness)

# 8. Optymalizacja i eksperymenty
# ... (Dodaj kod do przeprowadzania eksperymentów)

# 9. Dokumentacja
# ... (Dodaj kod do przygotowania dokumentacji)

# 8. Optymalizacja i eksperymenty
# ... (to jest miejsce na dodanie kodu do przeprowadzania eksperymentów)

# 9. Dokumentacja
# ... (to jest miejsce na dodanie kodu do przygotowania dokumentacji)
