import random
import numpy as np
import matplotlib.pyplot as plt

# Parametry algorytmu genetycznego
population_size = 100
generations = 30
mutation_rate = 0.6
tournament_size = 5

# Dane
customers = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"]
locations = np.array([(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(len(customers))])
cost_matrix = np.array([[round(np.linalg.norm(locations[i] - locations[j]), 2) for j in range(len(customers))] for i in range(len(customers))])

# Wyświetlanie danych w konsoli
print("Klienci:")
for i, customer in enumerate(customers):
    print(f"{customer}: {locations[i]}")
# macierz kosztów tylko na 1 plocie
plt.figure(figsize=(12, 6))  # Adjust the figure size

plt.subplot(1, 2, 1)
plt.imshow(cost_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(customers)), customers)
plt.yticks(range(len(customers)), customers)

# Display numbers in the cost matrix
for i in range(len(customers)):
    for j in range(len(customers)):
        plt.text(j, i, cost_matrix[i][j], ha='center', va='center', color='black')

plt.title('Cost Matrix')

# Tworzenie wykresu punktowego
plt.subplot(1, 2, 2)
plt.scatter(locations[:, 0], locations[:, 1], c='red', label='Klienci')


def generate_individual():
    route = random.sample(customers, len(customers))
    #route.append(route[0])  # Add the starting point to close the loop
    return route

def evaluate_fitness(individual):
    total_cost = 0
    for i in range(len(individual) - 1):
        total_cost += cost_matrix[customers.index(individual[i])][customers.index(individual[i + 1])]
    return total_cost

def tournament_selection(population):
    tournament = random.sample(population, tournament_size)
    best_individual = min(tournament, key=evaluate_fitness)
    return best_individual

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child

def mutate(individual):
    mutated_individual = individual.copy()
    gene1, gene2 = random.sample(range(len(individual)), 2)
    mutated_individual[gene1], mutated_individual[gene2] = mutated_individual[gene2], mutated_individual[gene1]
    return mutated_individual

def plot_route(route, generation, cost):
    route_locations = [locations[customers.index(customer)] for customer in route]
    route_coordinates = np.array(route_locations).T
    plt.plot(route_coordinates[0], route_coordinates[1], marker='o', linestyle='-', color='b')
    plt.scatter(route_coordinates[0], route_coordinates[1], color='r')
    #plt.plot([route_coordinates[0][-1], route_coordinates[0][0]], [route_coordinates[1][-1], route_coordinates[1][0]], linestyle='-', color='b')  # Closing the loop
    for i, customer in enumerate(route):
        plt.text(route_locations[i][0], route_locations[i][1], customer, fontsize=8, ha='right', va='bottom')
    plt.title(f'Optymalna Trasa Dostaw - Generacja {generation} - Koszt: {cost}')
    plt.xlabel('Szerokość geograficzna')
    plt.ylabel('Długość geograficzna')
    plt.show()

def genetic_algorithm():
    population = [generate_individual() for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        new_population = [tournament_selection(population) for _ in range(population_size)]

        for i in range(0, population_size, 2):
            parent1, parent2 = new_population[i], new_population[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population[i], new_population[i + 1] = child1, child2

        for i in range(population_size):
            if random.random() < mutation_rate:
                new_population[i] = mutate(new_population[i])

        population = new_population

        best_solution = min(population, key=evaluate_fitness)
        best_fitness = evaluate_fitness(best_solution)
        
        plot_route(best_solution, generation, best_fitness)

    return best_solution, best_fitness

if __name__ == "__main__":
    best_route, best_cost = genetic_algorithm()
    print("Najlepsza trasa:", best_route)
    print("Koszt najkrotszej trasy:", best_cost)
