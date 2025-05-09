#pip install numpy matplotlib pandas
import numpy as np

# Sphere function to minimize
def fitness_function(x):
    return np.sum(np.square(x))

# Initialize population
def initialize_population(pop_size, dim, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, dim))

# Clone antibodies
def clone_population(population, beta):
    clones = []
    for antibody in population:
        num_clones = int(beta * len(population))
        clones.extend([antibody.copy() for _ in range(num_clones)])
    return np.array(clones)

# Hypermutation
def hypermutate(clones, fitness_values, mutation_rate=0.1):
    mutated = []
    max_fit = max(fitness_values)
    for i, clone in enumerate(clones):
        mutation_strength = mutation_rate * (1 - fitness_values[i] / (max_fit + 1e-9))
        mutation = np.random.normal(0, mutation_strength, clone.shape)
        mutated.append(clone + mutation)
    return np.array(mutated)

# Replace worst antibodies
def replace_worst(population, new_candidates, n_replace):
    combined = np.vstack((population, new_candidates))
    fitnesses = np.apply_along_axis(fitness_function, 1, combined)
    sorted_indices = np.argsort(fitnesses)
    return combined[sorted_indices[:len(population)]]

# CSA main function
def clonal_selection_algorithm(dim=5, pop_size=10, generations=50, beta=2, n_replace=2,
                               lower_bound=-5, upper_bound=5):
    population = initialize_population(pop_size, dim, lower_bound, upper_bound)

    for gen in range(generations):
        fitnesses = np.apply_along_axis(fitness_function, 1, population)
        sorted_indices = np.argsort(fitnesses)
        population = population[sorted_indices]  # sort by fitness

        clones = clone_population(population, beta)
        clone_fitnesses = np.apply_along_axis(fitness_function, 1, clones)
        mutated_clones = hypermutate(clones, clone_fitnesses)

        mutated_fitnesses = np.apply_along_axis(fitness_function, 1, mutated_clones)
        best_mutated_indices = np.argsort(mutated_fitnesses)[:pop_size]

        best_mutated = mutated_clones[best_mutated_indices]
        population = replace_worst(population, best_mutated, n_replace)

        best_fitness = fitness_function(population[0])
        print(f"Generation {gen+1}, Best Fitness: {best_fitness:.4f}")

    return population[0], fitness_function(population[0])

# Run the algorithm
best_solution, best_fitness = clonal_selection_algorithm()
print("Best solution found:", best_solution)
print("Fitness of best solution:", best_fitness)
