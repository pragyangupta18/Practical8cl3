import numpy as np
import random
import math
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, n_cities, distances, n_ants=10, n_iterations=100, decay=0.95, alpha=1, beta=2):
        self.n_cities = n_cities
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # Initialize pheromone levels to a small constant
        self.pheromone = np.ones((n_cities, n_cities)) * 0.1
        
    def distance(self, city1, city2):
        return self.distances[city1, city2]
    
    def total_distance(self, path):
        dist = 0
        for i in range(len(path) - 1):
            dist += self.distance(path[i], path[i + 1])
        dist += self.distance(path[-1], path[0])  # Returning to the start city
        return dist
    
    def choose_next_city(self, current_city, visited_cities):
        pheromone = np.copy(self.pheromone[current_city])
        
        # Convert visited_cities set to a list of indices
        visited_cities = list(visited_cities)
        
        # Set pheromone values for visited cities to 0
        for city in visited_cities:
            pheromone[city] = 0  # Can't go to visited cities
        
        # Calculate desirability (heuristic) = 1 / distance
        heuristic = 1 / (self.distances[current_city] + 1e-10)
        
        # Calculate probabilities of visiting each city
        pheromone_power = pheromone ** self.alpha
        heuristic_power = heuristic ** self.beta
        probabilities = pheromone_power * heuristic_power
        probabilities = probabilities / probabilities.sum()
        
        # Choose next city based on the probabilities
        next_city = np.random.choice(range(self.n_cities), p=probabilities)
        return next_city
    
    def run(self):
        shortest_path = None
        shortest_distance = float('inf')
        
        # Iterate through iterations
        for iteration in range(self.n_iterations):
            all_paths = []
            all_distances = []
            
            # Simulate ants' tours
            for ant in range(self.n_ants):
                path = [random.randint(0, self.n_cities - 1)]  # Start from a random city
                visited_cities = set(path)
                
                for _ in range(self.n_cities - 1):
                    current_city = path[-1]
                    next_city = self.choose_next_city(current_city, visited_cities)
                    path.append(next_city)
                    visited_cities.add(next_city)
                
                # Add the distance of the full tour (return to the start city)
                tour_distance = self.total_distance(path)
                all_paths.append(path)
                all_distances.append(tour_distance)
                
                # Update the shortest path found
                if tour_distance < shortest_distance:
                    shortest_distance = tour_distance
                    shortest_path = path
            
            # Update pheromones
            self.pheromone = self.pheromone * self.decay  # Evaporation
            for ant in range(self.n_ants):
                for i in range(len(all_paths[ant]) - 1):
                    city1 = all_paths[ant][i]
                    city2 = all_paths[ant][i + 1]
                    self.pheromone[city1, city2] += 1 / all_distances[ant]  # Add pheromone based on the tour length
                    self.pheromone[city2, city1] += 1 / all_distances[ant]  # Symmetric
                
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Shortest Distance: {shortest_distance}")
        
        return shortest_path, shortest_distance

# Example: TSP with 5 cities
n_cities = 5
np.random.seed(42)

# Randomly generate a distance matrix
cities = np.random.rand(n_cities, 2) * 100  # Random cities in 2D space
distances = np.zeros((n_cities, n_cities))

# Calculate distances between each pair of cities
for i in range(n_cities):
    for j in range(n_cities):
        distances[i, j] = np.linalg.norm(cities[i] - cities[j])

# Run the Ant Colony Optimization
aco = AntColony(n_cities, distances, n_ants=10, n_iterations=100)
shortest_path, shortest_distance = aco.run()

# Output the result
print("\nShortest Path:", shortest_path)
print("Shortest Distance:", shortest_distance)

# Visualize the result
x = cities[:, 0]
y = cities[:, 1]

# Plot cities
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red')

# Plot the path
for i in range(len(shortest_path) - 1):
    city1 = shortest_path[i]
    city2 = shortest_path[i + 1]
    plt.plot([cities[city1, 0], cities[city2, 0]], [cities[city1, 1], cities[city2, 1]], color='blue')

# Complete the loop
plt.plot([cities[shortest_path[-1], 0], cities[shortest_path[0], 0]], 
         [cities[shortest_path[-1], 1], cities[shortest_path[0], 1]], color='blue')

plt.title("Ant Colony Optimization - TSP")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
