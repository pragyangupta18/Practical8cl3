import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from deap import base, creator, tools, algorithms
import random

# Neural Network for modeling spray drying process
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer (e.g., yield or moisture content)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Example: generate random training data (replace with real data)
X_train = np.random.rand(100, 3)  # 100 samples, 3 parameters (e.g., temperature, pressure, etc.)
y_train = np.random.rand(100, 1)  # Corresponding outputs (e.g., moisture content or yield)

# Train the NN model
nn_model = create_nn_model(X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=50, batch_size=10)

# Define the problem as a maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the evaluation function (this will use the neural network model)
def evaluate(individual):
    # For simplicity, we use the individual as input parameters to the NN model
    # Convert the individual to the shape of the NN's input
    input_data = np.array(individual).reshape(1, -1)
    prediction = nn_model.predict(input_data)
    return prediction[0][0],  # Return a tuple

# Define the genetic algorithm
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)  # Random float between 0 and 1
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float,) * 3, n=1)  # 3 parameters
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Create the population
population = toolbox.population(n=100)

# Run the genetic algorithm
result = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)

# Get the best individual
best_individual = tools.selBest(population, 1)[0]
print(f"Best individual: {best_individual}")
