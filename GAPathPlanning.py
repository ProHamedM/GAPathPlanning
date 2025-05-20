import random
import matplotlib.pyplot as plt
import numpy as np

# The lawn is represented as a 5x5 grid.
YARD_SIZE = 5
NUM_GENES = YARD_SIZE * YARD_SIZE  # Each gene represents a cell in the yard (0 or 1 for mowed/unmowed)

def create_yard(size):
    # Creates an initial yard (all unmowed)
    return np.zeros((size, size), dtype=int)

def calculate_fitness(chromosome, yard_size=YARD_SIZE):
    # Reshape the chromosome into a 2D array representing the yard
    yard_representation = np.array(chromosome).reshape(yard_size, yard_size)
    # Count the number of mowed cells (cells with a value of 1)
    fitness = np.sum(yard_representation)
    return fitness

# --- 2. Genetic Algorithm Functions ---
def create_individual(num_genes):
    # Creates a random individual (chromosome) with binary genes
    return [random.choice([0, 1]) for _ in range(num_genes)]

def create_population(population_size, num_genes):
    # Creates a population of individuals
    return [create_individual(num_genes) for _ in range(population_size)]

def selection(population, fitnesses):
    # Selects the top individuals for mating using tournament selection.
    tournament_size = 3  # Size of the tournament
    selected_parents = []
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        # Select the individual with the highest fitness in the tournament
        winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        selected_parents.append(population[winner_index])
    return selected_parents

def crossover(parent1, parent2):
    # Performs single-point crossover to create two offspring
    if random.random() < 0.7:  # Crossover probability
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2
    else:
        return parent1, parent2  # No crossover occurs

def mutate(individual, mutation_rate=0.01):
    # Mutates an individual by flipping bits with a given probability
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            mutated_individual.append(1 - gene)  # Flip the bit (0 to 1 or 1 to 0)
        else:
            mutated_individual.append(gene)
    return mutated_individual

def genetic_algorithm(population_size, num_generations, yard_size=YARD_SIZE):
    # Implements the genetic algorithm to find a solution for mowing the lawn
    population = create_population(population_size, yard_size * yard_size)
    fitness_history = []
    best_fitness_history = [] # Keep track of the best fitness in each generation

    for generation in range(num_generations):
        fitnesses = [calculate_fitness(individual, yard_size) for individual in population]
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness) #store the best fitness
        fitness_history.append(fitnesses)
        selected_parents = selection(population, fitnesses)
        next_population = []
        for i in range(0, len(selected_parents) - 1, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            next_population.extend([offspring1, offspring2])
        population = next_population

    best_solution = population[fitnesses.index(max(fitnesses))]
    best_fitness = max(fitnesses)
    return best_solution, fitness_history, best_fitness_history

# --- 3. Visualization ---
def visualize_yard(solution, yard_size=YARD_SIZE):
    # Visualizes the yard and the lawnmower's path based on the solution
    yard = np.array(solution).reshape(yard_size, yard_size)
    plt.figure(figsize=(6, 6))
    plt.imshow(yard, cmap='Greens', interpolation='nearest')  # Use 'Greens' for mowed/unmowed
    plt.title('Lawnmower Path and Coverage')
    plt.xticks(np.arange(0, yard_size, 1))
    plt.yticks(np.arange(0, yard_size, 1))
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Add grid lines
    plt.grid(color='black', linestyle='-', linewidth=1)

    # Annotate the plot to show 0 and 1
    for i in range(yard_size):
        for j in range(yard_size):
            text = "Mowed" if yard[i, j] == 1 else "Unmowed"
            plt.text(j, i, text, ha="center", va="center", color="black")

    plt.show()

def visualize_fitness_history(fitness_history, best_fitness_history):
    # Visualizes the fitness history of the population
    plt.figure(figsize=(8, 6))
    plt.plot(np.mean(fitness_history, axis=1), label='Mean Fitness')
    plt.plot(best_fitness_history, label = "Best Fitness") # Plot the best fitness
    plt.title('Fitness History')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 4. Run the Simulation ---
if __name__ == "__main__":
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 50
    YARD_SIZE = 5

    best_solution, fitness_history, best_fitness_history = genetic_algorithm(POPULATION_SIZE, NUM_GENERATIONS, YARD_SIZE)
    print("Best Solution (Lawnmower Path - 1 for mowed, 0 for unmowed):", best_solution)
    print("Best Fitness:", calculate_fitness(best_solution, YARD_SIZE))

    visualize_yard(best_solution, YARD_SIZE)
    visualize_fitness_history(fitness_history, best_fitness_history)
