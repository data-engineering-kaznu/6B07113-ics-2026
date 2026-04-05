import numpy as np
import matplotlib.pyplot as plt

# Функция Розенброка
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Класс для генетического алгоритма
class GeneticAlgorithm:
    def __init__(self, pop_size=100, mutation_rate=0.01, generations=100):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.bounds = (-2, 2)  # Границы для x и y

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, 2))

    def fitness(self, individual):
        x, y = individual
        return 1 / (1 + rosenbrock(x, y))  # Инвертированная для максимизации

    def select(self, population, fitnesses):
        # Турнирная селекция
        selected = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(len(population), 2, replace=False)
            if fitnesses[i] > fitnesses[j]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        # Одноточечный кроссовер
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(len(individual))
            individual[idx] += np.random.normal(0, 0.1)
            individual[idx] = np.clip(individual[idx], self.bounds[0], self.bounds[1])
        return individual

    def evolve(self):
        population = self.initialize_population()
        best_fitness_history = []
        best_individual = None
        best_fitness = -np.inf

        for gen in range(self.generations):
            fitnesses = np.array([self.fitness(ind) for ind in population])
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_individual = population[current_best_idx]

            best_fitness_history.append(best_fitness)

            selected = self.select(population, fitnesses)
            new_population = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            population = np.array(new_population[:self.pop_size])

        return best_individual, best_fitness_history

# Запуск
if __name__ == "__main__":
    ga = GeneticAlgorithm(pop_size=50, mutation_rate=0.1, generations=200)
    best_ind, history = ga.evolve()
    print(f"Best individual: {best_ind}")
    print(f"Best Rosenbrock value: {rosenbrock(*best_ind)}")

    # Визуализация
    plt.plot(history)
    plt.title("Evolution of Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()