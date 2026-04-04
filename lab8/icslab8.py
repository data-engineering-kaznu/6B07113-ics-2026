import csv
import random
from pathlib import Path


INPUT_FILE = Path("input8.csv")
OUTPUT_FILE = Path("output8.csv")
SEARCH_RANGE = (-3.0, 3.0)
TOURNAMENT_SIZE = 3
DEFAULT_SEED = 42


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def load_experiments(filename):
    experiments = []
    with filename.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            experiments.append(
                {
                    "run_id": row["run_id"],
                    "population_size": int(row["population_size"]),
                    "mutation_rate": float(row["mutation_rate"]),
                    "generations": int(row["generations"]),
                    "crossover_rate": float(row["crossover_rate"]),
                    "seed": int(row["seed"]) if row.get("seed") else DEFAULT_SEED,
                }
            )
    return experiments


def create_individual():
    return [
        random.uniform(*SEARCH_RANGE),
        random.uniform(*SEARCH_RANGE),
    ]


def fitness(individual):
    return 1.0 / (1.0 + rosenbrock(individual[0], individual[1]))


def tournament_selection(population):
    contenders = random.sample(population, TOURNAMENT_SIZE)
    return max(contenders, key=fitness)


def crossover(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    alpha = random.random()
    child1 = [
        alpha * parent1[0] + (1 - alpha) * parent2[0],
        alpha * parent1[1] + (1 - alpha) * parent2[1],
    ]
    child2 = [
        alpha * parent2[0] + (1 - alpha) * parent1[0],
        alpha * parent2[1] + (1 - alpha) * parent1[1],
    ]
    return child1, child2


def mutate(individual, mutation_rate):
    for index in range(len(individual)):
        if random.random() < mutation_rate:
            individual[index] += random.gauss(0, 0.2)
            individual[index] = clamp(individual[index], *SEARCH_RANGE)
    return individual


def run_genetic_algorithm(population_size, mutation_rate, generations, crossover_rate):
    population = [create_individual() for _ in range(population_size)]
    best_individual = min(population, key=lambda item: rosenbrock(item[0], item[1]))

    for _ in range(generations):
        new_population = []
        elite = min(population, key=lambda item: rosenbrock(item[0], item[1]))
        new_population.append(elite[:])

        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))

        population = new_population
        generation_best = min(population, key=lambda item: rosenbrock(item[0], item[1]))
        if rosenbrock(generation_best[0], generation_best[1]) < rosenbrock(
            best_individual[0], best_individual[1]
        ):
            best_individual = generation_best[:]

    best_value = rosenbrock(best_individual[0], best_individual[1])
    distance_to_optimum = ((best_individual[0] - 1) ** 2 + (best_individual[1] - 1) ** 2) ** 0.5
    return best_individual, best_value, distance_to_optimum


def build_comment(best_value, mutation_rate, population_size):
    notes = []

    if mutation_rate < 0.05:
        notes.append("низкая мутация: слабое исследование пространства")
    elif mutation_rate > 0.2:
        notes.append("высокая мутация: больше случайности")
    else:
        notes.append("сбалансированная мутация")

    if population_size < 40:
        notes.append("маленькая популяция")
    elif population_size > 80:
        notes.append("большая популяция")
    else:
        notes.append("средняя популяция")

    if best_value < 0.01:
        notes.append("решение близко к глобальному минимуму")
    elif best_value < 1:
        notes.append("найдено хорошее приближение")
    else:
        notes.append("качество решения можно улучшить")

    return "; ".join(notes)


def save_results(filename, results):
    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "run_id",
                "population_size",
                "mutation_rate",
                "generations",
                "crossover_rate",
                "best_x",
                "best_y",
                "best_value",
                "distance_to_optimum",
                "comment",
            ]
        )
        writer.writerows(results)


def main():
    experiments = load_experiments(INPUT_FILE)
    results = []

    for experiment in experiments:
        random.seed(experiment["seed"])
        best_individual, best_value, distance_to_optimum = run_genetic_algorithm(
            population_size=experiment["population_size"],
            mutation_rate=experiment["mutation_rate"],
            generations=experiment["generations"],
            crossover_rate=experiment["crossover_rate"],
        )

        results.append(
            [
                experiment["run_id"],
                experiment["population_size"],
                experiment["mutation_rate"],
                experiment["generations"],
                experiment["crossover_rate"],
                round(best_individual[0], 6),
                round(best_individual[1], 6),
                round(best_value, 8),
                round(distance_to_optimum, 8),
                build_comment(
                    best_value,
                    experiment["mutation_rate"],
                    experiment["population_size"],
                ),
            ]
        )

    save_results(OUTPUT_FILE, results)
    print("Результаты сохранены в output8.csv")


if __name__ == "__main__":
    main()
