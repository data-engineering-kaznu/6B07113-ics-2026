import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class Individual:
    genes: list[float]
    fitness: float


def rosenbrock(x: float, y: float) -> float:
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def evaluate(genes: list[float]) -> float:
    x, y = genes
    return rosenbrock(x, y)


def create_individual(bounds: list[tuple[float, float]]) -> Individual:
    genes = [random.uniform(low, high) for low, high in bounds]
    return Individual(genes=genes, fitness=evaluate(genes))


def tournament_selection(population: list[Individual], tournament_size: int = 3) -> Individual:
    participants = random.sample(population, tournament_size)
    winner = min(participants, key=lambda item: item.fitness)
    return Individual(genes=winner.genes[:], fitness=winner.fitness)


def crossover(parent_a: Individual, parent_b: Individual) -> tuple[list[float], list[float]]:
    alpha = random.random()
    child_a = [
        alpha * gene_a + (1 - alpha) * gene_b
        for gene_a, gene_b in zip(parent_a.genes, parent_b.genes)
    ]
    child_b = [
        alpha * gene_b + (1 - alpha) * gene_a
        for gene_a, gene_b in zip(parent_a.genes, parent_b.genes)
    ]
    return child_a, child_b


def mutate(genes: list[float], mutation_rate: float, bounds: list[tuple[float, float]]) -> list[float]:
    mutated = genes[:]
    for index, (low, high) in enumerate(bounds):
        if random.random() < mutation_rate:
            sigma = (high - low) * 0.1
            mutated[index] += random.gauss(0, sigma)
            mutated[index] = max(low, min(high, mutated[index]))
    return mutated


def evolve(
    population_size: int,
    mutation_rate: float,
    generations: int,
    bounds: list[tuple[float, float]],
    crossover_rate: float = 0.9,
    elite_size: int = 2,
) -> tuple[Individual, list[float]]:
    population = [create_individual(bounds) for _ in range(population_size)]
    history: list[float] = []

    for _ in range(generations):
        population.sort(key=lambda item: item.fitness)
        history.append(population[0].fitness)

        next_population = [
            Individual(genes=item.genes[:], fitness=item.fitness)
            for item in population[:elite_size]
        ]

        while len(next_population) < population_size:
            parent_a = tournament_selection(population)
            parent_b = tournament_selection(population)

            if random.random() < crossover_rate:
                child_a_genes, child_b_genes = crossover(parent_a, parent_b)
            else:
                child_a_genes = parent_a.genes[:]
                child_b_genes = parent_b.genes[:]

            child_a_genes = mutate(child_a_genes, mutation_rate, bounds)
            child_b_genes = mutate(child_b_genes, mutation_rate, bounds)

            next_population.append(
                Individual(genes=child_a_genes, fitness=evaluate(child_a_genes))
            )
            if len(next_population) < population_size:
                next_population.append(
                    Individual(genes=child_b_genes, fitness=evaluate(child_b_genes))
                )

        population = next_population

    population.sort(key=lambda item: item.fitness)
    history.append(population[0].fitness)
    return population[0], history


def plot_convergence(experiments: list[dict]) -> None:
    plt.figure(figsize=(10, 6))
    for experiment in experiments:
        label = (
            f"pop={experiment['population_size']}, "
            f"mutation={experiment['mutation_rate']}"
        )
        plt.plot(experiment["history"], label=label)

    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Genetic Algorithm Convergence for Rosenbrock Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
    plt.close()


def plot_parameter_analysis(experiments: list[dict]) -> None:
    population_sizes = sorted({item["population_size"] for item in experiments})
    mutation_rates = sorted({item["mutation_rate"] for item in experiments})
    value_map = {
        (item["population_size"], item["mutation_rate"]): item["best"].fitness
        for item in experiments
    }

    matrix = [
        [value_map[(population_size, mutation_rate)] for mutation_rate in mutation_rates]
        for population_size in population_sizes
    ]

    plt.figure(figsize=(8, 5))
    image = plt.imshow(matrix, cmap="viridis_r", aspect="auto")
    plt.colorbar(image, label="Best fitness")
    plt.xticks(range(len(mutation_rates)), [str(item) for item in mutation_rates])
    plt.yticks(range(len(population_sizes)), [str(item) for item in population_sizes])
    plt.xlabel("Mutation rate")
    plt.ylabel("Population size")
    plt.title("Parameter Influence on Optimization Quality")

    for row_index, population_size in enumerate(population_sizes):
        for col_index, mutation_rate in enumerate(mutation_rates):
            value = value_map[(population_size, mutation_rate)]
            plt.text(
                col_index,
                row_index,
                f"{value:.2e}",
                ha="center",
                va="center",
                color="white" if value > 0.01 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_analysis.png", dpi=150)
    plt.close()


def print_summary(experiments: list[dict]) -> None:
    print("Lab 7: Genetic Algorithm for Rosenbrock Optimization")
    print("-" * 72)
    print(f"{'Population':>10} | {'Mutation':>10} | {'Best x':>10} | {'Best y':>10} | {'Fitness':>12}")
    print("-" * 72)

    for experiment in experiments:
        best = experiment["best"]
        print(
            f"{experiment['population_size']:>10} | "
            f"{experiment['mutation_rate']:>10.2f} | "
            f"{best.genes[0]:>10.4f} | "
            f"{best.genes[1]:>10.4f} | "
            f"{best.fitness:>12.6e}"
        )

    best_experiment = min(experiments, key=lambda item: item["best"].fitness)
    print("-" * 72)
    print("Best configuration:")
    print(
        f"population_size={best_experiment['population_size']}, "
        f"mutation_rate={best_experiment['mutation_rate']}, "
        f"solution=({best_experiment['best'].genes[0]:.6f}, {best_experiment['best'].genes[1]:.6f}), "
        f"fitness={best_experiment['best'].fitness:.6e}"
    )
    print("Target optimum for Rosenbrock function: x=1, y=1, fitness=0")
    print(f"Plots saved to: {OUTPUT_DIR}")


def main() -> None:
    random.seed(42)
    OUTPUT_DIR.mkdir(exist_ok=True)

    bounds = [(-3.0, 3.0), (-2.0, 8.0)]
    generations = 200
    population_sizes = [20, 50, 100]
    mutation_rates = [0.01, 0.05, 0.1]

    experiments: list[dict] = []

    for population_size in population_sizes:
        for mutation_rate in mutation_rates:
            best, history = evolve(
                population_size=population_size,
                mutation_rate=mutation_rate,
                generations=generations,
                bounds=bounds,
            )
            experiments.append(
                {
                    "population_size": population_size,
                    "mutation_rate": mutation_rate,
                    "best": best,
                    "history": history,
                }
            )

    print_summary(experiments)
    plot_convergence(experiments)
    plot_parameter_analysis(experiments)


if __name__ == "__main__":
    main()
