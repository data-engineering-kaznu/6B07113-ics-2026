# Lab 7 Short Explanation

## What This Lab Does

This lab solves an optimization problem with a genetic algorithm.

The function we try to minimize is the Rosenbrock function:

```text
f(x, y) = (1 - x)^2 + 100(y - x^2)^2
```

Its best value is:

```text
f(1, 1) = 0
```

So the algorithm tries to find values of `x` and `y` as close as possible to `(1, 1)`.

## What The Program Does

The file `lab7.py`:

1. Creates a population of random candidate solutions.
2. Evaluates how good each solution is.
3. Selects better solutions more often.
4. Combines them with crossover.
5. Randomly changes some values with mutation.
6. Repeats this for many generations.
7. Compares different parameter settings.
8. Saves plots showing the results.

## Main Terms

- `Genetic algorithm` - an optimization method inspired by natural selection.
- `Population` - a group of candidate solutions.
- `Individual` - one candidate solution.
- `Gene` - one value inside a solution. Here the genes are `x` and `y`.
- `Chromosome` - the full set of genes of one individual. Here it is `[x, y]`.
- `Fitness` - a score showing how good the solution is.
- `Selection` - choosing better individuals for the next step.
- `Crossover` - mixing two parent solutions to create children.
- `Mutation` - random small changes in genes.
- `Generation` - one full iteration of the algorithm.

## Parameters In This Lab

- `population size` - how many solutions exist in one generation
- `mutation rate` - how often random changes happen
- `generations` - how many times the algorithm improves the population

## How To Understand The Result

Smaller fitness is better.

If the fitness gets closer to `0`, the algorithm is finding a better solution.

In this run, the best result was close to:

```text
x = 0.979270
y = 0.958813
fitness = 0.000432
```

This is good because it is close to the ideal point `(1, 1)`.

## Output Files

- `outputs/convergence.png` - shows how fitness improves over generations
- `outputs/parameter_analysis.png` - shows how mutation rate and population size affect the result
