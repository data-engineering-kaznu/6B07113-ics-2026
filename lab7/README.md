# Lab 7

## Topic

Genetic algorithms for optimization.

## Goal

Study the basic stages of a genetic algorithm and apply it to optimization of the Rosenbrock function.

## Practical Tasks

- implement a genetic algorithm for Rosenbrock function optimization
- tune algorithm parameters
- analyze how mutation rate and population size affect the result

## Files

- `lab7.py` - main Python implementation
- `requirements.txt` - dependencies
- `outputs/` - generated plots after execution

## Run

```bash
cd lab7
pip install -r requirements.txt
python3 lab7.py
```

## Notes

The script:

- uses tournament selection
- uses arithmetic crossover
- uses Gaussian mutation
- compares several population sizes and mutation probabilities
- saves convergence and parameter analysis plots to `outputs/`
