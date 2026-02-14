import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.cos(x - 0.5) - x**2

def df(x):
    return -math.sin(x - 0.5) - 2*x

def newton_method(x0, eps):
    x_curr = x0
    print(f"{'n':<5} {'x_n':<12} {'f(x_n)':<12} {'|x_n-x_prev|':<12}")
    print("-" * 50)
    
    for n in range(1, 11):
        x_next = x_curr - f(x_curr) / df(x_curr)
        diff = abs(x_next - x_curr)
        print(f"{n:<5} {x_next:<12.6f} {f(x_next):<12.6f} {diff:<12.6f}")
        
        if diff < eps:
            return x_next
        x_curr = x_next
    return x_curr

# Данные задачи
x_start = 0.8  # Выбрано из окрестности корня
epsilon = 0.001

# Расчет
root = newton_method(x_start, epsilon)

