from PIL import Image
import numpy as np
import heapq
import matplotlib.pyplot as plt
import time

img = Image.open("112.png").convert("L")

img = img.resize((60, 60))

maze = np.array(img)

# Белый = проход (1)
# Черный = стена (0)
maze = (maze > 128).astype(int)

print("Размер лабиринта:", maze.shape)

plt.figure(figsize=(6, 6))
plt.imshow(maze, cmap="gray")
plt.title("Лабиринт")
plt.show()


# 2. Автоматический поиск старта и финиша
def find_open_point_from_top_left():
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 1:
                return (i, j)

def find_open_point_from_bottom_right():
    for i in range(maze.shape[0] - 1, -1, -1):
        for j in range(maze.shape[1] - 1, -1, -1):
            if maze[i][j] == 1:
                return (i, j)

start = find_open_point_from_top_left()
goal = find_open_point_from_bottom_right()

print("Старт:", start)
print("Финиш:", goal)


# 3. Эвристика Манхэттена
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# 4. Соседи
def get_neighbors(node):
    x, y = node
    neighbors = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            if maze[nx][ny] == 1:
                neighbors.append((nx, ny))

    return neighbors


# 5. Восстановление пути
def reconstruct_path(came_from, start, goal):
    path = []
    current = goal

    while current in came_from:
        path.append(current)
        current = came_from[current]

    if current != start:
        return []

    path.append(start)
    path.reverse()
    return path

# 6. Greedy Search
def greedy_search(start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), start))

    came_from = {}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            return reconstruct_path(came_from, start, goal), visited

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                heapq.heappush(
                    open_list,
                    (heuristic(neighbor, goal), neighbor)
                )

                if neighbor not in came_from:
                    came_from[neighbor] = current

    return [], visited


# 7. A* Search

def a_star(start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            return reconstruct_path(came_from, start, goal), visited

        if current in visited:
            continue

        visited.add(current)

        for neighbor in get_neighbors(current):
            temp_g = g_score[current] + 1

            if neighbor not in g_score or temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                f_score = temp_g + heuristic(neighbor, goal)

                heapq.heappush(open_list, (f_score, neighbor))
                came_from[neighbor] = current

    return [], visited

# 8. Визуализация пути

def visualize_path(path, title):
    display_maze = np.copy(maze).astype(float)

    # путь
    for x, y in path:
        display_maze[x][y] = 0.5

    # старт
    sx, sy = start
    display_maze[sx][sy] = 0.8

    # финиш
    gx, gy = goal
    display_maze[gx][gy] = 0.2

    plt.figure(figsize=(6, 6))
    plt.imshow(display_maze, cmap="gray")
    plt.title(title)
    plt.show()


start_time = time.time()
path_greedy, visited_greedy = greedy_search(start, goal)
greedy_time = time.time() - start_time

start_time = time.time()
path_astar, visited_astar = a_star(start, goal)
astar_time = time.time() - start_time


print("\n--- Результаты ---")
print("Greedy длина пути:", len(path_greedy))
print("Greedy время:", round(greedy_time, 5), "сек")

print("A* длина пути:", len(path_astar))
print("A* время:", round(astar_time, 5), "сек")

if len(path_greedy) > 0:
    visualize_path(path_greedy, "Greedy Search")
else:
    print("Greedy не нашел путь")

if len(path_astar) > 0:
    visualize_path(path_astar, "A* Search")
else:
    print("A* не нашел путь")