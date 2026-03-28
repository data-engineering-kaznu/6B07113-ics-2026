import heapq


# 1. Определение эвристики (Манхэттенское расстояние)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# 2. Универсальный алгоритм поиска
def heuristic_search(maze, start, goal, algorithm="A*"):
    rows, cols = len(maze), len(maze[0])
    frontier = []
    # Храним: (приоритет, текущая точка)
    heapq.heappush(frontier, (0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_nodes = 0  # Для сравнения производительности

    while frontier:
        current = heapq.heappop(frontier)[1]
        visited_nodes += 1

        if current == goal:
            break

        # Проверяем 4 направления (вверх, вниз, влево, вправо)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_node = (current[0] + dx, current[1] + dy)

            if 0 <= next_node[0] < rows and 0 <= next_node[1] < cols:
                if maze[next_node[0]][next_node[1]] == 1:  # Стена
                    continue

                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost

                    # Различие алгоритмов: [cite: 9, 10]
                    if algorithm == "Greedy":
                        priority = heuristic(goal, next_node)  # Только h(n)
                    else:  # A*
                        priority = new_cost + heuristic(goal, next_node)  # g(n) + h(n)

                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

    return reconstruct_path(came_from, start, goal), visited_nodes


def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        if current not in came_from: return None
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


# --- Практическая часть: Тестирование --- [cite: 13]

# 0 - проход, 1 - стена
maze = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]
start, goal = (0, 0), (4, 4)

for alg in ["Greedy", "A*"]:
    path, visited = heuristic_search(maze, start, goal, alg)
    print(f"--- Алгоритм: {alg} ---")
    print(f"Длина пути: {len(path) if path else 'Не найден'}")
    print(f"Посещено узлов: {visited}")
    print(f"Путь: {path}\n")