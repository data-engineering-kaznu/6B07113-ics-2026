import heapq
import time
import matplotlib.pyplot as plt
import numpy as np

maze = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
]

start = (0, 0)
goal = (9, 9)


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(node, maze):
    neighbors = []
    rows = len(maze)
    cols = len(maze[0])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx = node[0] + dx
        ny = node[1] + dy

        if 0 <= nx < rows and 0 <= ny < cols:
            if maze[nx][ny] == 0:
                neighbors.append((nx, ny))

    return neighbors


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def greedy_search(maze, start, goal):
    start_time = time.perf_counter()

    pq = []
    heapq.heappush(pq, (heuristic(start, goal), start))

    visited = set()
    came_from = {}
    visit_order = []

    while pq:
        h, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        visit_order.append(current)

        if current == goal:
            end_time = time.perf_counter()
            path = reconstruct_path(came_from, current)
            return {
                "path": path,
                "visited": visited,
                "visit_order": visit_order,
                "time_ms": (end_time - start_time) * 1000
            }

        for neighbor in get_neighbors(current, maze):
            if neighbor not in visited:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))

    end_time = time.perf_counter()
    return {
        "path": [],
        "visited": visited,
        "visit_order": visit_order,
        "time_ms": (end_time - start_time) * 1000
    }


def a_star_search(maze, start, goal):
    start_time = time.perf_counter()

    pq = []
    heapq.heappush(pq, (heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    visited = set()
    visit_order = []

    while pq:
        f, g, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        visit_order.append(current)

        if current == goal:
            end_time = time.perf_counter()
            path = reconstruct_path(came_from, current)
            return {
                "path": path,
                "visited": visited,
                "visit_order": visit_order,
                "time_ms": (end_time - start_time) * 1000
            }

        for neighbor in get_neighbors(current, maze):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_score, tentative_g, neighbor))

    end_time = time.perf_counter()
    return {
        "path": [],
        "visited": visited,
        "visit_order": visit_order,
        "time_ms": (end_time - start_time) * 1000
    }


def create_visual_grid(maze, visit_order, path, start, goal):
    rows = len(maze)
    cols = len(maze[0])
    grid = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                grid[i][j] = -1

    for step, cell in enumerate(visit_order, start=1):
        x, y = cell
        grid[x][y] = step

    for x, y in path:
        grid[x][y] = len(visit_order) + 20

    sx, sy = start
    gx, gy = goal
    grid[sx][sy] = len(visit_order) + 40
    grid[gx][gy] = len(visit_order) + 60

    return grid


def draw_result(ax, maze, result, title, start, goal):
    grid = create_visual_grid(
        maze,
        result["visit_order"],
        result["path"],
        start,
        goal
    )

    ax.imshow(grid, cmap="viridis")
    ax.set_title(title, fontsize=12)

    rows = len(maze)
    cols = len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                ax.text(j, i, "X", ha="center", va="center", color="white", fontsize=10)
            elif (i, j) == start:
                ax.text(j, i, "S", ha="center", va="center", color="white", fontsize=10)
            elif (i, j) == goal:
                ax.text(j, i, "G", ha="center", va="center", color="white", fontsize=10)
            elif (i, j) in result["path"]:
                ax.text(j, i, "*", ha="center", va="center", color="white", fontsize=10)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)


def print_comparison(greedy_result, a_star_result):
    print("СРАВНЕНИЕ АЛГОРИТМОВ")
    print("-" * 40)

    print("Жадный поиск:")
    print("Длина пути:", len(greedy_result["path"]))
    print("Посещено вершин:", len(greedy_result["visited"]))
    print("Время выполнения (мс):", round(greedy_result["time_ms"], 3))
    print()

    print("A*:")
    print("Длина пути:", len(a_star_result["path"]))
    print("Посещено вершин:", len(a_star_result["visited"]))
    print("Время выполнения (мс):", round(a_star_result["time_ms"], 3))
    print()

    if len(a_star_result["path"]) > 0 and len(greedy_result["path"]) > 0:
        if len(a_star_result["path"]) < len(greedy_result["path"]):
            print("Вывод: A* нашел более короткий путь.")
        elif len(a_star_result["path"]) > len(greedy_result["path"]):
            print("Вывод: Жадный поиск нашел более короткий путь.")
        else:
            print("Вывод: Оба алгоритма нашли путь одинаковой длины.")

    if len(a_star_result["visited"]) < len(greedy_result["visited"]):
        print("A* посетил меньше вершин.")
    elif len(a_star_result["visited"]) > len(greedy_result["visited"]):
        print("Жадный поиск посетил меньше вершин.")
    else:
        print("Оба алгоритма посетили одинаковое число вершин.")


def main():
    greedy_result = greedy_search(maze, start, goal)
    a_star_result = a_star_search(maze, start, goal)

    print_comparison(greedy_result, a_star_result)

    plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    draw_result(ax1, maze, greedy_result, "Жадный поиск", start, goal)

    ax2 = plt.subplot(1, 2, 2)
    draw_result(ax2, maze, a_star_result, "A* поиск", start, goal)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()