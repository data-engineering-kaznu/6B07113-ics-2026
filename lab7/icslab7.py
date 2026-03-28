import csv
import heapq

# --- 1. Вспомогательные функции ---

def read_maze(filename):
    maze = []
    start_pos = None
    end_pos = None
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            maze.append(row)
            for c, val in enumerate(row):
                if val == 'S':
                    start_pos = (r, c)
                elif val == 'E':
                    end_pos = (r, c)
    return maze, start_pos, end_pos

def get_neighbors(pos, maze):
    rows, cols = len(maze), len(maze[0])
    r, c = pos
    neighbors = []
    # Направления: вверх, вниз, влево, вправо
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != '1':
            neighbors.append((nr, nc))
    return neighbors

def heuristic(a, b):
    # Расстояние Манхэттена
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# --- 2. Алгоритмы поиска ---

def search(maze, start, end, mode='A*'):
    # Очередь с приоритетом: (приоритет, текущая_точка)
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited_nodes = 0

    while priority_queue:
        _, current = heapq.heappop(priority_queue)
        visited_nodes += 1

        if current == end:
            break

        for next_node in get_neighbors(current, maze):
            new_cost = cost_so_far[current] + 1
            
            if next_node not in cost_so_far or (mode == 'A*' and new_cost < cost_so_far[next_node]):
                cost_so_far[next_node] = new_cost
                
                # Выбор приоритета в зависимости от алгоритма
                if mode == 'Greedy':
                    priority = heuristic(next_node, end)
                else: # A*
                    priority = new_cost + heuristic(next_node, end)
                
                heapq.heappush(priority_queue, (priority, next_node))
                came_from[next_node] = current

    # Восстановление пути
    path = []
    curr = end
    if end in came_from:
        while curr:
            path.append(curr)
            curr = came_from[curr]
    return path[::-1], visited_nodes

# --- 3. Основной блок выполнения ---

def main():
    maze, start, end = read_maze('input7.csv')

    # Жадный поиск
    path_greedy, visits_greedy = search(maze, start, end, mode='Greedy')
    
    # A* поиск
    path_astar, visits_astar = search(maze, start, end, mode='A*')

    # Запись результатов в файл
    with open('output7.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Path Length', 'Nodes Visited', 'Path'])
        writer.writerow(['Greedy', len(path_greedy), visits_greedy, path_greedy])
        writer.writerow(['A*', len(path_astar), visits_astar, path_astar])

    # Визуализация (упрощенная в консоли)
    print(f"Результаты сохранены в output7.csv")
    print(f"Жадный поиск: посещено {visits_greedy} узлов, длина пути {len(path_greedy)}")
    print(f"A*: посещено {visits_astar} узлов, длина пути {len(path_astar)}")

if __name__ == "__main__":
    main()