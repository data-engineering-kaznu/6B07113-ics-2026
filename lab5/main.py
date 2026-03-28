import heapq

# Лабиринт: 0 = проход, 1 = стена
# S = старт (0,0), E = финиш (4,4)
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

START = (0, 0)
END   = (4, 4)


# Манхэттенское расстояние до финиша
def h(r, c):
    return abs(r - END[0]) + abs(c - END[1])


# Соседи клетки (вверх, вниз, влево, вправо)
def neighbors(r, c):
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 5 and 0 <= nc < 5 and maze[nr][nc] == 0:
            result.append((nr, nc))
    return result


# Восстановить путь от финиша к старту
def get_path(came_from):
    path = []
    node = END
    while node != START:
        path.append(node)
        node = came_from[node]
    path.append(START)
    path.reverse()
    return path


# ── ЖАДНЫЙ ПОИСК ──────────────────────────
def greedy(maze):
    visited    = set()
    came_from  = {}
    queue      = [(h(*START), START)]  # приоритет = h(n)

    while queue:
        _, (r, c) = heapq.heappop(queue)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if (r, c) == END:
            break

        for nr, nc in neighbors(r, c):
            if (nr, nc) not in visited:
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(queue, (h(nr, nc), (nr, nc)))

    return get_path(came_from), len(visited)


# ── A* ────────────────────────────────────
def astar(maze):
    visited    = set()
    came_from  = {}
    g          = {START: 0}        # сколько шагов прошли
    queue      = [(h(*START), START)]  # приоритет = g + h

    while queue:
        _, (r, c) = heapq.heappop(queue)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if (r, c) == END:
            break

        for nr, nc in neighbors(r, c):
            new_g = g[(r, c)] + 1
            if new_g < g.get((nr, nc), 999):
                g[(nr, nc)]         = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(queue, (new_g + h(nr, nc), (nr, nc)))

    return get_path(came_from), len(visited)


# ── ВЫВОД ─────────────────────────────────
def show(path, title):
    print(title)
    path_set = set(path)
    for r in range(5):
        row = ""
        for c in range(5):
            if   (r,c) == START:       row += "S "
            elif (r,c) == END:         row += "E "
            elif (r,c) in path_set:    row += "* "
            elif maze[r][c] == 1:      row += "# "
            else:                      row += ". "
        print(row)
    print(f"Длина пути: {len(path)}, проверено клеток: {visited}\n")


# ── ЗАПУСК ────────────────────────────────
path1, visited = greedy(maze)
show(path1, "=== Жадный ===")

path2, visited = astar(maze)
show(path2, "=== A* ===")
