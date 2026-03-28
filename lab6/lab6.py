import numpy as np
import matplotlib.pyplot as plt
import heapq

maze = np.array([
    [0,0,0,0,1,0,0],
    [1,1,0,1,1,0,1],
    [0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0]
])

start = (0, 0)
goal = (4, 6)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node):
    x, y = node
    neighbors = []
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
            if maze[nx][ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def greedy_search(start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), start))
    came_from = {}
    visited = set()
    
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            break
        visited.add(current)
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                heapq.heappush(open_list, (heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), visited

def a_star(start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            break
        visited.add(current)

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))
                came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), visited

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []
    path.append(start)
    path.reverse()
    return path

def visualize(path, visited, title):
    grid = np.copy(maze)

    for x, y in visited:
        grid[x][y] = 0.5 

    for x, y in path:
        grid[x][y] = 0.8 

    plt.figure()
    plt.imshow(grid)
    plt.title(title)
    plt.colorbar()
    plt.show()

greedy_path, greedy_visited = greedy_search(start, goal)
a_star_path, a_star_visited = a_star(start, goal)

print("Жадный путь:", greedy_path)
print("A* путь:", a_star_path)

print("Жадный посещено:", len(greedy_visited))
print("A* посещено:", len(a_star_visited))

visualize(greedy_path, greedy_visited, "Жадный поиск")
visualize(a_star_path, a_star_visited, "A* поиск")