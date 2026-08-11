# 模块说明：提供网格地图上的路径搜索算法。
import numpy as np
import heapq

class PathFinding:
    def __init__(self, walkable, grid_size=(64, 64)):
        # 变量说明（walkable）：寻路算法中标记各网格位置能否通行的布尔数组。
        # 变量说明（grid_size）：语义地图或寻路网格的行列尺寸。
        # 变量说明（self.cols）：网格布局需要划分的列数。
        # 变量说明（self.rows）：`__init__` 中待批量处理的行数集合。
        self.rows, self.cols = grid_size
        # 变量说明（self.walkable）：寻路算法中标记各网格位置能否通行的布尔数组。
        self.walkable = walkable
        # 变量说明（self.directions）：`__init__` 中待批量处理的方向集合。
        self.directions = {(-1, 0): 3, (1, 0): 4, (0, -1): 1, (0, 1): 2}
    
    def is_valid(self, node):
        # 变量说明（node）：当前遍历或处理的知识图谱节点标识。
        # 变量说明（c）：当前循环正在处理的颜色通道、列或计数值。
        # 变量说明（r）：当前循环正在处理的行索引、半径或红色通道值。
        r, c = node
        return 0 <= r < self.rows and \
                0 <= c < self.cols and \
                self.grid[node] in self.curr_walkable
    
    def generate_path(self, parent, node):
        # 变量说明（parent）：当前路径、节点或对象的父级对象。
        # 变量说明（node）：当前遍历或处理的知识图谱节点标识。
        # 变量说明（path）：当前读取或写入对象对应的文件系统路径。
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path
    
    def find(self, grid, start, target_item):
        # 变量说明（grid）：按世界坐标存储材质或寻路状态的二维网格。
        # 变量说明（start）：搜索、切片或计时过程使用的起始位置。
        # 变量说明（target_item）：`find` 中当前处理的物品。
        # 变量说明（self.grid）：按世界坐标存储材质或寻路状态的二维网格。
        self.grid = grid
        # 变量说明（self.curr_walkable）：寻路过程中基于最新障碍物更新的可通行网格。
        self.curr_walkable = self.walkable.copy()
        start = tuple(start)  # Convert to tuple for hashing

        # 变量说明（open_list）：A* 搜索中等待按估计代价扩展的候选节点优先队列。
        open_list = []
        heapq.heappush(open_list, (0, start))
        # 变量说明（closed_set）：A* 搜索中已经完成扩展、不需再次访问的坐标集合。
        closed_set = set()
        # 变量说明（parent）：当前路径、节点或对象的父级对象。
        parent = {start: None}
        while open_list:
            # 变量说明（current_cost）：A* 搜索从起点到当前节点的已知最小代价。
            # 变量说明（current_node）：`find` 中当前处理的节点。
            current_cost, current_node = heapq.heappop(open_list)
            if current_node in closed_set:
                continue
            closed_set.add(current_node)
            if self.grid[current_node] == target_item:
                # 变量说明（path）：当前读取或写入对象对应的文件系统路径。
                path = self.generate_path(parent, current_node)
                # find direction
                # 变量说明（np_path）：`find` 读取或写入当前文件时使用的完整路径。
                np_path = np.array(path)
                # 变量说明（movements）：寻路时允许尝试的四个相邻方向向量列表。
                movements = np_path[1:] - np_path[:-1]
                # 变量说明（move）：A* 搜索当前尝试的相邻移动方向向量。
                return [self.directions[tuple(move)] for move in movements]

            # Explore neighbors
            # 变量说明（c）：当前循环正在处理的颜色通道、列或计数值。
            # 变量说明（r）：当前循环正在处理的行索引、半径或红色通道值。
            for r, c in self.directions.keys():
                # 变量说明（neighbor）：A* 搜索中与当前坐标相邻的候选网格坐标。
                neighbor = (current_node[0] + r, current_node[1] + c)
                if neighbor in closed_set or not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                if self.grid[neighbor] == target_item:
                    parent[neighbor] = current_node
                    path = self.generate_path(parent, neighbor)
                    # find direction
                    np_path = np.array(path)
                    movements = np_path[1:] - np_path[:-1]
                    return [self.directions[tuple(move)] for move in movements]
                if not self.is_valid(neighbor):
                    continue
                # 变量说明（new_cost）：经当前节点到达相邻节点时计算出的候选路径代价。
                new_cost = current_cost + 1  # Uniform cost
                heapq.heappush(open_list, (new_cost, neighbor))
                parent[neighbor] = current_node

        # If the target is not reachable, find the closest node
        # 变量说明（target_positions）：`find` 中待批量处理的位置集合。
        target_positions = np.argwhere(self.grid == target_item)
        if target_positions.size == 0:
            # Target item not in grid at all
            return None

        # 变量说明（min_distance）：候选目标与当前玩家之间记录到的最小距离。
        min_distance = float('inf')
        # 变量说明（closest_node）：`find` 中当前处理的节点。
        closest_node = None
        # 变量说明（node）：当前遍历或处理的知识图谱节点标识。
        for node in closed_set:
            # 变量说明（target_pos）：寻路器准备到达的目标世界坐标。
            for target_pos in target_positions:
                # 变量说明（distance）：当前两个位置、对象或节点之间计算出的距离。
                distance = abs(node[0] - target_pos[0]) + abs(node[1] - target_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node

        if closest_node is not None:
            path = self.generate_path(parent, closest_node)
            np_path = np.array(path)
            movements = np_path[1:] - np_path[:-1]
            return [self.directions[tuple(move)] for move in movements]
        else:
            return None

if __name__ == '__main__':
    # Example usage with 2D numpy array:
    # 变量说明（grid）：按世界坐标存储材质或寻路状态的二维网格。
    grid = np.array([
        [2, 3, 3, 2, 2, 2],
        [16, 4, 4, 2, 2, 2],
        [8, 3, 3, 2, 2, 2],
        [16, 1, 2, 4, 4, 3],
        [1, 1, 13, 16, 4, 3],
        [1, 1, 1, 4, 4, 3],
    ])
    
    # 变量说明（pf）：根据可通行网格构造的 A* 寻路器实例。
    pf = PathFinding(walkable={2,4}, grid_size=(6, 6))
    # 变量说明（target_item）：模块 中当前处理的物品。
    target_item = 16  # User-defined target item to find the nearest walkable cell adjacent to
    # 变量说明（path）：当前读取或写入对象对应的文件系统路径。
    path = pf.find(grid, start=(0, 0), target_item=13)
    print(path)