import math

class Node:
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __eq__(self, other):
        return isinstance(other, Node) and self.X == other.X and self.Y == other.Y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.X, self.Y))

    def __lt__(self, other):
        return (self.X, self.Y) < (other.X, other.Y)


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def enqueue(self, item, priority):
        self.elements.append((item, priority))
        self.elements.sort(key=lambda x: x[1])

    def dequeue(self):
        return self.elements.pop(0)[0]

    def is_empty(self):
        return len(self.elements) == 0

class PathPlanning:
    def __init__(self):
        self.graph = {}
        self.MaxDistanceNode = 1.0
        self.ExtraNodeDistance = 0.1

    def initial_path_planning(self, nodes):
        self.automatic_adding_edge(nodes)

    def execution(self, start_node, end_node):
        start_node = self.find_closest_node(start_node)
        return self.shortest_path(start_node, end_node)

    def add_edge(self, _from, to, cost):
        if _from not in self.graph:
            self.graph[_from] = []
        self.graph[_from].append((to, cost))

    def automatic_adding_edge(self, all_nodes):
        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                if abs(all_nodes[i].X - all_nodes[j].X) <= self.MaxDistanceNode and \
                        abs(all_nodes[i].Y - all_nodes[j].Y) <= self.MaxDistanceNode:
                    self.add_edge(all_nodes[i], all_nodes[j], 1.0)
                    self.add_edge(all_nodes[j], all_nodes[i], 1.0)

                    distance_between = self.calculate_distance(all_nodes[i], all_nodes[j])
                    num_of_extra_nodes = int(distance_between / self.ExtraNodeDistance)

                    for k in range(1, num_of_extra_nodes + 1):
                        ratio = k / (num_of_extra_nodes + 1)
                        new_x = all_nodes[i].X + ratio * (all_nodes[j].X - all_nodes[i].X)
                        new_y = all_nodes[i].Y + ratio * (all_nodes[j].Y - all_nodes[i].Y)
                        extra_node = Node(new_x, new_y)

                        self.add_edge(all_nodes[i], extra_node, self.ExtraNodeDistance)
                        self.add_edge(extra_node, all_nodes[j], self.ExtraNodeDistance)

    def calculate_distance(self, node_a, node_b):
        return math.sqrt((node_a.X - node_b.X) ** 2 + (node_a.Y - node_b.Y) ** 2)

    def find_closest_node(self, target):
        closest_node = None
        min_distance = float('inf')

        for node in self.graph.keys():
            distance = math.sqrt((node.X - target.X) ** 2 + (node.Y - target.Y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_node = node

        return closest_node

    def shortest_path(self, start, destination):
        distance = {node: float('inf') for node in self.graph}
        previous = {node: None for node in self.graph}
        visited = set()
        queue = PriorityQueue()

        distance[start] = 0.0
        queue.enqueue(start, 0.0)

        while not queue.is_empty():
            current = queue.dequeue()

            if current in visited:
                continue

            visited.add(current)

            if current == destination:
                break

            if current not in self.graph:
                continue

            for neighbor_tuple in self.graph[current]:
                neighbor, weight = neighbor_tuple

                alt_distance = distance[current] + weight
                if alt_distance < distance.get(neighbor, float('inf')):
                    distance[neighbor] = alt_distance
                    previous[neighbor] = current
                    queue.enqueue(neighbor, alt_distance)

        path = []
        current_path_node = destination
        while current_path_node is not None:
            path.insert(0, current_path_node)
            current_path_node = previous[current_path_node]

        return path
