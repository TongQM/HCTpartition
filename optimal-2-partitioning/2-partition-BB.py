import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


class Square:
    def __init__(self, side=10):
        self.side = side
        self.perimeter = 4*self.side
    
    def c2d_map(self, c):
        '''
        c: the distance from the left-bottom corner of the Square to the starting end point, along the boundary of the Square counterclockwise, c in [0, 2*side);
        d: the distance from the left-bottom corner of the Square to the ending end point, along the aboundary of the Square counterclockwise, d in [2*side, perimeter)
        '''
        return c + 2*self.side

    def get_coordinates(self, perimeter_distance):
        if perimeter_distance > 4*self.side:
            raise ValueError

        if perimeter_distance <= self.side:
            return (perimeter_distance, 0)
        elif perimeter_distance <= 2*self.side:
            return (self.side, perimeter_distance - self.side)
        elif perimeter_distance <= 3*self.side:
            return (self.side - perimeter_distance + 2*self.side, self.side)
        else:
            return (0, self.side - perimeter_distance + 3*self.side)

    def get_partition_line(self, c, d):
        c_coordinate = self.get_coordinates(c)
        d_coordinate = self.get_coordinates(d)
        if c_coordinate[0] == d_coordinate[0]: return "vertical"
        k = (d_coordinate[1] - c_coordinate[1])/(d_coordinate[0] - c_coordinate[0])
        b = c_coordinate[1] - k*c_coordinate[0]

        return (k, b)


class c2d_map:
    def __init__(self, c, square) -> None:
        self.c = c
        self.square = square

    def find_d(self):
        pass

class Node:
    def __init__(self, c_range, lb, ub, father, children=[]) -> None:
        self.c_range = c_range
        self.lb = lb
        self.ub = ub
        self.father = father
        self.children = children

class BranchAndBound:
    def __init__(self, compact_set, cost_funcs, density_func, c0=0, tol=1e-3) -> None:
        '''
        compact set: the set of whole region;
        cost_func: the cost function of a subregion;
        starting_c0: one end point of the starting partition.
        '''
        self.square = compact_set
        self.cost_func = cost_funcs[0]
        self.W1 = cost_funcs[1]
        self.W2 = cost_funcs[2]
        self.c0 = c0
        self.d0 = compact_set.c2d_map(self.c0)
        self.density_func = density_func
        self.tol = tol

        initial_node = [self.c0, self.d0]
        lb, ub = self.bound(initial_node)
        self.node_ls = [Node(initial_node, lb, ub, None)]
        self.unexplored_node_ls = self.node_ls[:]
        self.best_ub = ub
        self.worst_lb = lb

    def branch(self, node):
        '''
        A node is a pair of lb and ub of c
        This function divides one node into two nodes by separating the node from the center of the interval.
        '''
        node = node.c_range
        subnode1_range, subnode2_range = [node[0], (node[0] + node[1])/2], [(node[0] + node[1])/2, node[1]]
        subnode1_lb, subnode1_ub = self.bound(subnode1_range)
        subnode2_lb, subnode2_ub = self.bound(subnode2_range)
        subnode1, subnode2 = Node(subnode1_range, subnode1_lb, subnode1_ub, node), Node(subnode2_range, subnode2_lb, subnode2_ub, node)
        return subnode1, subnode2

    def bound(self, node):
        median_c = (node[0] + node[1])/2
        median_d = self.square.c2d_map(median_c)
        UB = self.cost_func(median_c, median_d, self.density_func, self.square)

        LB_left_bottom = self.W1(node[0], self.square.c2d_map(node[1]), self.density_func, self.square)
        LB_right_top = self.W2(node[1], self.square.c2d_map(node[0]), self.density_func, self.square)
        LB = max(LB_left_bottom, LB_right_top)

        return LB, UB

    def get_iter_bds(self, node):
        if not node.children:
            return (node.lb, node.ub)
        children_bds = [self.get_iter_bds(child) for child in node.children]
        children_lb = [bd[0] for bd in children_bds]
        children_ub = [bd[1] for bd in children_bds]
        return min(children_lb), min(children_ub)
    

    def solve(self):

        while self.best_ub - self.worst_lb > self.tol:
            # Branch and bound
            node = self.unexplored_node_ls[0]
            self.unexplored_node_ls.remove(node)
            new_node1, new_node2 = self.branch(node)
            node.children = [new_node1, new_node2]
            self.node_ls = self.node_ls + node.children
            self.unexplored_node_ls = self.unexplored_node_ls + node.children
            bds_ls = [self.get_iter_bds(node) for node in self.node_ls]
            self.lb_ls = [bd[0] for bd in bds_ls]
            self.ub_ls = [bd[1] for bd in bds_ls]
            
            # Pruning unnecessary nodes
            self.best_ub = min(self.ub_ls)
            self.worst_lb = min(self.lb_ls)
            print(f"DEBUG: best ub: {self.best_ub}\n\t worst lb: {self.worst_lb}\n\t gap: {self.best_ub - self.worst_lb}.")
            for node in self.node_ls:
                if node.lb > self.best_ub:
                    self.node_ls.remove(node)
            for node in self.unexplored_node_ls:
                if node.lb > self.best_ub:
                    self.unexplored_node_ls.remove(node)
        
        best_node = min(self.node_ls, key=lambda node: self.get_iter_bds(node)[1])
        best_c = (best_node.c_range[0] + best_node.c_range[1])/2
        best_d = self.square.c2d_map(best_c)

        return best_c, best_d, self.square.get_coordinates(best_c), self.square.get_coordinates(best_d)


def W1(c, d, density_func, square):
    c_coordiantes, d_coordinates = square.get_coordinates(c), square.get_coordinates(d)
    partition_line = square.get_partition_line(c, d)
    if partition_line == "vertical":
        result, error = integrate.dblquad(density_func, 0, c_coordiantes[0], 0, square.side)
    else:
        k, b = partition_line
        if c_coordiantes[0] < d_coordinates[0]:
            result, error = integrate.dblquad(density_func, 0, d_coordinates[0], lambda x: max(0, k*x + b), square.side)
        elif c_coordiantes[0] > d_coordinates[0]:
            result, error = integrate.dblquad(density_func, 0, c_coordiantes[0], 0, lambda x: min(k*x + b, square.side))
    return result

def W2(c, d, density_func, square):
    c_coordiantes, d_coordinates = square.get_coordinates(c), square.get_coordinates(d)
    partition_line = square.get_partition_line(c, d)
    if partition_line == "vertical":
        result, error = integrate.dblquad(density_func, c_coordiantes[0], square.side, 0, square.side)
    else:
        k, b = partition_line
        if c_coordiantes[0] < d_coordinates[0]:
            result, error = integrate.dblquad(density_func, c_coordiantes[0], square.side, 0, lambda x: min(square.side, k*x + b))
        elif c_coordiantes[0] > d_coordinates[0]:
            result, error = integrate.dblquad(density_func, d_coordinates[0], square.side, lambda x: max(k*x + b, 0), square.side)
    return result
    
def W(c, d, density_func, square):
    return max(W1(c, d, density_func, square), W2(c, d, density_func, square))


side = 10
density_func = lambda y, x: 1/(side*side)
square = Square(side)
problem = BranchAndBound(square, (W, W1, W2), density_func)
best_c, best_d, c_coord, d_coord = problem.solve()

# Define the side length of the square
side_length = 10

# Define the coordinates for the line
x_coords = [c_coord[0], d_coord[0]] # x-coordinates
y_coords = [c_coord[1], d_coord[1]] # y-coordinates

# Create a new figure
plt.figure()

# Draw the square
square_x = [0, side_length, side_length, 0, 0]
square_y = [0, 0, side_length, side_length, 0]
plt.plot(square_x, square_y, label='Square', color='blue')

# Draw the line
plt.plot(x_coords, y_coords, label='Line', color='red')

# Set axis limits and labels
plt.xlim(-5, 15)
plt.ylim(-5, 15)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.title('Square with Line')
plt.savefig('2-partition-BB.png')