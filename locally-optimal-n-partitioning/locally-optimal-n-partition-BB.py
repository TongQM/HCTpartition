import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import sqrt
from numba import jit, njit

@njit
def density_func(xc, yc):
    return 1/side**2

@njit
def distance(xc, yc):
    return np.linalg.norm(xc - yc, 2)

@njit
def indicator(xc, yc, i, w, depots):
    '''
    This function judges whether the point (xc, yc) belongs to the ith district
    xc, yc: the coordinate of the point to be considered
    i: the index of the district
    '''
    n = w.shape[0]
    distances_ls = np.array([distance(np.array([xc, yc]), depot) for depot in depots])
    pd_distances_ls = np.power(distances_ls, 2) - w
    smallest_i = np.argmin(pd_distances_ls)
    if i == smallest_i:
        return 1
    return 0


def make_i_indicator(i, w, depots):
    '''
    This function makes an indicator function that judges whether the point (xc, yc) belongs to the ith district
    i: the index of the district
    depots: the coordinates of the depots
    '''
    @njit
    def i_indicator(xc, yc):
        return indicator(xc, yc, i, w, depots)
    return i_indicator

@njit
def integrand(yc, xc, district_indicator, density_func):
    return district_indicator(xc, yc)*np.sqrt(density_func(xc, yc))


def BHHcoef(district_indicator, density_func, region_side):
    coef, error = integrate.dblquad(integrand, 0, region_side, 0, region_side, args=(district_indicator, density_func))
    return coef # error

class Square:
    def __init__(self, depots, side=10):
        self.depots = depots
        self.side = side
        self.perimeter = 4*self.side
        self.diameter = sqrt(2)*side

class Simplex:
    def __init__(self) -> None:
        pass



class Node:
    def __init__(self, density_func, region, xi_range, father) -> None:
        '''
        xi_range, an n x 2 array where ith row denotes the range of xi_i for this node
        father, the father 
        n, the dimension of xi, i.e. the number of districts
        '''
        self.n = xi_range.shape[0]
        self.region = region
        self.xi_range = xi_range
        self.father = father
        self.children = []
        self.lb = self.get_lb(density_func)
        self.ub = self.get_ub(density_func)

    def xi2w(self, xi):
        '''
        xi, a numpy narray
        '''
        xi_i2w_i = lambda xi_i: 1/(1 - xi_i) - 1/(xi_i)
        w = np.array([xi_i2w_i(xi_i) for xi_i in xi])
        return w

    def get_median(self):
        median = np.mean(self.xi_range, axis=1)
        return median
    
    def evaluate_single_BHHcoef(self, i, w, density_func):
        i_indicator = make_i_indicator(i, w, self.region.depots.copy())
        return BHHcoef(i_indicator, density_func, self.region.side)
    
    def evaluate(self, w, density_func):
        BHHcoefs_ls = np.array([self.evaluate_single_BHHcoef(i, w, density_func) for i in range(self.n)])
        denominator = sum(BHHcoefs_ls)
        return BHHcoefs_ls/denominator, denominator, BHHcoefs_ls

    def get_lb(self, density_func):
        
        xi_max = self.xi_range[:, 1]
        w_max = self.xi2w(xi_max)
        print(f"DEBUG: xi_max {xi_max}\n w_max {w_max}")
        denominator = self.evaluate(w_max, density_func)[1]

        xi_i_ls = []
        for i in range(self.n):
            xi_i = self.xi_range[:, 1]
            xi_i[i] = self.xi_range[i, 0]
            xi_i_ls.append(xi_i)
        w_i_ls = [self.xi2w(xi_i) for xi_i in xi_i_ls]
        biggest_wi_BHHcoef = max([self.evaluate_single_BHHcoef(i, w_i_ls[i],density_func) for i in range(self.n)])
        self.lb = biggest_wi_BHHcoef/denominator
        return self.lb
    
    def get_ub(self, density_func):
        xi_median = self.get_median()
        w_median = self.xi2w(xi_median)
        self.ub = max(self.evaluate(w_median, density_func)[0])
        return self.ub
    
    def branch(self):
        i_range = lambda i: self.xi_range[i][1] - self.xi_range[i][0]
        wide_i = max(list(range(self.n)), key=i_range)
        median_i = (self.xi_range[wide_i][0] + self.xi_range[wide_i][1])/2
        node1_xi_range, node2_xi_range = self.xi_range[:], self.xi_range[:]
        node1_xi_range[wide_i], node2_xi_range = [self.xi_range[0], median_i], [median_i, self.xi_range[1]]
        node1, node2 = Node(node1_xi_range, self), Node(node2_xi_range, self)
        self.children = [node1, node2]
        return self.children
    
    def get_bounds(self, density_func):
        return self.get_lb(density_func), self.get_ub(density_func)
    

class BranchAndBound:
    def __init__(self, region, density_func, initial_node, tol=1e-3) -> None:
        '''
        compact set: the set of whole region;
        '''
        self.region = region
        self.density_func = density_func
        self.tol = tol

        self.current_node = initial_node
        lb, ub = self.current_node.get_bounds(density_func)
        self.node_ls = [initial_node]
        self.unexplored_node_ls = self.node_ls[:]
        self.best_ub = ub
        self.worst_lb = lb

    def branch(self, node):
        '''
        This function divides one node into two nodes by separating the node from the center of the interval.
        '''
        subnode1, subnode2 = node.branch()
        return subnode1, subnode2

    def bound(self, node):
        LB, UB = node.get_bounds(self.density_func)
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


        return best_node
    

n = 3
side = 10
depots = np.random.rand(3, 2)*side
square = Square(depots, side)
initial_node = Node(density_func, square, np.tile([0.005, 0.995], (n, 1)), None)
problem = BranchAndBound(square, density_func, initial_node)
best_node = problem.solve()