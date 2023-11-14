import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from scipy import integrate, optimize
from math import sqrt, pi
from numba import jit, njit

counter = 0
# Help functions

def polar2Cartesian(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])

def Cartesian2polar(x, y):
    return np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)])

@njit
def distance(xc, yc):
    '''
    return the Euclidean distance between two points with coordinates (xc, yc) under Cartesian system
    '''
    return np.linalg.norm(xc - yc, 2)

# Global functions

@njit
def density_func(r, t):
    return 1/(2*pi)*np.exp(-t**2/2)

@njit
def indicator(r, t, i, phis, depots):
    '''
    This function judges whether the point (rc, tc) belongs to the ith district
    rc, tc: the polar coordinate of the point to be considered
    i: the index of the district
    phis: the [starting angle, ending angle] of districts
    '''
    if t < phis[i][1] and t >= phis[i][0]: return 1
    return 0


def make_i_indicator(i, phis, region):
    '''
    This function makes an indicator function that judges whether the point (xc, yc) belongs to the ith district
    i: the index of the district
    depots: the coordinates of the depots
    '''
    # @njit
    def i_indicator(r, t):
        return indicator(r, t, i, phis, region.depots)
    return i_indicator

# @njit
def integrand(r, t, district_indicator, density_func):
    return density_func(r, t)*district_indicator(r, t)*r


def BHHcoef(i, phis, density_func, region):
    district_indicator = make_i_indicator(i, phis, region)
    coef, error = integrate.dblquad(integrand, 0, 2*pi, lambda _: 0, lambda _: region.radius, args=(district_indicator, density_func))
    return coef # error

# Classes

class Region:
    def __init__(self, radius: float, depot=np.array([0, 0])):
        self.radius = radius
        self.diam = 2*radius
        self.depots = depot

    def __repr__(self) -> str:
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.scatter([], [])
        # plt.show()
        return f'radius: {self.radius}'
    
    def __str__(self) -> str:
        return self.__repr__()


class Polyhedron:
    '''
    The polyhedron of all potential partitions, on which branch and bound is applied
    '''
    def __init__(self, A, b, B, c, dimension):
        '''
        Polyhedron determined by Ax<=b form and Bx=c
        '''
        self.A, self.b = A, b
        self.B, self.c = B, c
        self.dim = dimension
        self.eq_constraints = {'type': 'eq', 'fun': lambda x: self.B @ x - self.c , 'jac': lambda _: self.B}
        # optimize.LinearConstraint(B, c, c)
        self.ineq_constraints = {'type': 'ineq', 'fun': lambda x: self.b - self.A @ x + 1e-6, 'jac': lambda _: -self.A}
        # optimize.LinearConstraint(A, -np.inf, b + 1e-4, keep_feasible=False)

    def add_ineq_constraint(self, ai, bi):
        '''
        Add an inequality constraint and generate a new polyhedron
        While the original polyhedron is not changed
        '''
        # self.A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        # self.b = np.append(self.b, bi)
        # self.ineq_constraints = optimize.LinearConstraint(self.A, -np.inf, self.b)
        A = np.append(self.A, ai.reshape(1, ai.size), axis=0)
        b = np.append(self.b, bi)
        return Polyhedron(A, b, self.B, self.c, self.dim)

    def find_analytic_center(self, x0):
        # Find a feasible solution to the problem first
        find_feasible_sol = gp.Model('find_feasible_sol')
        find_feasible_sol.setParam('OutputFlag', 0)
        x = find_feasible_sol.addMVar(shape=self.dim, lb=0, ub=pi, name='x')
        find_feasible_sol.addConstr(self.B @ x == self.c)
        find_feasible_sol.addConstr(self.A @ x <= self.b)
        find_feasible_sol.setObjective(0, gp.GRB.MINIMIZE)
        find_feasible_sol.optimize()
        # assert find_feasible_sol.status == gp.GRB.OPTIMAL, find_feasible_sol.status
        x0 = x.X

        objective = lambda x: -np.sum(np.log(self.b - self.A @ x + 1e-6))  # To ensure log(b - A @ x) is defined.
        objective_jac = lambda x: np.sum(np.array([self.A[i, :]/(self.b[i] - self.A[i, :] @ x) for i in range(self.A.shape[0])]), axis=0)
        result = optimize.minimize(objective, x0, method='SLSQP', constraints=[self.ineq_constraints, self.eq_constraints], jac='cs', tol=0.1, options={'disp': True})
        assert result.success, result.message
        analytic_center, analytic_center_val = result.x, result.fun            
        return analytic_center, analytic_center_val

    def find_width_in_dimension(self, i):
        '''
        Find the width of the polyhedron in the ith dimension
        '''
        # Find the max value of the dimension i in the polyhedron
        model_max = gp.Model('max')
        model_max.setParam('OutputFlag', 0)
        phi_max = model_max.addMVar(shape=self.dim, lb=0, ub=pi, name='phi')
        model_max.addConstr(self.B @ phi_max == self.c)
        model_max.addConstr(self.A @ phi_max <= self.b)
        model_max.setObjective(phi_max[i], gp.GRB.MAXIMIZE)
        model_max.optimize()

        # Find the min value of the dimension i in the polyhedron
        model_min = gp.Model('min')
        model_min.setParam('OutputFlag', 0)
        phi_min = model_min.addMVar(shape=self.dim, lb=0, ub=pi, name='phi')
        model_min.addConstr(self.B @ phi_min == self.c)
        model_min.addConstr(self.A @ phi_min <= self.b)
        model_min.setObjective(phi_min[i], gp.GRB.MINIMIZE)
        model_min.optimize()

        return [phi_min[i].X, phi_max[i].X]

    def get_ranges(self):
        '''
        For each dimension of the polyhedron, get the range of that dimension
        '''
        ranges = []
        for i in range(self.dim):
            ranges.append(self.find_width_in_dimension(i))
        return np.array(ranges)

    def show_constraints(self):
        print(f'A: {self.A} \n b: {self.b} \n B: {self.B} \n c: {self.c}.')



class Node:
    def __init__(self, density_func, region, polyhedron, father) -> None:
        '''
        ri_range, an n x 2 array where ith row denotes the range of xi_i for this node
        father, the father 
        n, the dimension of xi, i.e. the number of districts
        '''
        self.n = polyhedron.dim
        self.density_func = density_func
        self.region = region
        self.polyhedron = polyhedron
        self.father = father
        self.children = []
        self.lb = self.get_lb(density_func)
        self.ub = self.get_ub(density_func)

    def parition2range(self, partition):
        '''
        partition, a list of n-1 numbers, which are the boundaries of the n districts
        '''
        partition = np.array(partition)
        partition = np.append(partition, 2*pi)
        partition = np.insert(partition, 0, 0)
        return np.array([partition[:-1], partition[1:]]).T

    def get_median(self):
        '''
        Get a median point of the node, to calculate the upper bound
        '''
        analytic_center, analytic_center_val = self.polyhedron.find_analytic_center(np.ones(self.n))
        return analytic_center
    
    def evaluate_single_BHHcoef(self, i, phis, density_func):
        return BHHcoef(i, phis, density_func, self.region)
    
    def evaluate(self, phis, density_func):
        BHHcoefs_ls = np.array([self.evaluate_single_BHHcoef(i, phis, density_func) for i in range(self.n)])
        denominator = sum(BHHcoefs_ls)
        return BHHcoefs_ls/denominator, denominator, BHHcoefs_ls

    def get_lb(self, density_func):
        
        # xi_max = self.ri_range[:, 1]
        # w_max = self.xi2w(xi_max)
        # print(f"DEBUG: xi_max {xi_max}\n w_max {w_max}")
        # denominator = self.evaluate(w_max, density_func)[1]

        # xi_i_ls = []
        # for i in range(self.n):
        #     xi_i = self.ri_range[:, 1]
        #     xi_i[i] = self.ri_range[i, 0]
        #     xi_i_ls.append(xi_i)
        # w_i_ls = [self.xi2w(xi_i) for xi_i in xi_i_ls]
        # biggest_wi_BHHcoef = max([self.evaluate_single_BHHcoef(i, w_i_ls[i],density_func) for i in range(self.n)])
        # self.lb = biggest_wi_BHHcoef/denominator
        return 0
    
    def get_ub(self, density_func):
        partition_median = self.get_median()
        phis_median = self.parition2range(partition_median)
        self.ub = max(self.evaluate(phis_median, density_func)[0])
        return self.ub
    
    def branch(self):
        '''
        Branch the node into two nodes by separating the node along the widest dimension of the corresponding polyhedorn
        '''
        ranges = self.polyhedron.get_ranges()
        wide_i = np.argmax(ranges[:, 1] - ranges[:, 0])
        range_i = ranges[wide_i]
        median_i = (range_i[0] + range_i[1])/2
        print(f"DEBUG: wide_i: {wide_i}\n\t range_i: {range_i}\n\t median_i: {median_i}")
        node1_polyhedron = self.polyhedron.add_ineq_constraint(np.array([0 if i != wide_i else 1 for i in range(self.n)]), median_i)
        node2_polyhedron = self.polyhedron.add_ineq_constraint(np.array([0 if i != wide_i else -1 for i in range(self.n)]), -median_i)
        node1 = Node(self.density_func, self.region, node1_polyhedron, self)
        node2 = Node(self.density_func, self.region, node2_polyhedron, self)
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

        while self.best_ub - self.worst_lb > self.tol and self.best_ub >= 1/n + 0.01:
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
        
        best_node = min(self.node_ls, key=lambda node: node.ub)

        return best_node
    

n = 3
radius = 10
region = Region(radius=radius)

A = np.eye(n)
b = np.ones(n)*pi
B = np.ones((1, n))
c = 2*pi
initial_polyhedtron = Polyhedron(A, b, B, c, n)
initial_node = Node(density_func, region, initial_polyhedtron, None)
problem = BranchAndBound(region, density_func, initial_node)
best_node = problem.solve()

# write a function to cumulate numbers in a list
def cumulate(ls):
    cumulated_ls = []
    for i in range(len(ls)):
        cumulated_ls.append(sum(ls[:i+1]))
    return cumulated_ls