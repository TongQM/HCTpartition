import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from scipy import integrate, optimize, stats
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
    if t < 0 or t > 2*pi:
        t = t % (2*pi)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    # Parameters of the Gaussian peaks: (center_r, center_theta, amplitude, width)
    # You need to convert the (center_x, center_y) of each peak to polar coordinates as well
    peaks_polar = [
        (0.424, np.arctan2(0.3, 0.3), 1.0, 0.1),  # Converted (0.3, 0.3) to polar
        (0.566, np.arctan2(0.4, -0.4), 1.0, 0.1), # Converted (-0.4, 0.4) to polar
        (0.5, np.arctan2(-0.5, 0.0), 1.0, 0.1)     # Converted (0, -0.5) to polar
    ]
    
    # Initialize the density to zero
    density = 0
    
    # Sum the contributions from each Gaussian peak
    for (center_r, center_theta, amplitude, width) in peaks_polar:
        # Calculate the Cartesian coordinates of the peak center
        center_x = center_r * np.cos(center_theta)
        center_y = center_r * np.sin(center_theta)
        
        # Calculate the squared distance from the peak center in Cartesian coordinates
        distance_squared = (x - center_x)**2 + (y - center_y)**2
        
        # Gaussian function value for this peak
        gaussian_value = amplitude * np.exp(-distance_squared / (2 * width**2))
        
        # Add the value to the total density
        density += gaussian_value
    
    return density

@njit
def integrand(r, t, density_func):
    return sqrt(density_func(r, t))*r


def BHHcoef(trange, density_func, region):
    start, end = trange
    coef, error = integrate.dblquad(integrand, start, end, lambda _: 0, lambda _: region.radius, args=(density_func,))
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
    
    def find_min_value_of(self, k):
        '''
        Find the minimum value of k'th decision variable
        '''
        model_min = gp.Model('min')
        model_min.setParam('OutputFlag', 0)
        phi_min = model_min.addMVar(shape=self.dim, lb=0, ub=pi, name='phi')
        model_min.addConstr(self.B @ phi_min == self.c)
        model_min.addConstr(self.A @ phi_min <= self.b)
        model_min.setObjective(phi_min[k], gp.GRB.MINIMIZE)
        model_min.optimize()

        return phi_min[k].X

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
        n, the dimension of xi, i.e. the number of districts
        '''
        self.n = polyhedron.dim // 2
        self.density_func = density_func
        self.region = region
        self.polyhedron = polyhedron
        self.father = father

        self.children = []
        self.lb = self.get_lb(density_func)
        self.ub = self.get_ub(density_func)


    def parition2trange(self, partition):
        '''
        partition, feasible solution in the node's polyhedron
        output a list of tranges, each row corresponding to a district
        '''
        tranges = []
        for i in range(self.n):
            if i == 0:
                tranges.append([-partition[2*i - 1], partition[2*i]])
            else:
                tranges.append([sum(partition[:2*i - 1]), sum(partition[:2*i + 1])])
        tranges = np.array(tranges)*2*pi
        return tranges

    def get_median(self):
        '''
        Get a median point of the node, to calculate the upper bound
        '''
        analytic_center, analytic_center_val = self.polyhedron.find_analytic_center(np.ones(self.n))
        return analytic_center
    
    def evaluate_single_BHHcoef(self, trange, density_func):
        return BHHcoef(trange, density_func, self.region)
    
    def evaluate(self, tranges, density_func):
        BHHcoefs_ls = np.array([self.evaluate_single_BHHcoef(tranges[i], density_func) for i in range(self.n)])
        return BHHcoefs_ls

    def get_lb(self, density_func):
        BHHcoef_ls = []
        for i in range(self.n):
            t1 = self.polyhedron.find_min_value_of(2*i - 1)
            t2 = self.polyhedron.find_min_value_of(2*i)
            trange = np.array([i/3 - t1, i/3 + t2])*2*pi
            BHHcoef = self.evaluate_single_BHHcoef(trange, density_func)
            BHHcoef_ls.append(BHHcoef)
        self.lb = max(BHHcoef_ls)
        return self.lb
    
    def get_ub(self, density_func):
        partition_median = self.get_median()
        phis_median = self.parition2trange(partition_median)
        self.ub = max(self.evaluate(phis_median, density_func))
        return self.ub
    
    def branch(self):
        '''
        Branch the node into two nodes by separating the node along the widest dimension of the corresponding polyhedorn
        '''
        ranges = self.polyhedron.get_ranges()
        wide_i = np.argmax(ranges[:, 1] - ranges[:, 0])
        range_i = ranges[wide_i]
        median_i = (range_i[0] + range_i[1])/2
        # print(f"DEBUG: wide_i: {wide_i}\n\t range_i: {range_i}\n\t median_i: {median_i}")
        node1_polyhedron = self.polyhedron.add_ineq_constraint(np.array([0 if i != wide_i else 1 for i in range(self.n*2)]), median_i)
        node2_polyhedron = self.polyhedron.add_ineq_constraint(np.array([0 if i != wide_i else -1 for i in range(self.n*2)]), -median_i)
        node1 = Node(self.density_func, self.region, node1_polyhedron, self)
        node2 = Node(self.density_func, self.region, node2_polyhedron, self)
        self.children = [node1, node2]
        return self.children
    
    def get_bounds(self, density_func):
        return self.get_lb(density_func), self.get_ub(density_func)
    

class BranchAndBound:
    def __init__(self, region, density_func, initial_node, tol=0.01, maxiter=1000) -> None:
        '''
        compact set: the set of whole region;
        '''
        self.region = region
        self.density_func = density_func
        self.tol = tol
        self.maxiter = maxiter

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
        counter = 0
        bounds_tracker = {}
        while self.best_ub - self.worst_lb > self.tol and counter < self.maxiter:
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
            counter += 1
            bounds_tracker[counter] = (self.best_ub, self.worst_lb)
        
        best_node = min(self.node_ls, key=lambda node: node.ub)

        return best_node, bounds_tracker
    

n = 3
radius = 10
region = Region(radius=radius)

A1 = np.zeros((n, 2*n))
for i in range(n):
    A1[i, 2*i-1] = 1
    A1[i, 2*i] = 1
A2 = -np.eye(2*n)
A = np.append(A1, A2, axis=0)

b1 = np.ones(n)*1/2
b2 = np.zeros(2*n)
b = np.append(b1, b2)

B = np.zeros((n, 2*n))
for i in range(n):
    B[i, 2*i] = 1
    B[i, 2*i+1] = 1
c = 1/n

initial_polyhedron = Polyhedron(A, b, B, c, 2*n)
initial_node = Node(density_func, region=region, polyhedron=initial_polyhedron, father=None)
problem = BranchAndBound(region, density_func, initial_node)
best_node, bounds_tracker = problem.solve()
lb_ls = [bd[0] for bd in bounds_tracker.values()]
ub_ls = [bd[1] for bd in bounds_tracker.values()]
plt.plot(lb_ls, label='lb')
plt.plot(ub_ls, label='ub')
plt.legend()
plt.show()
