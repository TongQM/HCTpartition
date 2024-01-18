import time
import torch
import torchquad
from problem14 import minimize_problem14
from problem7 import minimize_problem7
import torch
import numpy as np

import numba as nb
import torchquad
from scipy import optimize
from classes import Region, Coordinate, Demand, Polyhedron

from torchquad import set_up_backend
set_up_backend("torch", data_type="float32")

def findWorstTSPDensity(region, demands, thetarange, t, epsilon, tol: float=1e-4):
    '''
    Algorithm by Carlsson, Behroozl, and Mihic, 2018.
    Code by Yidi Miao, 2023.

    This algorithm (find the worst TSP density) takes as input a compact planar region containing 
    a set of n distinct points, a distance threshold t, and a tolerance epsilon.

    Input: A compact, planar region Rg containing a set of distinct points x1, x2,..., xn, which are 
    interpreted as an empirical distribution f_hat, a distance parameter t, and a tolerance epsilon.

    Output: An epsilon-approximation of the distribution f* that maximizes iint_Rg sqrt(f(x)) dA 
    subject to the constraint that D(f_hat, f) <= t.

    This is a standard analytic center cutting plane method applied to problem (13), which has an 
    n-dimensional variable space.
    '''

    start, end = thetarange
    n = demands.shape[0]
    simpson = torchquad.Simpson()
    UB, LB = np.inf, -np.inf
    UB_lst, LB_lst = [], []
    lambdas_bar = np.zeros(n)
    polyhedron = Polyhedron(np.eye(n), region.diam*np.ones(n), np.ones((1, n)), 0, n)
    k = 1
    while (abs(UB - LB) > epsilon and k < 100):
        print(f'\t Looking for worst-distribution on {[start, end]}:\n\t Iteration {k} begins: \n')
        starttime = time.time()
        try:
            result = polyhedron.find_analytic_center(lambdas_bar)
            if result == "EMPTY": return lambda X: f_tilde(X, demands_locations, lambdas_bar, v_tilde), UB_lst, LB_lst
            lambdas_bar, lambdas_bar_func_val = result
        except Exception as e:
            print(f"An error occurred: {e}")
            return polyhedron, lambdas_bar, v_tilde

        print(f'\t Find analytic center: Lambdas_bar is {lambdas_bar}, with value {lambdas_bar_func_val}, took {time.time() - starttime}s.\n')
        with open("./timerecords/find_analytic_center.txt", "a") as f:
            f.write(f"{time.time() - starttime}\n")

        demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])

        '''Build an upper bounding f_bar for the original problem (4).'''
        time_before_minimize_problem14 = time.time()
        v_bar, problem14_func_val = minimize_problem14(demands, thetarange, lambdas_bar, t, region.radius)
        with open("./timerecords/minimize_problem14.txt", "a") as f:
            f.write(f"{time.time() - time_before_minimize_problem14}\n")

        time_before_ub_integral = time.time()
        upper_integrand = lambda X: X[:, 0]*torch.sqrt(f_bar(X, demands_locations, lambdas_bar, v_bar))
        UB = simpson.integrate(upper_integrand, dim=2, N=999999, integration_domain=[[0, region.radius],[start, end]], backend='torch').item()
        with open("./timerecords/ub_integral.txt", "a") as f:
            f.write(f"{time.time() - time_before_ub_integral}\n")

        print(f'\t UB is {UB}, took {time.time() - starttime}s.\n')

        if UB < 0:
            print(f'\t UB is negative: {UB}.')
            print(f'\t v_bar is {v_bar}, problem14_func_val is {problem14_func_val}.')
            break

        '''Build an lower bounding f_tilde that us feasible for (4) by construction.'''
        time_before_minimize_problem7 = time.time()
        v_tilde, problem7_func_val = minimize_problem7(lambdas_bar, demands, thetarange, t, region.radius, tol)
        with open("./timerecords/minimize_problem7.txt", "a") as f:
            f.write(f"{time.time() - time_before_minimize_problem7}\n")

        time_before_lb_integral = time.time()
        lower_integrand = lambda X: X[:, 0]*torch.sqrt(f_tilde(X, demands_locations, lambdas_bar, v_tilde))
        LB = simpson.integrate(lower_integrand, dim=2, N=999999, integration_domain=[[0, region.radius],[start, end]], backend='torch').item()
        with open("./timerecords/lb_integral.txt", "a") as f:
            f.write(f"{time.time() - time_before_lb_integral}\n")
        
        print(f'\t LB is {LB}, took {time.time() - starttime}s.\n')

        '''Update g.'''
        g = np.zeros(len(demands))
        time_before_g_integral = time.time()
        for i in range(len(demands)):
            integrandi = lambda X: X[:, 0]*f_bar(X, demands_locations, lambdas_bar, v_bar, masked=True, j=i+1) 
            g[i] = simpson.integrate(integrandi, dim=2, N=999999, integration_domain=[[0, region.radius], [start, end]], backend='torch').item()
        with open("./timerecords/g_integral.txt", "a") as f:
            f.write(f"{time.time() - time_before_g_integral}\n")
        
        '''Update polyheron Lambda to get next analytic center.'''
        polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar)

        print(f'\t End of iteration {k}.\n  The whole iteration took {time.time() - starttime}s.\n')
        k += 1
        UB_lst.append(UB), LB_lst.append(LB)
        if UB < LB:
            raise Exception(f"UB {UB} is smaller than LB {LB}.")

    return lambda X: f_tilde(X, demands_locations, lambdas_bar, v_tilde), UB_lst, LB_lst


def f_bar(X, demands_locations, lambdas_bar, v_bar, masked=False, j=-1):
    '''
    X is a n-by-2 matrix, the first column is r, the second column is theta.
    '''
    dtype = X.dtype
    lambdas = torch.tensor(lambdas_bar, dtype=dtype)
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    x_cdnt = X.clone()
    x_cdnt[:, 0] = X[:, 0]*torch.cos(x_cdnt[:, 1])
    x_cdnt[:, 1] = X[:, 0]*torch.sin(x_cdnt[:, 1])
    norms = torch.cdist(x_cdnt, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    raw_intgrd = 1 / (4*torch.square(v_bar[0]*modified_norms + v_bar[1]))
    if masked:        
        mask = modified_norms_indices != j-1
        masked_intgrd = raw_intgrd
        masked_intgrd[mask] = 0
        return masked_intgrd
    return raw_intgrd


def f_tilde(X, demands_locations, lambdas_bar, v_tilde):
    '''
    X is a n-by-2 matrix, the first column is r, the second column is theta.
    '''
    dtype = X.dtype
    lambdas, v1, v0 = torch.tensor(lambdas_bar, dtype=dtype), torch.tensor(v_tilde[1:], dtype=dtype), v_tilde[0]
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    x_cdnt = X.clone()
    x_cdnt[:, 0] = X[:, 0]*torch.cos(x_cdnt[:, 1])
    x_cdnt[:, 1] = X[:, 0]*torch.sin(x_cdnt[:, 1])
    norms = torch.cdist(x_cdnt, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    corresponding_norms = norms[torch.arange(norms.shape[0]), modified_norms_indices]
    raw_intgrd = 1 / (4*torch.square(v0*corresponding_norms + v1[modified_norms_indices]))
    return raw_intgrd