import numpy as np
import time
import torch
import torchquad
import numba as nb
from scipy import optimize, integrate, linalg
from classes import Region, Coordinate, Demands_generator, append_df_to_csv

# Some previously used instrumental functions
@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

@nb.jit(nopython=True)
def min_modified_norm(x_cdnt, demands_locations, lambdas):
    n = demands_locations.shape[0]
    norms = np.array([norm_func(x_cdnt, demands_locations[i]) - lambdas[i] for i in range(n)])
    return np.min(norms)

# Current implementation involving torchquad and vectorized integrands
def integrand(X, v, demands_locations, lambdas):
    '''
    X is a n-by-2 matrix, the first column is r, the second column is theta.
    '''
    dtype = X.dtype
    lambdas, v1, v0 = torch.tensor(lambdas, dtype=dtype), torch.tensor(v[1:], dtype=dtype), v[0]
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    x_cdnt = X.clone()
    x_cdnt[:, 0] = X[:, 0]*torch.cos(x_cdnt[:, 1])
    x_cdnt[:, 1] = X[:, 0]*torch.sin(x_cdnt[:, 1])
    norms = torch.cdist(x_cdnt, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    raw_intgrd = 1 / (v[0]*modified_norms + v[1])
    return X[:, 0]*raw_intgrd    # r as Jacobian


def jac_integrand0(X, v, demands_locations, lambdas):
    '''
    X is a n-by-2 matrix, the first column is r, the second column is theta.
    '''
    dtype = X.dtype
    lambdas, v1, v0 = torch.tensor(lambdas, dtype=dtype), torch.tensor(v[1:], dtype=dtype), v[0]
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    x_cdnt = X.clone()
    x_cdnt[:, 0] = X[:, 0]*torch.cos(x_cdnt[:, 1])
    x_cdnt[:, 1] = X[:, 0]*torch.sin(x_cdnt[:, 1])
    norms = torch.cdist(x_cdnt, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    raw_intgrd = -modified_norms / torch.square(v[0]*modified_norms + v[1])
    return X[:, 0]*raw_intgrd


def jac_integrand1(X, v, demands_locations, lambdas):
    '''
    X is a n-by-2 matrix, the first column is r, the second column is theta.
    '''
    dtype = X.dtype
    lambdas, v1, v0 = torch.tensor(lambdas, dtype=dtype), torch.tensor(v[1:], dtype=dtype), v[0]
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    x_cdnt = X.clone()
    x_cdnt[:, 0] = X[:, 0]*torch.cos(x_cdnt[:, 1])
    x_cdnt[:, 1] = X[:, 0]*torch.sin(x_cdnt[:, 1])
    norms = torch.cdist(x_cdnt, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    raw_intgrd = -1 / torch.square(v[0]*modified_norms + v[1])
    return X[:, 0]*raw_intgrd

def objective_function(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    simpson = torchquad.Simpson()

    time_before_problem14_obj_integral = time.time()
    area = simpson.integrate(lambda X: integrand(X, v, demands_locations, lambdas), dim=2, N=999999, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    with open('./timerecords/problem14_obj_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem14_obj_integral}\n')

    return area/4 + v[0]*t + v[1]

def objective_jac(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    simpson = torchquad.Simpson()

    time_before_problem14_jac_integral = time.time()
    area0 = simpson.integrate(lambda X: jac_integrand0(X, v, demands_locations, lambdas), dim=2, N=999999, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    area1 = simpson.integrate(lambda X: jac_integrand1(X, v, demands_locations, lambdas), dim=2, N=999999, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    with open('./timerecords/problem14_jac_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem14_jac_integral}\n')

    return np.array([area0/4 + t, area1/4 + 1])

def constraint_and_jac(demands_locations, lambdas, region_radius):
    return np.array([min(-lambdas), 1]), np.array([min(-lambdas), 1])

def minimize_problem14(demands, thetarange, lambdas, t, region_radius):
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    constraint_coeff, constraint_jac = constraint_and_jac(demands_locations, lambdas, region_radius)
    constraints_dict = {'type': 'ineq', 'fun': lambda v: constraint_coeff @ v, 'jac': lambda _: constraint_jac}
    bound = optimize.Bounds(0, np.inf)
    result = optimize.minimize(objective_function, x0=np.array([0., 1]), args=(demands_locations, lambdas, t, region_radius, thetarange), jac=objective_jac, method='SLSQP', bounds=bound, constraints=constraints_dict)
    return result.x, result.fun