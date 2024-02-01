from typing import Literal
import numpy as np
import torch
import time
import torchquad
import numba as nb
from scipy import optimize
from classes import Region, Coordinate, Demands_generator, Demand, append_df_to_csv

# Some previously used instrumental functions
@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

@nb.jit(nopython=True)
def modified_norm(x_cdnt: list[float], i: int, demands_locations: list[list[float]], lambdas: list[float]):
    return norm_func(x_cdnt, demands_locations[i]) - lambdas[i]

@nb.jit(nopython=True)
def region_indicator(i: int, x_cdnt: list[float], lambdas: list[float], demands_locations: list[list[float]]) -> Literal[0, 1]:
    i_modified_norm = modified_norm(x_cdnt, i, demands_locations, lambdas)
    for j in range(len(lambdas)):
        if i_modified_norm > modified_norm(x_cdnt, j, demands_locations, lambdas):
            return 0
    return 1

@nb.jit(nopython=True)
def categorize_x(x_cdnt: list[float], demands_locations: list[list[float]], lambdas: list[float], v: list[float]):
    modified_norms = np.array([modified_norm(x_cdnt, i, demands_locations, lambdas) for i in range(demands_locations.shape[0])])
    i = np.argmin(modified_norms)
    return demands_locations[i], v[i+1]

# Current implementation involving torchquad and vectorized integrands
def integrand(X, lambdas, v, demands_locations):
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
    raw_intgrd = 1 / (v0*norms[torch.arange(norms.shape[0]), modified_norms_indices] + v1[modified_norms_indices])
    if raw_intgrd.is_complex(): # This was to prevent the complex number error when using complex-step as the derivative apprxomation method.
        raise ValueError
    return X[:, 0]*raw_intgrd

def jac_integrand0(X, lambdas, v, demands_locations):
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
    cooresponding_norms = norms[torch.arange(norms.shape[0]), modified_norms_indices]
    raw_intgrd = cooresponding_norms / torch.square(v0*cooresponding_norms + v1[modified_norms_indices])
    return -X[:, 0] * raw_intgrd

def jac_integrandj(X, lambdas, v, demands_locations, j):
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
    maskj = modified_norms_indices != j-1
    cooresponding_norms = norms[torch.arange(norms.shape[0]), modified_norms_indices]
    raw_intgrd = 1 / torch.square(v0*cooresponding_norms + v1[modified_norms_indices])
    masked_intgrd = -X[:, 0] * raw_intgrd
    masked_intgrd[maskj] = 0
    return masked_intgrd

def objective_function(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    simpson = torchquad.Simpson()

    time_before_problem7_obj_integral = time.time()
    sum_integral = simpson.integrate(lambda X: integrand(X, lambdas, v, demands_locations), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    with open('./timerecords/problem7_obj_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem7_obj_integral}\n')
    
    return 1/4*sum_integral + v[0]*t + np.mean(v[1:])

def objective_jac(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    n = demands_locations.shape[0]
    jac = np.zeros(n + 1)
    simpson = torchquad.Simpson()

    time_before_problem7_jac_integral = time.time()
    jac[0] = 1/4 * simpson.integrate(lambda X: jac_integrand0(X, lambdas, v, demands_locations), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch').item() + t
    # The precompiled version is even slower, so we don't use it.
    # for j in torch.range(1, n): # notice torch.range is inclusive on both ends, hence we use n as upper bound
    #     if j == 1:
    #         simpsonj_compiled = torch.jit.trace(
    #             lambda vj_index: simpson.integrate(lambda X: jac_integrandj(X, lambdas, v, demands_locations, vj_index), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch'),
    #             (j,)
    #                 )
    #     jac[int(j.item())] = 1/4 * simpsonj_compiled(j).item() + 1/n


    for j in range(1, n+1):
        jac[j] = 1/4 * simpson.integrate(lambda X: jac_integrandj(X, lambdas, v, demands_locations, j), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch').item() + 1/n
    with open('./timerecords/problem7_jac_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem7_jac_integral}\n')

    return jac


def minimize_problem7(lambdas, demands, thetarange, t, region_radius, tol):
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    bounds = optimize.Bounds(0, np.inf)
    result = optimize.minimize(objective_function, x0=np.append(1e-6, np.ones(demands.shape[0])), args=(demands_locations, lambdas, t, region_radius, thetarange), jac=objective_jac, method='SLSQP',  bounds=bounds, options={'ftol': tol, 'disp': True})
    return result.x, result.fun
