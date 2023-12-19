from typing import Literal
import numpy as np
import torch
import time
import torchquad
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numba as nb
from scipy import optimize, integrate, linalg
from classes import Region, Coordinate, Demands_generator, Demand, append_df_to_csv

tol = 1e-3
# Instrumental functions
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

# Integrands of function and derivatives
# @nb.jit(nopython=True)
def integrand(X, lambdas, v, demands_locations):
    '''
    X is a n-by-2 matrix, the first column is r, the second column is theta.
    '''
    dtype = X.dtype
    print(f"lambdas is {lambdas}., v0 is {v[0]}, v1 is {v[1:]}.")
    lambdas, v1, v0 = torch.tensor(lambdas, dtype=dtype), torch.tensor(v[1:], dtype=dtype), v[0]
    print(f"lambdas is {lambdas}., v0 is {v0}, v1 is {v1}.")
    demands_locations = torch.tensor(demands_locations, dtype=dtype)
    x_cdnt = X.clone()
    x_cdnt[:, 0] = X[:, 0]*torch.cos(X[:, 1])
    x_cdnt[:, 1] = X[:, 0]*torch.sin(X[:, 1])
    norms = torch.cdist(x_cdnt, demands_locations, p=2)
    modified_norms, modified_norms_indices = torch.min(norms - lambdas, dim=1)
    raw_intgrd = 1 / (v0*norms[torch.arange(norms.shape[0]), modified_norms_indices] + v1[modified_norms_indices])
    if raw_intgrd.is_complex(): 
        # print(f"raw_intgrd is complex. {raw_intgrd}")
        raise ValueError
    return X[:, 0]*raw_intgrd

# @nb.njit
# def jac_integrand0(r: float, theta: float, lambdas: list[float], v: list[float], demands_locations: list[list[float]]):
#     x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
#     xi, vi = categorize_x(x_cdnt, demands_locations, lambdas, v)
#     the_norm = norm_func(x_cdnt, xi)
#     return -r * the_norm/pow(v[0]*the_norm + vi, 2)

'''NOTE: i is the index of v, whose values range from 0 to n
      But j is the index of regions, whose values range from 1 to n.
      For indexes of demands_locations (0 to n-1), we need to offset j by subtracting one.'''

# @nb.njit
# def jac_integrandj(r: float, theta: float, lambdas: list[float], v: list[float], demands_locations: list[list[float]], j: int):
#     x_cdnt = np.array([r*np.cos(theta), r*np.sin(theta)])
#     if region_indicator(j-1, x_cdnt, lambdas, demands_locations) == 0: return 0
#     return -r / pow(v[0] * norm_func(x_cdnt, demands_locations[j-1]) + v[j], 2)


def objective_function(v: list[float], demands_locations: list[list[float]], lambdas: list[float], t: float, region_radius, thetarange) -> float:
    # print(f"DEBUG: v is {v, type(v[1])}.")
    # v = np.real(v)
    start, end = thetarange
    simpson = torchquad.Simpson()
    sum_integral = simpson.integrate(lambda X: integrand(X, lambdas, v, demands_locations), dim=2, N=1000000, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    # print(f"sum_integral is {sum_integral}. with type {type(sum_integral)}")
    return 1/4*sum_integral + v[0]*t + np.mean(v[1:])


# def objective_jac(v: list[float], demands_locations: list[list[float]], lambdas: list[float], t: float, region_radius, thetarange):
#     start, end = thetarange
#     n = demands_locations.shape[0]
#     jac = np.zeros(n + 1)
#     jac[0] = 1/4 * integrate.dblquad(jac_integrand0, start, end, lambda _: 0, lambda _: region_radius, args=(lambdas, v, demands_locations), epsabs=tol)[0] + t
#     for j in range(1, n+1):
#         jac[j] = 1/4 * integrate.dblquad(jac_integrandj, start, end, lambda _: 0, lambda _: region_radius, args=(lambdas, v, demands_locations, j), epsabs=tol)[0] + 1/n
#     return jac

def minimize_problem7(lambdas: list[float], demands: list[Demand], thetarange: list, t: float, region_radius) -> list[float]:
    # constraints = [optimize.NonlinearConstraint(lambda v: constraint_func(lambdas, demands, v, region_radius), 0, np.inf)]
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    bounds = optimize.Bounds(0, np.inf)
    result = optimize.minimize(objective_function, x0=np.append(1, np.ones(demands.shape[0])), args=(demands_locations, lambdas, t, region_radius, thetarange), jac='cs', method='SLSQP',  bounds=bounds)
    return result.x, result.fun



'''Pre-satisfied constraints when bounding'''
# def constraint_objective(x_cdnt, region_radius, v, demands, lambdas):
#     xi, vi = categorize_x(x_cdnt, demands, lambdas, v)
#     return v[0]*linalg.norm(x_cdnt - xi.get_cdnt()) + vi


# def constraint_func(lambdas, demands, v, region_radius):
#     objective = lambda x_cdnt: constraint_objective(x_cdnt, region_radius, v, demands, lambdas)
#     x_in_R_constraint = optimize.NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 0, region_radius)
#     result = optimize.minimize(objective, x0=np.zeros(2), method='SLSQP', constraints=x_in_R_constraint)
#     return result.fun


# x, fun_val = minimize_problem7(lambdas_temporary, demands, t_temporary, region)