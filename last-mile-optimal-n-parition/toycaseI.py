from typing import Literal
import numpy as np
import torch
import time
import torchquad
import numba as nb
from openpyxl import load_workbook
from scipy import optimize
from classes import Region, Coordinate, Polyhedron, Demand, append_df_to_csv

# Problem 7
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
def integrand7(X, lambdas, v, demands_locations):
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

def jac_integrand07(X, lambdas, v, demands_locations):
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

def jac_integrandj7(X, lambdas, v, demands_locations, j):
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

def objective_function7(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    simpson = torchquad.Simpson()

    time_before_problem7_obj_integral = time.time()
    sum_integral = simpson.integrate(lambda X: integrand7(X, lambdas, v, demands_locations), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    with open('./timerecords_plus/problem7_obj_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem7_obj_integral}\n')
    
    return 1/4*sum_integral + v[0]*t + np.mean(v[1:])

def objective_jac7(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    n = demands_locations.shape[0]
    jac = np.zeros(n + 1)
    simpson = torchquad.Simpson()

    time_before_problem7_jac_integral = time.time()
    jac[0] = 1/4 * simpson.integrate(lambda X: jac_integrand07(X, lambdas, v, demands_locations), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch').item() + t
    # The precompiled version is even slower, so we don't use it.
    # for j in torch.range(1, n): # notice torch.range is inclusive on both ends, hence we use n as upper bound
    #     if j == 1:
    #         simpsonj_compiled = torch.jit.trace(
    #             lambda vj_index: simpson.integrate(lambda X: jac_integrandj(X, lambdas, v, demands_locations, vj_index), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch'),
    #             (j,)
    #                 )
    #     jac[int(j.item())] = 1/4 * simpsonj_compiled(j).item() + 1/n


    for j in range(1, n+1):
        jac[j] = 1/4 * simpson.integrate(lambda X: jac_integrandj7(X, lambdas, v, demands_locations, j), dim=2, N=10000001, integration_domain=[[0, region_radius], [start, end]], backend='torch').item() + 1/n
    with open('./timerecords_plus/problem7_jac_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem7_jac_integral}\n')

    return jac


def minimize_problem7(lambdas, demands, thetarange, t, region_radius, tol):
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    bounds = optimize.Bounds(0, np.inf)
    result = optimize.minimize(objective_function7, x0=np.append(1e-6, np.ones(demands.shape[0])), args=(demands_locations, lambdas, t, region_radius, thetarange), jac=objective_jac7, method='SLSQP',  bounds=bounds, options={'ftol': tol, 'disp': True})
    return result.x, result.fun




# Problem 14

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
    with open('./timerecords_plus/problem14_obj_integral_time.txt', 'a') as f:
        f.write(f'{time.time() - time_before_problem14_obj_integral}\n')

    return area/4 + v[0]*t + v[1]

def objective_jac(v, demands_locations, lambdas, t, region_radius, thetarange):
    start, end = thetarange
    simpson = torchquad.Simpson()

    time_before_problem14_jac_integral = time.time()
    area0 = simpson.integrate(lambda X: jac_integrand0(X, v, demands_locations, lambdas), dim=2, N=999999, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    area1 = simpson.integrate(lambda X: jac_integrand1(X, v, demands_locations, lambdas), dim=2, N=999999, integration_domain=[[0, region_radius], [start, end]], backend='torch').item()
    with open('./timerecords_plus/problem14_jac_integral_time.txt', 'a') as f:
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
    lambdas_bar = np.zeros(n)
    polyhedron = Polyhedron(np.eye(n), region.diam*np.ones(n), np.ones((1, n)), 0, n)
    k = 1
    while (abs(UB - LB) > epsilon and k < 10):
        print(f'Looking for worst-distribution on {[start, end]}:\nIteration {k} begins: \n')
        starttime = time.time()
        lambdas_bar, lambdas_bar_func_val = polyhedron.find_analytic_center(lambdas_bar)
        print(f'Find analytic center: Lambdas_bar is {lambdas_bar}, with value {lambdas_bar_func_val}, took {time.time() - starttime}s.\n')
        with open("./timerecords_plus/find_analytic_center.txt", "a") as f:
            f.write(f"{time.time() - starttime}\n")

        demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])

        '''Build an upper bounding f_bar for the original problem (4).'''
        time_before_minimize_problem14 = time.time()
        v_bar, problem14_func_val = minimize_problem14(demands, thetarange, lambdas_bar, t, region.radius)
        with open("./timerecords_plus/minimize_problem14.txt", "a") as f:
            f.write(f"{time.time() - time_before_minimize_problem14}\n")

        time_before_ub_integral = time.time()
        upper_integrand = lambda X: X[:, 0]*torch.sqrt(f_bar(X, demands_locations, lambdas_bar, v_bar))
        UB = simpson.integrate(upper_integrand, dim=2, N=999999, integration_domain=[[0, region.radius],[start, end]], backend='torch').item()
        with open("./timerecords_plus/ub_integral.txt", "a") as f:
            f.write(f"{time.time() - time_before_ub_integral}\n")

        print(f'UB is {UB}, took {time.time() - starttime}s.\n')

        if UB < 0:
            print(f'UB is negative: {UB}.')
            print(f'v_bar is {v_bar}, problem14_func_val is {problem14_func_val}.')
            break

        '''Build an lower bounding f_tilde that us feasible for (4) by construction.'''
        time_before_minimize_problem7 = time.time()
        v_tilde, problem7_func_val = minimize_problem7(lambdas_bar, demands, thetarange, t, region.radius, tol)
        with open("./timerecords_plus/minimize_problem7.txt", "a") as f:
            f.write(f"{time.time() - time_before_minimize_problem7}\n")

        time_before_lb_integral = time.time()
        lower_integrand = lambda X: X[:, 0]*torch.sqrt(f_tilde(X, demands_locations, lambdas_bar, v_tilde))
        LB =simpson.integrate(lower_integrand, dim=2, N=999999, integration_domain=[[0, region.radius],[start, end]], backend='torch').item()
        with open("./timerecords_plus/lb_integral.txt", "a") as f:
            f.write(f"{time.time() - time_before_lb_integral}\n")
        
        print(f'LB is {LB}, took {time.time() - starttime}s.\n')

        '''Update g.'''
        g = np.zeros(len(demands))
        time_before_g_integral = time.time()
        for i in range(len(demands)):
            integrandi = lambda X: X[:, 0]*f_bar(X, demands_locations, lambdas_bar, v_bar, masked=True, j=i+1) 
            g[i] = simpson.integrate(integrandi, dim=2, N=999999, integration_domain=[[0, region.radius], [start, end]], backend='torch').item()
        with open("./timerecords_plus/g_integral.txt", "a") as f:
            f.write(f"{time.time() - time_before_g_integral}\n")
        
        '''Update polyheron Lambda to get next analytic center.'''
        polyhedron.add_ineq_constraint(g, g.T @ lambdas_bar)

        print(f'End of iteration {k}.\n  The whole iteration took {time.time() - starttime}s.\n')
        k += 1

    return lambda X: f_tilde(X, demands_locations, lambdas_bar, v_tilde)


@nb.jit(nopython=True)
def norm_func(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

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


def main():
    region = Region(1)
    depot = Coordinate(2, 0.3)
    t, epsilon = 0.25, 0.1

    thetarange = (3.147444264116321 - 2*np.pi, 0.00012460881117814946)
    demands = np.array([Demand(Coordinate(r = 0.1802696888767692, theta = 4.768809396779301), 1),
            Demand(Coordinate(r = 0.019475241487624584, theta = 5.1413757057591045), 1),
            Demand(Coordinate(r = 0.012780814590608647, theta = 4.478189127165961), 1),
            Demand(Coordinate(r = 0.4873716073198716, theta = 3.7670422583826495), 1),
            Demand(Coordinate(r = 0.10873607186897494, theta = 5.328009178184025), 1),
            Demand(Coordinate(r = 0.8939041702850812, theta = 4.5103794163992506), 1),
            Demand(Coordinate(r = 0.8571542470728296, theta = 3.7828800007876318), 1),
            Demand(Coordinate(r = 0.16508661759522714, theta = 3.4707299112070515), 1),
            Demand(Coordinate(r = 0.6323340138234961, theta = 5.963386240530588), 1),
            Demand(Coordinate(r = 0.020483612791232675, theta = 6.19945137229428), 1),
            Demand(Coordinate(r = 0.15791230667493694, theta = 5.004153427569559), 1)])
    demands_locations = np.array([demands[i].get_cdnt() for i in range(len(demands))])
    lambdas = np.zeros(demands.shape[0])

    optimal_ftilde = findWorstTSPDensity(region, demands, thetarange, t, epsilon, tol=1e-4)

if __name__ == '__main__':
    main()