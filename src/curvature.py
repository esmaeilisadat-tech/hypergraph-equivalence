import numpy as np
from fractions import Fraction
from scipy.optimize import linprog
from collections import deque

def get_neighbors(H_edges, x):
    n = set()
    for e in H_edges:
        if x in e:
            for z in e:
                if z != x:
                    n.add(z)
    return n

def compute_shortest_path_matrix(H_edges, num_nodes):
    # Construct adjacency list for clique expansion
    adj = {i: set() for i in range(num_nodes)}
    for e in H_edges:
        for u in e:
            for v in e:
                if u != v:
                    adj[u].add(v)
    
    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0)
    
    for i in range(num_nodes):
        q = deque([(i, 0)])
        visited = {i}
        while q:
            curr, d = q.popleft()
            dist[i, curr] = d
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))
    return dist

def en_measure(H_edges, x):
    """
    EN random walk: Uniform over distinct neighbors.
    mu_x^EN(z) = 1 / |N(x)| for z in N(x), 0 otherwise.
    """
    neighbors = get_neighbors(H_edges, x)
    if not neighbors:
        return {}
    val = Fraction(1, len(neighbors))
    return {z: val for z in neighbors}

def ee_measure(H_edges, x):
    """
    EE random walk: 
    mu_x^EE(z) = 1 / d_x * sum_{e containing x and z} 1 / (|e| - 1)
    """
    dx = sum(1 for e in H_edges if x in e)
    if dx == 0:
        return {}
    
    mu = {}
    for e in H_edges:
        if x in e:
            size = len(e)
            if size > 1:
                weight = Fraction(1, size - 1)
                for z in e:
                    if z != x:
                        if z not in mu:
                            mu[z] = Fraction(0)
                        mu[z] += weight / dx
    return mu

def wasserstein_distance(mu_x, mu_y, dist_matrix):
    """
    Computes Earth Mover's Distance between two probability measures mu_x and mu_y 
    using scipy.optimize.linprog, restricted to their supports.
    """
    if not mu_x and not mu_y:
        return 0.0
    if not mu_x or not mu_y:
        return np.inf
        
    supp_x = list(mu_x.keys())
    supp_y = list(mu_y.keys())
    
    # Flatten variables xi_ij for i in supp_x, j in supp_y
    # len(supp_x) * len(supp_y) variables
    num_x = len(supp_x)
    num_y = len(supp_y)
    
    c = np.zeros(num_x * num_y)
    for i, u in enumerate(supp_x):
        for j, v in enumerate(supp_y):
            c[i * num_y + j] = dist_matrix[u, v]
            if np.isinf(c[i * num_y + j]):
                # If path is infinity, this should be heavily penalized or distance is inf
                c[i * num_y + j] = 1e6 # large penalty

    # A_eq * x = b_eq
    # Rows for mu_x marginals: num_x constraints
    # Rows for mu_y marginals: num_y constraints
    A_eq = np.zeros((num_x + num_y, num_x * num_y))
    b_eq = np.zeros(num_x + num_y)
    
    # sum_j xi_ij = mu_x(u_i)
    for i in range(num_x):
        for j in range(num_y):
            A_eq[i, i * num_y + j] = 1.0
        b_eq[i] = float(mu_x[supp_x[i]])
        
    # sum_i xi_ij = mu_y(v_j)
    for j in range(num_y):
        for i in range(num_x):
            A_eq[num_x + j, i * num_y + j] = 1.0
        b_eq[num_x + j] = float(mu_y[supp_y[j]])
        
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    if res.success:
        return res.fun
    else:
        return np.inf

def compute_orc(H_edges, num_nodes, x, y, measure_type='EE'):
    """
    Computes Ollivier-Ricci Curvature between x and y.
    measure_type: 'EN' or 'EE'
    """
    dist_matrix = compute_shortest_path_matrix(H_edges, num_nodes)
    d_xy = dist_matrix[x, y]
    if np.isinf(d_xy) or d_xy == 0:
        return 0.0 # Curvature undefined or 0
        
    if measure_type == 'EN':
        mu_x = en_measure(H_edges, x)
        mu_y = en_measure(H_edges, y)
    else:
        mu_x = ee_measure(H_edges, x)
        mu_y = ee_measure(H_edges, y)
        
    W1 = wasserstein_distance(mu_x, mu_y, dist_matrix)
    if np.isinf(W1):
        return -np.inf
        
    kappa = 1.0 - (W1 / d_xy)
    return kappa
