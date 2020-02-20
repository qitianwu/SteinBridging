import numpy as np
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt
from utlis.plot_figure import *
from sklearn.mixture import GMM
import pickle

def create_meshgrid(grid_width, data):
    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    mid_x = (min_x + max_x) / 2.

    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])
    mid_y = (min_y + max_y) / 2.

    min_x += 0.5 * (min_x - mid_x)
    min_y += 0.5 * (min_y - mid_y)
                   
    max_x += 0.5 * (max_x - mid_x)
    max_y += 0.5 * (max_y - mid_y)

    axis = [min_x, max_x, min_y, max_y]

    x = np.linspace(min_x, max_x, grid_width)
    y = np.linspace(min_y, max_y, grid_width)
    dx, dy = np.meshgrid(x, y)
    data = np.concatenate([dx.reshape(-1, 1), dy.reshape(-1, 1)], axis=1)

    return dx, dy, data, axis

def Two_Circle(N=2000, X_dim=2, radius1=4., n_comp1=8, radius2=8., n_comp2=16, \
                            sd=0.2, p_weight=None, is_noise=False, n_ratio=0.05):
    '''
    input
    # N: sample size
    # X_dim: dimension of the target distribution
    # radius1: radius of the inner circle where modes on (equally spaced)
    # n_comp1: number of mixture component (number of modes) on inner circle
    # radius2: radius of the outer circle where modes on (equally spaced)
    # n_comp2: number of mixture component (number of modes) on outer circle
    # sd: variance of each component
    # p_weight: weight for each component. Use uniform weights if None
    # is_noise: whether add noise
    # n_ratio: ratio of noise samples

    output
    # true_sample: groung-truth samples from true distribution
    # noise_sample: noise samples
    # sample_density: density values for true_sample
    # mesh_density: density values for each point on mesh grid
    # (X_range, Y_range): X values and Y values on mesh grid
    '''

    # ground-truth distribution
    mu1 = np.array([[radius1 * math.cos((i*2*math.pi)/n_comp1), radius1 * math.sin((i*2*math.pi)/n_comp1)] for i in range(n_comp1)])
    Sigma1 = np.tile(np.array([[sd, 0], [0, sd]]), [n_comp1, 1]).reshape(n_comp1, X_dim, X_dim)
    mu2 = np.array([[radius2 * math.cos((i*2*math.pi)/n_comp2), radius2 * math.sin((i*2*math.pi)/n_comp2)] for i in range(n_comp2)])
    Sigma2 = np.tile(np.array([[sd, 0], [0, sd]]), [n_comp2, 1]).reshape(n_comp2, X_dim, X_dim)
    mu = np.vstack([mu1, mu2])
    Sigma = np.vstack([Sigma1, Sigma2])
    n_comp = n_comp1 + n_comp2
    if p_weight is None:
        p = np.ones(n_comp) / n_comp  # equal weights
    else:
        p = p_weight

    # true sample
    label = np.random.choice(n_comp, size=N, p=p)[:, np.newaxis]
    true_sample = np.sum(np.stack([np.random.multivariate_normal(mu[i], Sigma[i], N) * (label == i) for i in range(n_comp)]), 0)
    
    # noise sample
    N2 = int(N*n_ratio)
    label2 = np.random.choice(n_comp, size=N2, p=p)[:, np.newaxis]
    noise_sample = np.sum(np.stack([np.random.multivariate_normal(mu[i], Sigma[i], N2) * (label2 == i) for i in range(n_comp)]), 0) + np.random.normal(0, 1,[N2, 2])*2

    # true density
    sample_density = sum([p[i] * multivariate_normal.pdf(true_sample,
                mu[i], Sigma[i]) for i in range(n_comp)])


    # grids & mesh density
    grid_width = 300
    X_range, Y_range, mesh, axis = create_meshgrid(grid_width, true_sample)
    mesh_density = sum([p[i] * multivariate_normal.pdf(np.stack((X_range, Y_range), 2),
                mu[i], Sigma[i]) for i in range(n_comp)])
    
    # visualize toy example
    plot_sample_density(true_sample, mesh_density, (X_range, Y_range))
    
    if is_noise is True:
        return true_sample, noise_sample, sample_density, mesh_density, (X_range, Y_range)
    else:
        return true_sample, sample_density, mesh_density, (X_range, Y_range)

def Two_Spiral(N=5000, NC=100, start=120, sd = 0.5):
    '''
    input
    # N: sample size
    # NC: number of component in total
    # start: start degree of the first component
    # sd: variance of each component

    output
    # true_sample: groung-truth samples from true distribution
    # sample_density: density values for true_sample
    # mesh_density: density values for each point on mesh grid
    # (X_range, Y_range): X values and Y values on mesh grid
    '''

    degrees = 360.0
    deg2rad = (2 * np.pi) / 360.0
    start = start * deg2rad
    circle_ratio = 0.5

    num_per_comp = int((N + NC - 1) / NC)

    NC1 = int(NC / 2.0)
    NC2 = NC - NC1

    # ground-truth distribution
    Sigma = np.array([[sd, 0], [0, sd]])
    c1 = start + np.linspace(0., circle_ratio , NC1, endpoint=True) * degrees * deg2rad
    x1 = -np.cos(c1) * c1
    y1 =  np.sin(c1) * c1
    mu1 = np.stack([x1, y1], 1)
    c2 = start + np.linspace(0., circle_ratio , NC2, endpoint=True) * degrees * deg2rad
    x2 =  np.cos(c2) * c2
    y2 = -np.sin(c2) * c2
    mu2 = np.stack([x2, y2], 1)
    
    # true sample
    d1 = [np.random.multivariate_normal(mu1[i], Sigma, num_per_comp) for i in range(NC1)]
    d2 = [np.random.multivariate_normal(mu2[i], Sigma, num_per_comp) for i in range(NC2)]
    d1 = np.array(d1).reshape([NC1*num_per_comp, 2])
    d2 = np.array(d2).reshape([NC2*num_per_comp, 2])
    true_sample = np.concatenate((d1, d2), axis=0)[:N]

    # true density
    sample_density = np.zeros(true_sample.shape[0])
    eye2 = np.eye(2)*sd
    for offset in np.linspace(0., circle_ratio , NC1, endpoint=True):
        n = start + offset * degrees * deg2rad
        x = -np.cos(n) * n
        y =  np.sin(n) * n
        sample_density += 1/NC * multivariate_normal.pdf(true_sample, 
        np.array([x,y]), eye2)
    for offset in np.linspace(0., circle_ratio , NC2, endpoint=True):
        n = start + offset * degrees * deg2rad
        x =  np.cos(n) * n
        y = -np.sin(n) * n
        sample_density += 1/NC * multivariate_normal.pdf(true_sample, 
        np.array([x,y]), eye2)

    # grids & mesh density
    grid_width = 300
    X_range, Y_range, mesh, axis = create_meshgrid(grid_width, true_sample)
    mesh_density = np.zeros([X_range.shape[0], X_range.shape[1]])
    for offset in np.linspace(0., circle_ratio , NC1, endpoint=True):
        n = start + offset * degrees * deg2rad
        x = -np.cos(n) * n
        y =  np.sin(n) * n
        mesh_density += 1/NC * multivariate_normal.pdf(np.stack((X_range, Y_range), 2), 
        np.array([x,y]), eye2)
    for offset in np.linspace(0., circle_ratio , NC2, endpoint=True):
        n = start + offset * degrees * deg2rad
        x =  np.cos(n) * n
        y = -np.sin(n) * n
        mesh_density += 1/NC * multivariate_normal.pdf(np.stack((X_range, Y_range), 2), 
        np.array([x,y]), eye2)
    
    # visualize toy example
    plot_sample_density(true_sample, mesh_density, (X_range, Y_range))
    
    return true_sample, sample_density, mesh_density, (X_range, Y_range)

if __name__ == "__main__":

    datadir = '' # add your directory for datasets

    # Two-Spiral
    true_sample, true_density, mesh_t_density, XY_range = \
        Two_Spiral(N=5000, NC=100, start=120, sd = 0.1)

    with open(datadir+'Two_Spiral.pkl', 'wb') as f:
	    pickle.dump(true_sample, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(true_density, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(mesh_t_density, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(XY_range, f, pickle.HIGHEST_PROTOCOL)


    '''
    # Two-Circle
    true_sample, true_density, mesh_t_density, XY_range = \
        Two_Circle(N=2000, X_dim=2, radius1=4., n_comp1=8, radius2=8., n_comp2=16, sd=0.2)
    # if with noise
    # true_sample, noise_sample, true_density, mesh_t_density, XY_range = \
	#    Two_Circle(N=2000, X_dim=2, radius1=4., n_comp1=8, radius2=8., \
	#	 n_comp2=16, sd=0.2, is_noise=True, n_rate=0.25)
    # total_sample = np.vstack((true_sample, noise_sample))

    with open(datadir+'Two_Circle.pkl', 'wb') as f:
	    pickle.dump(true_sample, f, pickle.HIGHEST_PROTOCOL)
	    #pickle.dump(noise_sample, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(true_density, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(mesh_t_density, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(XY_range, f, pickle.HIGHEST_PROTOCOL)
    '''