import numpy as np
from scipy.stats import multivariate_normal
import math

def mmd_eval(x, y, kernel='rbf'):

    if kernel=='rbf':
        def rbf(x, y, h=1.):
            # x, y of shape (n, d)
            xy = np.matmul(x, y.T)
            x2 = np.sum(x ** 2, 1).reshape(-1, 1)
            y2 = np.sum(y ** 2, 1).reshape(1, -1)
            pdist = (x2 + y2) - 2 * xy
            return np.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

        nx, ny = x.shape[0], y.shape[0]
        kxx = rbf(x, x)
        kxy = rbf(x, y)
        kyy = rbf(y, y)
        return np.sum(kxx) / nx / (nx - 1) + np.sum(kyy) / ny / (ny - 1) - 2 * np.sum(kxy) / nx / ny

    if kernel=='imq':
        def imq(x, y, h=1.):
            # x, y of shape (n, d)
            c = 1
            beta = 0.5
            xy = np.matmul(x, y.T)
            x2 = np.sum(x ** 2, 1).reshape(-1, 1)
            y2 = np.sum(y ** 2, 1).reshape(1, -1)
            pdist = (x2 + y2) - 2 * xy
            return (c + pdist)**beta  # kernel matrix

        nx, ny = x.shape[0], y.shape[0]
        kxx = imq(x, x)
        kxy = imq(x, y)
        kyy = imq(y, y) 
        return np.abs(np.sum(kxx) / nx / (nx - 1) + np.sum(kyy) / ny / (ny - 1) - 2 * np.sum(kxy) / nx / ny)

def kl_div_eval(p, q, scale):
    p = np.clip(p, 1e-7, 1e7)
    q = np.clip(q, 1e-7, 1e7)
    return np.sum(p * np.log(p) - p * np.log(q)) * scale

def js_div_eval(p, q, scale):
    p = np.clip(p, 1e-7, 1e7)
    q = np.clip(q, 1e-7, 1e7)
    t = (p + q) / 2
    return 1/2*np.sum(kl_div_eval(p, t, 1) + kl_div_eval(q, t, 1)) * scale

def ce_y_eval(x, dataset):
    if dataset=='Two_Circle':
        radius1=4.
        n_comp1=8
        radius2=8.
        n_comp2=16
        sd=0.2

        mu1 = np.array([[radius1 * math.cos((i*2*math.pi)/n_comp1), radius1 * math.sin((i*2*math.pi)/n_comp1)] for i in range(n_comp1)])
        mu2 = np.array([[radius2 * math.cos((i*2*math.pi)/n_comp2), radius2 * math.sin((i*2*math.pi)/n_comp2)] for i in range(n_comp2)])
        mu = np.vstack([mu1, mu2])
        n_comp = n_comp1 + n_comp2
        y_dis = np.stack([np.sum((x-mu[i])**2, -1) for i in range(n_comp)], 1)
        y_prob_est = np.exp(-y_dis)
        y_prob_est = y_prob_est / np.tile(np.expand_dims(np.sum(y_prob_est, 1), 1), [1, n_comp])
        y_prob = np.mean(y_prob_est, 0)
        y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
        ce_y = -np.mean(y_prob*np.log(y_prob))

        high_rate = np.sum(np.max((y_dis<25*sd**2), -1)) / x.shape[0]

        y_prob_est = np.clip(y_prob_est, 1e-7, 1e7)
        ent_yx = -np.mean(np.sum(y_prob_est*np.log(y_prob_est), -1))

    if dataset=='Two_Spiral':
        NC = 100
        start = 120
        sd = 0.5

        degrees = 360.0
        deg2rad = (2 * np.pi) / 360.0
        start = start * deg2rad
        circle_ratio = 0.5

        NC1 = int(NC / 2.0)
        NC2 = NC - NC1

        c1 = start + np.linspace(0., circle_ratio , NC1, endpoint=True) * degrees * deg2rad
        x1 = -np.cos(c1) * c1
        y1 =  np.sin(c1) * c1
        mu1 = np.stack([x1, y1], 1)
        c2 = start + np.linspace(0., circle_ratio , NC2, endpoint=True) * degrees * deg2rad
        x2 =  np.cos(c2) * c2
        y2 = -np.sin(c2) * c2
        mu2 = np.stack([x2, y2], 1)
        mu = np.vstack([mu1, mu2])

        y_dis = np.stack([np.sum((x-mu[i])**2, -1) for i in range(NC)], 1)
        y_prob_est = np.exp(-y_dis)
        y_prob_est = y_prob_est / np.tile(np.expand_dims(np.sum(y_prob_est, 1), 1), [1, NC])
        y_prob = np.mean(y_prob_est, 0)
        y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
        ce_y = -np.mean(y_prob*np.log(y_prob))

        high_rate = np.sum(np.max((y_dis<sd**2), -1)) / x.shape[0]

        y_prob_est = np.clip(y_prob_est, 1e-7, 1e7)
        ent_yx = -np.mean(np.sum(y_prob_est*np.log(y_prob_est), -1))
        
    return ce_y, ent_yx, high_rate

def auc_calc(p_s, n_s):
    score_label = []
    for i in range(len(p_s)):
        score_label.append([p_s[i], 1])
    for i in range(len(n_s)):
        score_label.append([n_s[i], 0])
    score_label_ = sorted(score_label, key=lambda d:d[0], reverse=True)
    fp1, tp1, fp2, tp2, auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for s in score_label_:
        fp2 += (1-s[1])
        tp2 += s[1]
        auc += (tp2 - tp1) * (fp2 + fp1) / 2
        fp1, tp1 = fp2, tp2
    try:
        return 1- auc / (tp2 * fp2)
    except:
        return 0.5