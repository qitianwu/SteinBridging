import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal
import math
from utlis.input import DataInput
from utlis.plot_figure import *
from utlis.data_synthetic import *
from utlis.eval import *

from datetime import datetime
import os
import time
import logging
import pickle


flags = tf.app.flags
flags.DEFINE_string("gpus", "-1", "gpus")
FLAGS = flags.FLAGS

print(FLAGS.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus

# model parameter
h_dim_g = 128  # number of hidden neurons per layer of generator
h_dim_d = 128  # number of hidden neurons per layer of discriminator
h_dim_p = 128 # number of hidden neurons per layer of density estimator
z_dim = 4  # noise dimension
X_dim = 2 # sample dimension
exp_num = 4 # expert num in energy model
mb_size = 100 # mini-batch size

show_size = 2000 # num of generated samples to test

kernel = "rbf"  # kernel for KSD, ["rbf", "imq"]

datadir = '/home/wuiron/SteinBridging/data/' # add your data directory 
workdir = os.getcwd()
method = 'Joint_JS'
dataset = 'Two_Circle'

# generate synthetic data
with open(datadir+dataset+'.pkl', 'rb') as f:
	true_sample = pickle.load(f)
	true_density = pickle.load(f)
	mesh_t_density = pickle.load(f)
	XY_range = pickle.load(f)

#----------------------------- network model ------------------------------------
tf.reset_default_graph()
 
initializer = tf.contrib.layers.xavier_initializer()

X_true = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
Mesh = tf.placeholder(tf.float32, shape=[None, X_dim])
isTraining = tf.placeholder(tf.bool, shape=None)

def sample_z(m, n, std=10.):
    s1 = np.random.normal(0, std, size=[m, n])
    
    return s1


def generator(z, isTraining):
    with tf.variable_scope("g", reuse=tf.AUTO_REUSE):
        o1 = tf.layers.dense(z, h_dim_g, activation=tf.nn.tanh, name='l1')
        o2 = tf.layers.dense(o1, h_dim_g, activation=tf.nn.tanh, name='l2')
        o3 = tf.layers.dense(o2, X_dim, activation=None, name='l3')
        out = o3

    return out #[mb, X_dim]

def p_theta(x):
    with tf.variable_scope("p", reuse=tf.AUTO_REUSE):
        o1 = tf.layers.dense(x, h_dim_p, activation=tf.nn.tanh, name='l1')
        o2 = tf.layers.dense(o1, h_dim_p, activation=tf.nn.tanh, name='l2')
        o3 = tf.layers.dense(o2, exp_num, activation=tf.nn.sigmoid, name='l3')
        o4 = tf.reduce_sum(o3, axis=-1)
        E = tf.reshape(o4, [tf.shape(x)[0]])
    out = tf.reshape(-E, [tf.shape(x)[0], 1])

    return out  #[mb, 1]

def discriminator(x):
    with tf.variable_scope("d", reuse=tf.AUTO_REUSE):
        o1 = tf.layers.dense(x, h_dim_d, activation=tf.nn.tanh, name='l1')
        o2 = tf.layers.dense(o1, h_dim_d, activation=tf.nn.tanh, name='l2')
        out = tf.layers.dense(o2, 1, activation=tf.nn.sigmoid, name='l3')

    return out #[mb, 1]

def S_q(xs):
    return tf.gradients(p_theta(xs), xs)[0] #[mb, X_dim]

def rbf_kernel(x, dim=X_dim, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = tf.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), 1)
    dxkxy = tf.add(-tf.matmul(kxy, x), tf.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    dxykxy_tr = tf.multiply((dim * (h**2) - pdist), kxy) / (h**4)  # tr( dk(x, y)/dxdy )

    return kxy, dxkxy, dxykxy_tr


def imq_kernel(x, dim=X_dim, beta=-.5, c=1.):
    # IMQ kernel
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = (c + pdist) ** beta

    coeff = 2 * beta * (c + pdist) ** (beta-1)
    dxkxy = tf.matmul(coeff, x) - tf.multiply(x, tf.expand_dims(tf.reduce_sum(coeff, axis=1), 1))

    dxykxy_tr = tf.multiply((c + pdist) ** (beta - 2),
                            - 2 * dim * c * beta + (- 4 * beta ** 2 + (4 - 2 * dim) * beta) * pdist)

    return kxy, dxkxy, dxykxy_tr


kernels = {"rbf": rbf_kernel,
           "imq": imq_kernel}

Kernel = kernels[kernel]


def ksd_emp(x, dim=X_dim):
    sq = S_q(x)
    kxy, dxkxy, dxykxy_tr = Kernel(x, dim)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    n = tf.cast(tf.shape(x)[0], tf.float32)

    # ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
    ksd = (tf.reduce_sum(t13) + t2) / (n ** 2)

    return ksd

G_sample = generator(z, isTraining)

Density = tf.exp(p_theta(Mesh))


#######################################################################################################################

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, save_path=workdir+'/model/'+method)


fake_sample = sess.run(G_sample, feed_dict={z: sample_z(show_size, z_dim),
                                                isTraining: False})
fake_density = sess.run(Density, 
            feed_dict={Mesh: fake_sample, isTraining: False})

X_range, Y_range = XY_range[0], XY_range[1]
mesh_input = np.stack([np.reshape(X_range, [-1]), np.reshape(Y_range, [-1])], 1)
meshinput = DataInput(data=mesh_input, batch_size=mb_size)
mesh_f_density = np.zeros([mesh_input.shape[0], 1])
for i in range(mesh_input.shape[0]//mb_size):
    mesh_f_density[i*mb_size:min((i+1)*mb_size, mesh_input.shape[0])] = sess.run(Density, 
    feed_dict={Mesh: meshinput.get_batch(), isTraining: False})
mesh_f_density = np.reshape(mesh_f_density, [X_range.shape[0], X_range.shape[1]])

norm = np.sum(mesh_t_density) 
mesh_f_density_n = mesh_f_density / np.sum(mesh_f_density) * norm

#plot_sample_density(true_sample, mesh_t_density, (X_range, Y_range), fake_sample, mesh_f_density_n, \
# method=method, save_path=workdir+'/vis/vis.pdf')

mmd = mmd_eval(true_sample, fake_sample, 'rbf')
ce_y, ent_yx, high_rate = ce_y_eval(fake_sample, 'Two_Circle')

kl_div = kl_div_eval(mesh_t_density, mesh_f_density_n, 1/norm)

js_div = js_div_eval(mesh_t_density, mesh_f_density_n, 1/norm)


radius1=4.
n_comp1=8
radius2=8.
n_comp2=16
sd=0.2

mu1 = np.array([[radius1 * math.cos((i*2*math.pi)/n_comp1), radius1 * math.sin((i*2*math.pi)/n_comp1)] for i in range(n_comp1)])
mu2 = np.array([[radius2 * math.cos((i*2*math.pi)/n_comp2), radius2 * math.sin((i*2*math.pi)/n_comp2)] for i in range(n_comp2)])
center = np.vstack([mu1, mu2])
density = sess.run(Density, feed_dict={Mesh: center, isTraining: False})
t = 0.1*np.max(mesh_f_density_n)
density_n = density / np.sum(mesh_f_density) * norm
mode_cover = np.sum(density_n>t) / density.shape[0]

n_s = []
for i in range(center.shape[0]):
    n_s_i = [center[i]+(2*np.random.rand(X_dim)-1)*5*sd for j in range(10)]
    n_s.append(n_s_i)
n_s = np.array(n_s).reshape([-1, X_dim])
density2 = sess.run(Density, feed_dict={Mesh: n_s, isTraining: False})
auc = auc_calc(density, density2)


mmd_f = '{:.5f}'.format(mmd)
ce_y_f = '{:.4f}'.format(ce_y)
ent_yx_f = '{:.4f}'.format(ent_yx)
high_rate_f = '{:.4f}'.format(high_rate)
kl_div_f = '{:.4f}'.format(kl_div)
js_div_f = '{:.4f}'.format(js_div)
mode_cover_f = '{:.4f}'.format(mode_cover)
auc_f = '{:.4f}'.format(auc)

output = "MMD {:<7} CE_Y {:<6} ENT_YX {:<6} Rate {:<6} KL {:<6} JS {:<6} Cover {:<6} AUC {:<6}".\
    format(mmd_f, ce_y_f, ent_yx_f, high_rate_f, kl_div_f, js_div_f, mode_cover_f, auc_f)
print(output)