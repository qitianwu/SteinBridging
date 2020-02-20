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
lr_d = 1e-5  # learning rate for discriminator
lr_p = 1e-3 # learning rate for density estimator
lr_g = 1e-5 # learning rate for sample generator
h_dim_g = 128  # number of hidden neurons per layer of generator
h_dim_d = 128  # number of hidden neurons per layer of discriminator
h_dim_p = 128 # number of hidden neurons per layer of density estimator
z_dim = 4  # noise dimension
X_dim = 2 # sample dimension
exp_num = 4 # expert num in energy model
n_critic = 5 # epoches for discriminator before update for generator in GAN

mb_size = 100  # mini-batch size
n_iter = 1000000 # total interations for training
iter_display = 2000 # interations for one display
show_size = 2000 # num of generated samples to test
load_it = 0 # load the model trained after load_it interations

optimizer_KSD = tf.train.RMSPropOptimizer # optimizer for two KSD losses
optimizer_GAN = tf.train.AdamOptimizer # optimizer for JS loss

kernel = "rbf"  # kernel for KSD, ["rbf", "imq"]
 
datadir = '/home/wuiron/SteinBridging/data/' # add your data directory 
method = 'Joint_JS'
dataset = 'Two_Circle'

# generate synthetic data
with open(datadir+dataset+'.pkl', 'rb') as f:
	true_sample = pickle.load(f)
	true_density = pickle.load(f)
	mesh_t_density = pickle.load(f)
	XY_range = pickle.load(f)

# make output directory
workdir = os.getcwd()
os.system("mkdir "+method)
os.chdir(workdir+'/'+method)
workdir = os.getcwd()
os.system("mkdir vis")
os.system("mkdir model")
logging.basicConfig(level=logging.INFO, filename=method+'.log', format='%(message)s')

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
    out = tf.reshape(tf.exp(-E), [tf.shape(x)[0], 1])

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

    ksd = (tf.reduce_sum(t13) + t2) / (n ** 2)

    return ksd


G_sample = generator(z, isTraining)
D_loss = - tf.reduce_mean(tf.log(1-discriminator(G_sample)) + tf.log(discriminator(X_true)))
G_loss = - tf.reduce_mean(tf.log(discriminator(G_sample)))

Density = p_theta(Mesh)
KSD_gp = ksd_emp(G_sample)
KSD_rp = ksd_emp(X_true)

para_D = tf.trainable_variables('d')
para_G = tf.trainable_variables('g')
para_P = tf.trainable_variables('p')

solver_GAN_D = optimizer_GAN(learning_rate=lr_g).minimize(D_loss, var_list=para_D)

solver_GAN_G = optimizer_GAN(learning_rate=lr_g).minimize(G_loss, var_list=para_G)

solver_KSD_rp = optimizer_KSD(learning_rate=lr_p).minimize(KSD_rp, var_list=para_P)

solver_KSD_gp_g = optimizer_KSD(learning_rate=lr_g).minimize(KSD_gp, var_list=para_G)

solver_KSD_gp_p = optimizer_KSD(learning_rate=lr_p).minimize(KSD_gp, var_list=para_P)



#----------------------------- training ------------------------------------

sess = tf.Session()
if load_it==0:
    sess.run(tf.global_variables_initializer())
else:  
    saver = tf.train.Saver()
    saver.restore(sess, save_path=workdir+'/model/{}'.format(load_it))

best_mmd, best_kl, best_rate = 10.0, 10.0, 0.
datainput = DataInput(data=true_sample, batch_size=mb_size)
start_time = datetime.now()
iterations = 0

for it in range(load_it, load_it+n_iter+1):

    i_critic = 0

    while i_critic<=n_critic:
        _, loss_d = sess.run([solver_GAN_D, D_loss],
                        feed_dict={z: sample_z(mb_size, z_dim), 
                                X_true: datainput.get_batch(),
                                isTraining: True})
        i_critic += 1
    
    _, loss_g = sess.run([solver_GAN_G, G_loss],
                            feed_dict={z: sample_z(mb_size, z_dim),
                                    isTraining: True})
    _, ksd_rp = sess.run([solver_KSD_rp, KSD_rp],
                            feed_dict={X_true: datainput.get_batch()})
    _, ksd_gp = sess.run([solver_KSD_gp_g, KSD_gp],
                           feed_dict={z: sample_z(mb_size, z_dim)})
    _, ksd_gp = sess.run([solver_KSD_gp_p, KSD_gp],
                           feed_dict={z: sample_z(mb_size, z_dim)})

    iterations += 1
        

    if it % iter_display == 0:
        fake_sample = sess.run(G_sample, feed_dict={z: sample_z(show_size, z_dim),
                                                isTraining: False})
        fake_density = sess.run(Density, 
            feed_dict={Mesh: fake_sample, isTraining: False})
        mmd = mmd_eval(true_sample, fake_sample, 'rbf')
        
        ce_y, ent_yx, high_rate = ce_y_eval(fake_sample, 'Two_Circle')

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
        
        kl_div = kl_div_eval(mesh_t_density, mesh_f_density_n, 1/norm)
        

        if mmd < best_mmd:
            best_mmd = mmd
        if kl_div < best_kl:
            best_kl = kl_div
        if high_rate > best_rate:
            best_rate = high_rate

        if it % (iter_display*10) == 0:
            saver = tf.train.Saver()
            saver.save(sess, save_path=workdir+'/model/{}'.format(it))
            output = "Model Saved. Iter {} Best_MMD {} Best_KL {} Best_RATE {}" \
            .format(it, '{:.4f}'.format(best_mmd), '{:.4f}'.format(best_kl), \
                '{:.4f}'.format(best_rate))
            logging.info(output)
            print(output)
            #plot_sample_density(true_sample, mesh_t_density, (X_range, Y_range), fake_sample, mesh_f_density_n, \
            #    method=method, save_path=workdir+'/vis/{}.pdf'.format(it))
        

        Wgr_loss_f = '{:.4f}'.format(float(loss_d))
        Sre_f = '{:.4f}'.format(float(ksd_rp))
        Sge_f = '{:.4f}'.format(float(ksd_gp))
        mmd_f = '{:.4f}'.format(mmd)
        kl_div_f = '{:.4f}'.format(kl_div)
        high_rate_f = '{:.4f}'.format(high_rate)
        cost_time = str((datetime.now() - start_time) / (it+1) * (n_iter - it)).split('.')[0]
        
        log = "Iter {:<6}: Wgr_loss {:<6} Sre_loss {:<6} Sge_f {:<6} MMD {:<6} KL_Div {:<6} Rate {:<6} (left: {})".\
            format(it, Wgr_loss_f, Sre_f, Sge_f, mmd_f, kl_div_f, high_rate, cost_time)
        logging.info(log)
        print(log)
