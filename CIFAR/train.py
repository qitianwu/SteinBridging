import os
import numpy as np
from util import *
from model import *

from load import *
import pickle
from datetime import datetime
import time

import logging
logging.basicConfig(level=logging.INFO, filename='new_max_tanh.log', format='%(message)s')

flags = tf.app.flags
flags.DEFINE_string("gpus", "0", "gpus")
FLAGS = flags.FLAGS

print(FLAGS.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus

from metrics.metric import *


n_epochs = 500
lr_gan = 0.0002
lr_est = 0.0002
batch_size = 100
image_shape = [32,32,3]
dim_z = 100
dim_W1 = 256
dim_W2 = 128
dim_W3 = 64
dim_channel = 3
lambda_p = 100
n_critic = 1

visualize_dim=100

trX = cifar10()

dcgan_model = DCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        lambda_p=lambda_p
        )

Z_tf, image_tf, dis2gen_cost_tf, gen2dis_cost_tf, gen2est_cost_tf, est2real_cost_tf, \
est2gen_cost_tf, l2, p_real, p_gen = dcgan_model.build_model()

L = tf.placeholder(tf.float32, shape=None)

dis_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
dis2gen_vars = filter(lambda x: x.name.startswith('dis2gen'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
dis2est_vars = filter(lambda x: x.name.startswith('dis2est'), tf.trainable_variables())
est_vars = filter(lambda x: x.name.startswith('est'), tf.trainable_variables())
dis2gen_vars = [i for i in dis2gen_vars] + [i for i in dis_vars]
dis2est_vars = [i for i in dis2est_vars] + [i for i in dis_vars]
gen_vars = [i for i in gen_vars]
est_vars = [i for i in est_vars]

est_cost_tf = est2real_cost_tf + L*est2gen_cost_tf
dis2est_cost_tf = -est_cost_tf #+ l2
gen2est_cost_tf = L*est2gen_cost_tf

train_op_dis2gen = tf.train.AdamOptimizer(lr_gan, beta1=0.5).minimize(dis2gen_cost_tf, var_list=dis2gen_vars)
train_op_dis2est = tf.train.AdamOptimizer(lr_est, beta1=0.9).minimize(dis2est_cost_tf , var_list=dis2est_vars)
train_op_gen2dis = tf.train.AdamOptimizer(lr_gan, beta1=0.5).minimize(gen2dis_cost_tf, var_list=gen_vars)
train_op_gen2est = tf.train.AdamOptimizer(lr_est, beta1=0.9).minimize(gen2est_cost_tf, var_list=gen_vars)
train_op_est = tf.train.AdamOptimizer(lr_est, beta1=0.9).minimize(est_cost_tf, var_list=est_vars)

Z_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)


model_var = [var for var in tf.trainable_variables() if var.name.startswith("dis2gen")] \
        + [var for var in tf.trainable_variables() if var.name.startswith("dis2est")] \
        + [var for var in tf.trainable_variables() if var.name.startswith("gen")] \
        + [var for var in tf.trainable_variables() if var.name.startswith("est")]

it = '0' 
start_epoch = int(it)
sess = tf.Session()
saver = tf.train.Saver(model_var)
sess.run(tf.global_variables_initializer())
#saver.restore(sess, save_path='./model1/model_'+it)


Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))

step = 100
n_sample = trX.shape[0]
iterations = start_epoch * n_sample / batch_size
start_time = datetime.now()
k = 5

for epoch in range(start_epoch, n_epochs):
    index = np.arange(n_sample)
    np.random.shuffle(index)
    trX = trX[index]

    for start, end in zip(
            range(0, n_sample, batch_size),
            range(batch_size, n_sample, batch_size)
            ):

        Image = trX[start:end]
        Xs = Image.reshape( [-1, 32, 32, 3]) / 255.
        Xs = 2*Xs - 1
        Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

        n_iter = n_epochs * n_sample // batch_size
        l = min(4*iterations / n_iter, 1)

        i_critic = 0
        while i_critic < n_critic:
            _, dis2gen_loss_val = sess.run(
                    [train_op_dis2gen, dis2gen_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        image_tf:Xs
                        })
            i_critic += 1

        _, gen2dis_loss_val = sess.run(
                    [train_op_gen2dis, gen2dis_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        })

        if iterations % k ==0:
            i_critic = 0
            while i_critic < n_critic:
                _, dis2est_loss_val = sess.run(
                    [train_op_dis2est, dis2est_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        image_tf:Xs,
                        L:l
                        })
                i_critic += 1
            _, est_loss_val = sess.run(
                    [train_op_est, est_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        image_tf:Xs,
                        L:l
                        })
            _, gen2est_loss_val = sess.run(
                    [train_op_gen2est, gen2est_cost_tf],
                    feed_dict={
                        Z_tf:Zs,
                        L:l
                        })

        Wgr_loss, Sre_loss, Sge_loss = sess.run([dis2gen_cost_tf, est2real_cost_tf, est2gen_cost_tf], \
            feed_dict={Z_tf:Zs, image_tf:Xs})
        p_real_val, p_gen_val = sess.run([p_real, p_gen], feed_dict={Z_tf:Zs, image_tf:Xs})
        
        iterations += 1
        
        if np.mod(iterations, step) == 0:
            Wgr_loss_f = '{:.4f}'.format(float(Wgr_loss))
            Sre_f = '{:.4f}'.format(float(Sre_loss))
            Sge_f = '{:.4f}'.format(float(Sge_loss))
            cost_time = str((datetime.now() - start_time) / (iterations+1) * (n_iter - iterations)).split('.')[0]
        
            log = "Iter {:<6}: Wgr_loss {:<6} Sre_loss {:<6} Sge_f {:<6} (left: {})".\
            format(iterations, Wgr_loss_f, Sre_f, Sge_f, cost_time)
            logging.info(log)
            print(log)

    if epoch % 10 == 0:

        
        generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample:Z_np_sample
                    })
        generated_samples = (generated_samples + 1.)/2.
        save_visualization(generated_samples, (10,10), save_path='./vis/sample_%04d.jpg' % int(epoch))
        #save_visualization(Image, (10,10), save_path='./vis/true.jpg')
        

        
        saver = tf.train.Saver(model_var)
        saver.save(sess, save_path='./model/model_%03d' % epoch, write_meta_graph=False)
        
        print("====epoch {} finishes and model saved====".format(epoch))

        Image = np.zeros([len(trX), 32, 32, 3])
        for start, end in zip(
            range(0, len(trX)-batch_size, batch_size),
            range(batch_size, len(trX), batch_size)
            ):
            Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

            generated_samples = sess.run(image_tf_sample, feed_dict={Z_tf_sample:Zs})
            generated_samples = (generated_samples + 1.)/2. * 255
            Image[start:end, :, :, :] = generated_samples

        Image = Image.transpose([0, 3, 1, 2])
        IS_mean, IS_std = inception_score(Image)
        log = 'Epoch {} IS score: {}'.format(epoch, IS_mean)
        logging.info(log)
        print(log)
        

    if epoch > 490:
        
        saver = tf.train.Saver(model_var)
        saver.save(sess, save_path='./model1/model_%03d' % epoch, write_meta_graph=False)
        
        print("====epoch {} finishes and model saved====".format(epoch))

        Image = np.zeros([len(trX), 32, 32, 3])
        for start, end in zip(
            range(0, len(trX)-batch_size, batch_size),
            range(batch_size, len(trX), batch_size)
            ):
            Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

            generated_samples = sess.run(image_tf_sample, feed_dict={Z_tf_sample:Zs})
            generated_samples = (generated_samples + 1.)/2. * 255
            Image[start:end, :, :, :] = generated_samples

        Image = Image.transpose([0, 3, 1, 2])
        IS_mean, IS_std = inception_score(Image)
        log = 'Epoch {} IS score: {}'.format(epoch, IS_mean)
        logging.info(log)
        print(log)