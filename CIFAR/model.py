#-*- coding: utf-8 -*-
import tensorflow as tf

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[32,32,3],
            dim_z=300,
            dim_W1=256,
            dim_W2=128,
            dim_W3=64,
            dim_channel=3,
            lambda_p=10,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        self.lambda_p = lambda_p

        self.gen_W1 = tf.get_variable(shape=[4,4,dim_W1, dim_z], name='gen_W1')
        self.gen_W2 = tf.get_variable(shape=[4,4,dim_W2, dim_W1], name='gen_W2')
        self.gen_W3 = tf.get_variable(shape=[4,4,dim_W3,dim_W2], name='gen_W3')
        self.gen_W4 = tf.get_variable(shape=[4,4,dim_channel,dim_W3], name='gen_W4')

        self.dis2gen_W1 = tf.get_variable(shape=[4,4,dim_channel,dim_W3], name='dis2gen_W1')
        self.dis2gen_W2 = tf.get_variable(shape=[4,4,dim_W3,dim_W2], name='dis2gen_W2')
        self.dis2gen_W3 = tf.get_variable(shape=[4,4,dim_W2,dim_W1], name='dis2gen_W3')
        self.dis2gen_W4 = tf.get_variable(shape=[4,4,dim_W1,1], name='dis2gen_W4')

        self.est_W1 = tf.get_variable(shape=[4,4,dim_channel,dim_W3], name='est_W1')
        self.est_W2 = tf.get_variable(shape=[4,4,dim_W3,dim_W2], name='est_W2')
        self.est_W3 = tf.get_variable(shape=[4,4,dim_W2,dim_W1], name='est_W3')
        self.est_W4 = tf.get_variable(shape=[4,4,dim_W1,128], name='est_W4')

        self.dis2est_W1 = tf.get_variable(shape=[4,4,dim_channel,dim_W3], name='dis2est_W1')
        self.dis2est_W2 = tf.get_variable(shape=[4,4,dim_W3,dim_W2], name='dis2est_W2')
        self.dis2est_W3 = tf.get_variable(shape=[4,4,dim_W2,dim_W1], name='dis2est_W3')
        self.dis2est_W4 = tf.get_variable(shape=[4,4,dim_W1,1], name='dis2est_W4')


    def build_model(self):

        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        def dis_loss(x_gen, x_true):
            u = tf.random_uniform([tf.shape(x_gen)[0]], minval=0, maxval=1)
            u = tf.reshape(u, [-1, 1, 1, 1])
            u = tf.tile(u, [1, tf.shape(x_gen)[1], tf.shape(x_gen)[2], tf.shape(x_gen)[3]])
            x_ = tf.multiply(u, x_gen) + tf.multiply(tf.ones_like(u)-u, x_true)
            gradient_norm = tf.sqrt(tf.reduce_sum(
                tf.square(tf.gradients(self.dis2gen(x_), x_)[0]), axis=1))
            penalty = tf.reduce_mean(tf.nn.relu(gradient_norm - 1))
            loss_gen = tf.reduce_mean(self.dis2gen(x_gen))
            loss_true = tf.reduce_mean(self.dis2gen(x_true))
            loss =  loss_gen - loss_true + self.lambda_p*penalty
            return loss

        def ksd_emp(image):
            p = self.estimate(image)
            sq = tf.gradients(p, image)[0]
            sq = tf.reshape(sq, [-1, 32*32*3])
            f_x = self.dis2est(image)
            #l2 = tf.reduce_sum(tf.square(f_x))
            f_x = tf.tile(tf.reshape(f_x, [-1, 1]), [1, 32*32*3])
            df_x = tf.gradients(f_x, image)[0]
            df_x = tf.reshape(df_x, [-1, 32*32*3])
            l2 = tf.reduce_mean(tf.nn.relu(tf.reduce_sum(tf.square(df_x), 1)-1))
            ksd = tf.multiply(sq, f_x) + df_x

            return ksd, l2

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        image_gen = self.generate(Z)
        dis2gen_cost = dis_loss(image_gen, image_real)

        gen2dis_cost = -tf.reduce_mean(self.dis2gen(image_gen)) 
        ksd_gen, l2_gen = ksd_emp(image_gen)

        
        gen2est_cost = tf.reduce_mean(ksd_gen, 0)
        gen2est_cost = tf.reduce_max(tf.abs(gen2est_cost))
        est2gen_cost = gen2est_cost

        ksd_real, l2_real = ksd_emp(image_real)
        
        est2real_cost = tf.reduce_mean(ksd_real, 0)
        est2real_cost = tf.reduce_max(tf.abs(est2real_cost))
        
        l2 = l2_gen + l2_real #+ 100*(tf.nn.relu(l2_weight1-1) + tf.nn.relu(l2_weight2-1))

        p_real = tf.exp(self.estimate(image_real))
        p_gen = tf.exp(self.estimate(image_gen))

        return Z, image_real, dis2gen_cost, gen2dis_cost, \
            gen2est_cost, est2real_cost, est2gen_cost, l2, p_real, p_gen
        


    def dis2gen(self, image):

        h1 = lrelu( tf.nn.conv2d(image, self.dis2gen_W1, strides=[1,2,2,1], padding='SAME' ))
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.dis2gen_W2, strides=[1,2,2,1], padding='SAME')) )

        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.dis2gen_W3, strides=[1,2,2,1], padding='SAME')) )
        
        h4 = tf.nn.conv2d( h3, self.dis2gen_W4, strides=[1,1,1,1], padding='VALID')
        
        return h4

    def dis2est(self, image):

        h1 = lrelu( tf.nn.conv2d(image, self.dis2est_W1, strides=[1,2,2,1], padding='SAME' ))
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.dis2est_W2, strides=[1,2,2,1], padding='SAME')) )

        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.dis2est_W3, strides=[1,2,2,1], padding='SAME')) )
        
        h4 = tf.nn.tanh(tf.nn.conv2d( h3, self.dis2est_W4, strides=[1,1,1,1], padding='VALID'))
        
        return h4

    def estimate(self, image):
        h1 = lrelu( tf.nn.conv2d(image, self.est_W1, strides=[1,2,2,1], padding='SAME' ))
        
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.est_W2, strides=[1,2,2,1], padding='SAME')) )

        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.est_W3, strides=[1,2,2,1], padding='SAME')) )
        
        h4 = tf.nn.conv2d( h3, self.est_W4, strides=[1,1,1,1], padding='VALID')
        h4 = tf.reshape(h4, [-1, 128])
        
        E = tf.nn.softplus(tf.reduce_mean(h4, -1))
        out = -E

        return out

    def generate(self, Z):

        Z = tf.reshape(Z, [-1, 1, 1, self.dim_z])
        output_shape_l1 = [self.batch_size,4,4,self.dim_W1]
        h1 = tf.nn.conv2d_transpose(Z, self.gen_W1, output_shape=output_shape_l1, strides=[1,1,1,1], padding='VALID')
        h1 = tf.nn.relu(batchnormalize(h1))
        
        output_shape_l2 = [self.batch_size,8,8,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1], padding='SAME')
        h2 = tf.nn.relu(batchnormalize(h2))
        h2 = tf.reshape(h2, [self.batch_size,8,8,self.dim_W2])

        output_shape_l3 = [self.batch_size,16,16,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1], padding='SAME')
        h3 = tf.nn.relu( batchnormalize(h3) )

        output_shape_l4 = [self.batch_size,32,32,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1], padding='SAME')
        x = tf.nn.tanh(h4)
        return x

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])

        Z_ = tf.reshape(Z, [-1, 1, 1, self.dim_z])
        output_shape_l1 = [self.batch_size,4,4,self.dim_W1]
        h1 = tf.nn.conv2d_transpose(Z_, self.gen_W1, output_shape=output_shape_l1, strides=[1,1,1,1], padding='VALID')
        h1 = tf.nn.relu(batchnormalize(h1))
        
        output_shape_l2 = [self.batch_size,8,8,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1], padding='SAME')
        h2 = tf.nn.relu(batchnormalize(h2))
        h2 = tf.reshape(h2, [self.batch_size,8,8,self.dim_W2])

        output_shape_l3 = [self.batch_size,16,16,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1], padding='SAME')
        h3 = tf.nn.relu( batchnormalize(h3) )

        output_shape_l4 = [self.batch_size,32,32,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1], padding='SAME')

        x = tf.nn.tanh(h4)
        return Z,x

    def samples_estimator(self, batch_size):
        X = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)

        h1 = lrelu( tf.nn.conv2d(X, self.est_W1, strides=[1,2,2,1], padding='SAME' ))
        
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.est_W2, strides=[1,2,2,1], padding='SAME')) )

        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.est_W3, strides=[1,2,2,1], padding='SAME')) )
        
        h4 = tf.nn.conv2d( h3, self.est_W4, strides=[1,1,1,1], padding='VALID')
        h4 = tf.reshape(h4, [-1, 128])
        
        E = tf.nn.softplus(tf.reduce_mean(h4, -1))
        out = tf.exp(-E)

        return X, out