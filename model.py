#===============================================================================
# MIT License
#
# Copyright (c) 2017 Jake Bruce
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#===============================================================================

import numpy      as np
import tensorflow as tf

#===============================================================================
# MODEL

class VAEGAN:
    def __init__(self,
                 input_size,
                 encoder_sizes,
                 latent_size,
                 decoder_sizes,
                 advers_sizes,
                 drop_p,
                 noise_p,
                 learn_rate,
                 rec_loss_mul,
                 lat_loss_mul,
                 adv_loss_mul,
                 gradient_max):

        self.drop_p     = drop_p
        self.noise_p    = noise_p
        self.learn_rate = tf.Variable(learn_rate, trainable=False)

        # runtime input parameters
        self.input  = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.target = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None,          2], dtype=tf.float32)
        self.keep_p = tf.placeholder(shape=[                ], dtype=tf.float32)
        self.random = tf.placeholder(shape=[                ], dtype=tf.bool)

        # clip for numerical stability
        self.input  = tf.clip_by_value(self.input,  1e-8, 1-1e-8)
        self.target = tf.clip_by_value(self.target, 1e-8, 1-1e-8)

        # initializer shorthands
        xavier = tf.contrib.layers.xavier_initializer
        zeros  = tf.zeros_initializer

        # layer stack specs
        enc_spec = [(i,si,so) for i,(si,so) in enumerate(zip([ input_size]+encoder_sizes, encoder_sizes))]
        dec_spec = [(i,si,so) for i,(si,so) in enumerate(zip([latent_size]+decoder_sizes, decoder_sizes))]
        adv_spec = [(i,si,so) for i,(si,so) in enumerate(zip([ input_size]+ advers_sizes,  advers_sizes))]

        #--------------------------------
        # FORWARD VARIATIONAL AUTOENCODER
        #--------------------------------

        # encoder parameters
        enc_weights  = [tf.get_variable("e%dw"%i, shape=[in_s, out_s], initializer=xavier()) for i,in_s,out_s in enc_spec]
        enc_biases   = [tf.get_variable("e%db"%i, shape=[      out_s], initializer= zeros()) for i,in_s,out_s in enc_spec]
        mu_weights   =  tf.get_variable("mw",  shape=[encoder_sizes[-1],     latent_size], initializer=xavier())
        mu_biases    =  tf.get_variable("mb",  shape=[                       latent_size], initializer= zeros())
        sig_weights  =  tf.get_variable("sw",  shape=[encoder_sizes[-1],     latent_size], initializer=xavier())
        sig_biases   =  tf.get_variable("sb",  shape=[                       latent_size], initializer= zeros())

        # generator parameters
        dec_weights  = [tf.get_variable("d%dw"%i, shape=[in_s, out_s], initializer=xavier()) for i,in_s,out_s in dec_spec]
        dec_biases   = [tf.get_variable("d%db"%i, shape=[      out_s], initializer= zeros()) for i,in_s,out_s in dec_spec]
        out_weights  =  tf.get_variable("ow",  shape=[decoder_sizes[-1],      input_size], initializer=xavier())
        out_biases   =  tf.get_variable("ob",  shape=[                        input_size], initializer= zeros())

        # trainable variables for the forward variational autoencoder
        vae_vars = [mu_weights, mu_biases, sig_weights, sig_biases, out_weights, out_biases]
        vae_vars.extend(enc_weights)
        vae_vars.extend(enc_biases)
        vae_vars.extend(dec_weights)
        vae_vars.extend(dec_biases)

        # linear combination helper
        def lin(x, w, b): return tf.matmul(x,w)+b
        def drp(      x): return tf.nn.dropout(x,keep_prob=self.keep_p)

        # encoding pass to generate mean and stdev of generated distribution
        features     = self.input
        for w,b in zip(enc_weights, enc_biases)[:-1]: features = tf.nn.elu(lin(drp(features), w, b))
        features     = tf.nn.tanh    (lin(drp(features), enc_weights[-1], enc_biases[-1]))
        dropfeat     = drp           (features)
        mu           =               (lin(dropfeat,   mu_weights,     mu_biases))
        sig          = tf.nn.softplus(lin(dropfeat,  sig_weights,    sig_biases)) + 1e-8

        # sample a latent vector from this distribution
        self.latent  = tf.cond(self.random,lambda:mu+tf.random_normal(shape=[tf.shape(self.input)[0],latent_size])*sig,lambda:mu)

        # decoding pass to reconstruct the input
        features     = self.latent
        features     = tf.nn.tanh      (lin(features, dec_weights[0], dec_biases[0]))
        for w,b in zip(dec_weights, dec_biases)[1:]: features = tf.nn.elu(lin(drp(features), w, b))
        self.recon   = tf.clip_by_value(tf.sigmoid(lin(features, out_weights, out_biases)), 1e-8, 1-1e-8)

        #----------------------
        # ADVERSARIAL COMPONENT
        #----------------------

        # adversarial parameters
        adv_weights  = [tf.get_variable("a%dw"%i, shape=[in_s, out_s], initializer=xavier()) for i,in_s,out_s in adv_spec]
        adv_biases   = [tf.get_variable("a%db"%i, shape=[      out_s], initializer= zeros()) for i,in_s,out_s in adv_spec]
        cls_weights  =  tf.get_variable("cw",  shape=[ advers_sizes[-1],               2], initializer=xavier())
        cls_biases   =  tf.get_variable("cb",  shape=[                                 2], initializer= zeros())

        # trainable variables for the adversarial component
        adv_vars = [cls_weights, cls_biases]
        adv_vars.extend(adv_weights)
        adv_vars.extend(adv_biases)

        # sample noise vectors and generate digits
        features     = tf.random_normal(shape=[tf.shape(self.input)[0]/2,latent_size])
        features     = tf.nn.tanh      (lin(features, dec_weights[0], dec_biases[0]))
        for w,b in zip(dec_weights, dec_biases)[1:]: features = tf.nn.elu(lin(features, w, b))
        gen_digits   = tf.clip_by_value(tf.sigmoid(lin(features, out_weights, out_biases)), 1e-8, 1-1e-8)

        # adversarial pass to discriminate generated digits from real digits
        features     = tf.concat([gen_digits, self.target[tf.shape(self.target)[0]/2:,:]], axis=0)
        for w,b in zip(adv_weights, adv_biases): features = tf.nn.elu(lin(drp(features), w, b))
        logits       = lin(drp(features), cls_weights, cls_biases)

        #-------------
        # OPTIMIZATION
        #-------------

        # forward optimization
        latent_loss =  tf.reduce_mean(0.5*tf.reduce_sum(mu**2 + sig**2 - tf.log(1e-8 + sig**2) - 1, 1))*lat_loss_mul
        recon_loss  = -tf.reduce_mean(tf.reduce_sum(self.target*tf.log(self.recon)+(1-self.target)*tf.log(1-self.recon),1))*rec_loss_mul
        self.losses = [recon_loss, latent_loss]

        # adversarial optimization
        self.adv_loss   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))*adv_loss_mul

        # optimization steps with gradient clipping
        vae_opt  = tf.train.AdamOptimizer(self.learn_rate)
        adv_opt  = tf.train.AdamOptimizer(self.learn_rate)
        vae_grad = vae_opt.compute_gradients(latent_loss+recon_loss-self.adv_loss, var_list=vae_vars)
        adv_grad = adv_opt.compute_gradients(                       self.adv_loss, var_list=adv_vars)
        vae_clip = [(tf.clip_by_norm(g,gradient_max), v) for g,v in vae_grad]
        adv_clip = [(tf.clip_by_norm(g,gradient_max), v) for g,v in adv_grad]
        self.train_step = vae_opt.apply_gradients(vae_clip)
        self.adv_step   = adv_opt.apply_gradients(adv_clip)

    #---------------------------------------------------------------------------

    def train(self, sess, batch):
        # apply salt and pepper input noise
        noisy = batch.copy()
        for i in range(len(noisy)):
            noise = np.random.random(noisy[i].shape)
            noisy[i,noise>(  self.noise_p)] = 1
            noisy[i,noise<(1-self.noise_p)] = 0

        # labels for GAN
        labels  = np.zeros((batch.shape[0], 2), dtype=np.float32)
        labels[:batch.shape[0]/2 ] = [1,0]
        labels[ batch.shape[0]/2:] = [0,1]

        return sess.run([self.train_step,
                         self.adv_step,
                         self.latent,
                         self.recon,
                         self.losses,
                         self.adv_loss],
                        feed_dict={self.input  : noisy,
                                   self.target : batch,
                                   self.labels : labels,
                                   self.random : True,
                                   self.keep_p : self.drop_p})[2:]

    #---------------------------------------------------------------------------

    def test(self, sess, batch):
        return sess.run([self.latent,
                         self.recon,
                         self.losses],
                        feed_dict={self.input  : batch,
                                   self.target : batch,
                                   self.random : False,
                                   self.keep_p : 1})

    #---------------------------------------------------------------------------

    def generate(self, sess, latents):
        return sess.run([self.recon], feed_dict={self.latent : latents,
                                                 self.keep_p : 1})[0]

#===============================================================================

