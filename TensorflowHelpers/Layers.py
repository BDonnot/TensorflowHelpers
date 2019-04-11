import copy
import pdb

import numpy as np
import tensorflow as tf

DTYPE_USED = tf.float16 # doesn't work atm
DTYPE_USED = tf.float32

DTYPE_NPY = np.float16 if DTYPE_USED == tf.float16 else np.float32
from .Losses import l2

class Layer(object):
    """
    Represent a neural network layer type, for example a fully connected, or a residual block
    """
    def __init__(self):
        self.flops = 0
        self.nbparams = 0
        self.res = None

    def start_exp(self, sess):
        pass

    def tell_epoch(self, sess, epochnum):
        pass

    def initwn(self, sess, scale_init=1.0):
        pass

    def get_flops(self):
        return self.flops

    def get_res(self):
        return self.res

    def get_nb_params(self):
        return self.nbparams

    def start_train(self, sess):
        """
        Inform the model the training process has started
        :return:
        """
        pass

    def start_test(self, sess):
        """
        Inform the model the model
        Deactivate all stochastic stuff
        :return:
        """
        pass

    def add_loss(self, previous_loss):
        """

        :param previous_loss:
        :return:
        """
        return previous_loss

class DenseLayer(Layer):
    def __init__(self, input, size, relu=False, bias=True, weight_normalization=False,
                 keep_prob=None, layernum=0):
        """
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        :param input: input of the layer 
        :param size: layer size (number of outputs units)
        :param relu: do you use relu ?
        :param bias: do you add bias ?
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :param keep_prob: a scalar tensor for dropout layer (None if you don't want to use it)
        :param layernum: number of layer (this layer in the graph)
        """
        Layer.__init__(self)
        nin_ = int(input.get_shape()[1])
        self.nbparams = 0  # number of trainable parameters
        self.flops = 0  # flops for a batch on 1 data
        self.input = input
        self.weightnormed = False
        self.bias = False
        self.relued = False
        self.res = None
        with tf.variable_scope("dense_layer_{}".format(layernum)):
            self.w_ = tf.get_variable(name="weights_matrix",
                                      shape=[nin_, size],
                                      initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False),
                                      # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                      trainable=True,
                                      dtype=tf.float32)  # weight matrix
            if DTYPE_USED != tf.float32:
                self.w_ = tf.cast(self.w_, DTYPE_USED)

            self.nbparams += int(nin_ * size)

            if weight_normalization:
                self.weightnormed = True
                self.g = tf.get_variable(shape=[size],
                                         name="weight_normalization_g",
                                         initializer=tf.constant_initializer(value=1.0, dtype=tf.float32),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weight_normalization_g:0"),
                                    trainable=True,
                                      dtype=tf.float32)
                if DTYPE_USED != tf.float32:
                    self.g = tf.cast(self.g, DTYPE_USED)
                self.nbparams += int(size)
                self.scaled_matrix = tf.nn.l2_normalize(self.w_, dim=0, name="weight_normalization_scaled_matrix")
                self.flops += size*(2*nin_-1)  # clomputation of ||v|| (size comptuation of inner product of vector of size nin_)
                self.flops += 2*nin_-1  # division by ||v|| (matrix vector product)
                self.w = tf.multiply(self.scaled_matrix, self.g, name="weight_normalization_weights")
                self.flops += 2*nin_-1  # multiplication by g (matrix vector product)
            else:
                self.w = self.w_

            self.res_ = tf.matmul(self.input, self.w, name="multiplying_weight_matrix")
            self.flops += 2*nin_*size-size

            if bias:
                self.bias = True
                self.b = tf.get_variable(shape=[size],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/bias:0"),
                                         name="bias",
                                         trainable=True,
                                         dtype=tf.float32)
                if DTYPE_USED != tf.float32:
                    self.b = tf.cast(self.b, DTYPE_USED)
                self.nbparams += int(size)
                self.res_ = tf.add(self.res_, self.b, name="adding_bias")
                self.flops += size # vectors addition of size "size"

            if relu:
                self.relued = True
                res = tf.nn.relu(self.res_, name="applying_relu")
                self.flops += size  # we consider relu of requiring 1 computation per number (one max)
            else:
                # I must have access to the output of the layer, before the non linearity
                # for debugginh initialization of weight normalization
                res = self.res_

            if keep_prob is not None:
                res = tf.nn.dropout(res, rate=1.-keep_prob, name="applying_dropout")
                # we consider that generating random number count for 1 operation
                self.flops += size  # generate the "size" real random numbers
                self.flops += size  # building the 0-1 vector of size "size" (thresholding "size" random values)
                self.flops += size  # element wise multiplication with res

            self.res = res

    def initwn(self, sess, scale_init=1.0):
        """
        initialize the weight normalization as describe in https://arxiv.org/abs/1602.07868
        don't do anything if the weigth normalization have not been "activated"
        :param sess: the tensorflow session
        :param scale_init: the initial scale
        :return: 
        """
        if not self.weightnormed:
            return
        # input = sess.run(input)
        with tf.variable_scope("init_wn_layer"):
            m_init, v_init = sess.run(tf.nn.moments(tf.matmul(self.input, self.scaled_matrix), [0]))
            sess.run(tf.assign(self.g, scale_init/tf.sqrt(v_init + 1e-10), name="weigth_normalization_init_g"))
            if self.bias:
                sess.run(tf.assign(self.b, -m_init*scale_init/tf.sqrt(v_init + 1e-10), name="weigth_normalization_init_b"))
        # pdb.set_trace()
        #     res = sess.run()
        # return res

class ResidualBlock(Layer):
    def __init__(self, input, size, relu=False, bias=True,
                 weight_normalization=False, keep_prob=None, outputsize=None, layernum=0,
                 layerClass=DenseLayer,
                 kwardslayer={}):
        """
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        this block is insipired from https://arxiv.org/abs/1603.05027 :
        X -> Bn(.) -> Relu(.) -> W * . -> Bn(.) -> Relu(.) -> W_2 * . -> X + .
        with "." being the output of the previous computation
        Bn() -> batch normalization (currently unused)
        Relu(.) -> rectifier linear unit
        * -> matrix product
        
        dropout (regular) is done at the end of the comptuation
        
        the number of input and number of output is the same. Size is just the intermediate size
        
        :param input: input of the layer 
        :param size: layer size (number of layer after "X -> Bn(.) -> Relu(.) -> W * .")
        :param relu: do you use relu ? (at the end, just before standard dropout)
        :param bias: do you add bias ?
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :param keep_prob: a scalar tensor for dropout layer (None if you don't want to use it)
        :param layernum: number of layer (this layer in the graph)
        :param layerClass: the class used to build the layers (DenseLayer or one of its derivatives) -- do not pass an object, but the class
        :param kwardslayer: the key-words arguments pass when building the instances of class layerClass
        """

        Layer.__init__(self)

        self.input = input
        self.weightnormed = weight_normalization
        self.res = input

        with tf.variable_scope("residual_block_{}".format(layernum)):
            # treating "-> Bn() -> Relu()"
            if relu:
                self.res = tf.nn.relu(self.res, name="first_relu")
                self.flops += int(self.res.get_shape()[1])

            #treating "-> W * . -> Bn(.) -> Relu(.)"
            self.first_layer = None
            with tf.variable_scope("resBlock_first_layer"):
                self.first_layer = layerClass(self.res, size, relu=True, bias=bias,
                                             weight_normalization=weight_normalization,
                                             keep_prob=None,
                                              **kwardslayer)
                self.flops += self.first_layer.flops
                self.nbparams += self.first_layer.get_nb_params()
                self.res = self.first_layer.get_res()

            # treating "-> W_2 * . "
            self.second_layer = None
            with tf.variable_scope("resBlock_second_layer"):
                self.second_layer = layerClass(self.res, int(input.get_shape()[1]), relu=False, bias=bias,
                                               weight_normalization=weight_normalization,
                                               keep_prob=None,
                                               **kwardslayer)
                self.flops += self.second_layer.get_flops()
                self.nbparams += self.second_layer.get_nb_params()

            # treating "-> X + ."

            if keep_prob is not None:
                #TODO : copy pasted from DenseLayer
                tmp = tf.nn.dropout(self.second_layer.get_res(), rate=1.-keep_prob, name="applying_dropout")
                # we consider that generating random number count for 1 operation
                self.flops += size  # generate the "size" real random numbers
                self.flops += size  # building the 0-1 vector of size "size" (thresholding "size" random values)
                self.flops += size  # element wise multiplication with res
            else:
                tmp = self.second_layer.get_res()

            self.res = tmp + input
            self.flops += int(self.res.get_shape()[1])

            if relu:
                self.res = tf.nn.relu(self.res, name="applying_relu")
                self.flops += size  # we consider relu of requiring 1 computation per number (one max)


    def initwn(self, sess, scale_init=1.0):
        """
        initialize the weight normalization as describe in https://arxiv.org/abs/1602.07868
        don't do anything if the weigth normalization have not been "activated"
        :param sess: the tensorflow session
        :param scale_init: the initial scale
        :return: nothing
        """
        if not self.weightnormed:
            return

        self.first_layer.initwn(sess=sess, scale_init=0.05)
        self.second_layer.initwn(sess=sess, scale_init=0.05)


class DenseBlock(Layer):
    def __init__(self, input, size, relu=False, bias=True, weight_normalization=False,
                 keep_prob=None, nblayer=2, layernum=0,
                 layerClass=DenseLayer,
                 kwardslayer={}):
        """
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        
        this block is insipired from 
        https://www.researchgate.net/publication/306885833_Densely_Connected_Convolutional_Networks :
        
        dropout (regular) is done at the end of the comptuation
        
        the size of each layer should be the same. That's why there is no "sizes" parameter.
        
        :param input: input of the layer 
        :param relu: do you use relu ? (at the end, just before standard dropout)
        :param bias: do you add bias ?
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :param keep_prob: a scalar tensor for dropout layer (None if you don't want to use it)
        :param nblayer: the number of layer in the dense block
        :param layernum: number of layer (this layer in the graph)
        :param size: unused, for compatibility with ResidualBlock and DenseLayer
        :param layerClass: the class used to build the layers (DenseLayer or one of its derivatives) -- do not pass an object, but the class
        :param kwardslayer: the key-words arguments pass when building the instances of class layerClass
        """
        Layer.__init__(self)
        self.input = input
        self.weightnormed = weight_normalization
        size = int(input.get_shape()[1])
        self.res = input
        self.layers = []

        with tf.variable_scope("dense_block_{}".format(layernum)):
            for i in range(nblayer):
                tmp_layer = layerClass(self.res, size, relu=True, bias=bias,
                                       weight_normalization=weight_normalization,
                                       keep_prob=None, layernum=i,
                                       **kwardslayer)
                self.flops += tmp_layer.get_flops()
                self.nbparams += tmp_layer.get_nb_params()
                self.res = tmp_layer.get_res()
                for l in self.layers:
                    self.res = self.res + l.get_res()
                    self.flops += size
                self.layers.append(tmp_layer)

            if relu:
                self.res = tf.nn.relu(self.res, name="applying_relu")
                self.flops += size  # we consider relu of requiring 1 computation per number (one max)

            if keep_prob is not None:
                #TODO : copy pasted from DenseLayer
                self.res = tf.nn.dropout(self.res, rate=1.-keep_prob, name="applying_dropout")
                # we consider that generating random number count for 1 operation
                self.flops += size  # generate the "size" real random numbers
                self.flops += size  # building the 0-1 vector of size "size" (thresholding "size" random values)
                self.flops += size  # element wise multiplication with res

    def initwn(self, sess, scale_init=1.0):
        """
        initialize the weight normalization as describe in https://arxiv.org/abs/1602.07868
        don't do anything if the weigth normalization have not been "activated"
        :param sess: the tensorflow session
        :param scale_init: the initial scale
        :return: nothing
        """
        if not self.weightnormed:
            return
        for i, l in enumerate(self.layers):
            l.initwn(sess=sess, scale_init=0.1/(i+1))


class VAEBlock(Layer):
    def __init__(self, input, size, cvae=None, sizes=[], relu=False, bias=True,
                 weight_normalization=False, keep_prob=None, layernum=0,
                 reconstruct_loss=l2
                 ):
        """

        :param input:
        :param size: size of the latent space
        :param cvae: variable to pass as conditional variational encoder (None if none...) [tensorflow tensor]
        :param sizes:
        :param bias:
        :param weight_normalization:
        :param keep_prob:
        :param layernum:
        :param reconstruct_loss:
        """
        Layer.__init__(self)
        self.input = input
        self.weightnormed = weight_normalization
        self.res = input
        self.layers = []
        self.size_latent = size
        self.cvae = cvae
        self.reconstruct_loss = reconstruct_loss
        self.layernum = layernum
        self.last_pen_loss = 1.

        with tf.variable_scope("vae_block_{}".format(layernum)):
            with tf.variable_scope("vae_def_flexibility"):
                # define variable to add noise: 0 no noise are added in the embedding, 1: all noise is added
                self.amount_vae_ph = tf.placeholder(dtype=DTYPE_USED, shape=(), name="skip_conn")
                self.amount_vae = tf.Variable(tf.zeros(shape=self.amount_vae_ph.get_shape(), dtype=DTYPE_USED),
                                              trainable=False)
                self.assign_vae = tf.assign(self.amount_vae, self.amount_vae_ph, name="assign_vae")

                # to activate / deactivate the encoder part of the VAE (deactivate it for data generation)
                # 1 = I use the encoder part of the VAE, 0 i do pure generation
                self.use_vae_enc_ph = tf.placeholder(dtype=DTYPE_USED, shape=(), name="use_vae_pred")
                # use_vae_pred is set to 1 during training and 0 when making forecast
                # it deactivated the input of the VAE, making proper predictions
                self.use_vae_enc = tf.Variable(tf.zeros(shape=self.use_vae_enc_ph.get_shape(), dtype=DTYPE_USED),
                                               trainable=False)
                self.assign_use_vae_enc = tf.assign(self.use_vae_enc, self.use_vae_enc_ph, name="assign_use_vae_enc")


                # to penalize more or less the reconstruction loss
                self.pen_reco_loss_ph = tf.placeholder(dtype=DTYPE_USED, shape=(), name="use_vae_pred")
                # use_vae_pred is set to 1 during training and 0 when making forecast
                # it deactivated the input of the VAE, making proper predictions
                self.pen_reco_loss = tf.Variable(tf.zeros(shape=self.pen_reco_loss_ph.get_shape(), dtype=DTYPE_USED),
                                               trainable=False)
                self.assign_pen_reco_loss = tf.assign(self.pen_reco_loss, self.pen_reco_loss_ph, name="assign_use_vae_enc")

            if cvae is not None:
                with tf.variable_scope("cvae_concat"):
                    self.res = tf.concat((self.latent_z_, self.cvae), axis=1, name="cvae_input_concatenantion")

            with tf.variable_scope("encoder"):
                for i, sz in enumerate(sizes):
                    tmp_layer = DenseLayer(self.res, sz, relu=True, bias=bias,
                                           weight_normalization=weight_normalization,
                                           keep_prob=keep_prob, layernum=i)
                    self.flops += tmp_layer.get_flops()
                    self.nbparams += tmp_layer.get_nb_params()
                    self.res = tmp_layer.get_res()
                    self.layers.append(tmp_layer)

                # extract mean and std
                self.mu_vae = DenseLayer(input=self.res, size=self.size_latent, layernum="mu",
                                         relu=False, keep_prob=None, bias=False)
                # build the mean
                self.log_std = DenseLayer(input=self.res, size=self.size_latent , layernum="log_std",
                                          relu=False, keep_prob=None, bias=False)

                self.res = tf.identity(self.mu_vae.get_res(), name="extracting_embedding")

            with tf.variable_scope("sampling"):

                self.std_vae = tf.exp(.5 * self.log_std.get_res())

                # sample a N(0,1) same shape as log std
                self.epsilon = tf.random_normal(tf.shape(self.mu_vae.get_res()), name='sample_epsilon')

                # compute the latent variable (cf. formula in the links)
                self.latent_z_ = (self.use_vae_enc*(self.mu_vae.get_res() + self.amount_vae*tf.multiply(self.std_vae, self.epsilon)) +
                                (1. - self.use_vae_enc)*self.epsilon )
                if cvae is not None:
                    with tf.variable_scope("cvae_concat"):
                        self.latent_z = tf.concat((self.latent_z_, self.cvae), axis=1, name="cvae_z_concatenantion")
                else:
                    self.latent_z = self.latent_z_
                tmp = self.latent_z


            with tf.variable_scope("decoder"):
                for i, sz in enumerate(sizes[::-1]):
                    tmp_layer = DenseLayer(tmp, sz, relu=True, bias=bias,
                                           weight_normalization=weight_normalization,
                                           keep_prob=keep_prob, layernum=i)
                    self.flops += tmp_layer.get_flops()
                    self.nbparams += tmp_layer.get_nb_params()
                    tmp = tmp_layer.get_res()
                    self.layers.append(tmp_layer)

                # TODO add final layer
                self.reconstruct = DenseLayer(input=self.res, size=self.input.shape[1], layernum="reconstruction",
                                              relu=False, keep_prob=None, bias=False)
                self.flops += self.reconstruct.get_flops()
                self.nbparams += self.reconstruct.get_nb_params()
                self.layers.append(self.reconstruct)
                self.reconstruction = self.reconstruct.get_res()

            with tf.variable_scope("loss_vae_{}".format(layernum)):
                self.l_reco = self.reconstruct_loss(pred=self.reconstruction, true=self.input,
                                             name="reconstruction_loss_{}".format(self.layernum),
                                             multiplier=self.pen_reco_loss)

                self.kld_ = -.5 * self.pen_reco_loss * tf.reduce_mean(
                    tf.reduce_sum(
                        1. + self.log_std.get_res() - tf.square(self.mu_vae.get_res()) - tf.exp(self.log_std.get_res()),
                        reduction_indices=1,
                        name="kl_divergence_{}".format(self.layernum)))
                self.myloss = tf.add(self.pen_reco_loss, self.l_reco, name="total_loss_vae")


    def deactivate_encoder(self, sess):
        """
        Used for pure data generation, deactivate the encoder part. The latent representation is purely random.
        :return:
        """
        sess.run([self.assign_use_vae_enc], feed_dict={self.use_vae_enc_ph: 0.0})

    def activate_encoder(self, sess):
        """
        Used for activate the encoder part. Can't do pure data generation.
        :return:
        """
        sess.run([self.assign_use_vae_enc], feed_dict={self.use_vae_enc_ph: 1.0})


    def start_exp(self, sess):
        """
        TODO documentation
        :param sess:
        :return:
        """
        sess.run([self.assign_vae, self.assign_use_vae_enc],
                     feed_dict={self.amount_vae_ph: 1.0, self.use_vae_enc_ph: 1.0})

    def start_train(self, sess):
        """
        Inform the model the training process has started
        :return:
        """
        sess.run([self.assign_vae, self.assign_use_vae_enc],
                     feed_dict={self.amount_vae_ph: 1.0, self.use_vae_enc_ph: 1.0})
        sess.run([self.assign_pen_reco_loss],
                     feed_dict={self.pen_reco_loss_ph: self.last_pen_loss })

    def start_test(self, sess):
        """
        Inform the model the model
        Deactivate all stochastic stuff
        :return:
        """
        sess.run([self.assign_vae, self.assign_use_vae_enc],
                     feed_dict={self.amount_vae_ph: 0.0, self.use_vae_enc_ph: 1.0})
        sess.run([self.assign_pen_reco_loss],
                     feed_dict={self.pen_reco_loss_ph: 0. })

    def add_loss(self, previous_loss):
        """

        :param previous_loss:
        :return:
        """
        return tf.add(previous_loss, self.myloss,
                      name="adding_reco_loss_{}".format(self.layernum))

    def tell_epoch(self, sess, epochnum):
        self.last_pen_loss = 1./(1.+epochnum)
        sess.run([self.assign_pen_reco_loss],
                     feed_dict={self.pen_reco_loss_ph: self.last_pen_loss })