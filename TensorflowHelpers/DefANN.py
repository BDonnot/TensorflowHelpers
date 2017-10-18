import copy
import pdb

import numpy as np
import tensorflow as tf

# TODO same API for all blocl, using some king of parameter
# TODO so that I do not have to recode everything for guided dropout
# struct : nIn,nout,k for guided dropout, and nin,nout for standard net

# TODO find coherence between "params" of the net, and params for the brick
# TODO forward pass for initwn!


# TODO remove the placeHolder and stuff like that from here,
# TODO move it to another file
# TODO remove the loss of the definition of the neural network.


class DenseLayer:
    def __init__(self, input, size, relu=False, bias=True, guided_dropout_mask=None, weight_normalization=False):
        """
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        :param input: input of the layer 
        :param size: layer size (number of outputs units)
        :param relu: do you use relu ?
        :param bias: do you add bias ?
        :param guided_dropout_mask: tensor of the mask matrix  #TODO 
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :return: the output after computation
        """

        nin_ = int(input.get_shape()[1])
        self.nbparams = 0  # number of trainable parameters
        self.flops = 0  # flops for a batch on 1 data
        self.input = input
        self.weightnormed = False
        self.bias = False

        self.w = tf.get_variable(name="weights_matrix",
                            shape=[nin_, size],
                            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                            trainable=True)  # weight matrix
        self.nbparams += int(nin_ * size)

        if weight_normalization:
            self.weightnormed = True
            self.g = tf.get_variable(shape=[size],
                                name="weigth_normalization_g",
                                initializer=tf.constant_initializer(value=1.0, dtype="float32"),
                                trainable=True)
            self.nbparams += int(size)
            self.scaled_matrix = tf.nn.l2_normalize(self.w, dim=0, name="weigth_normalization_scaled_matrix")
            self.flops += size*(2*nin_-1) # clomputation of ||v|| (size comptuation of inner product of vector of size nin_)
            self.flops += 2*nin_-1 # division by ||v|| (matrix vector product)
            self.w = tf.multiply(self.scaled_matrix, self.g, name="weigth_normalization_weights")
            self.flops += 2*nin_-1 # multiplication by g (matrix vector product)

        if guided_dropout_mask is not None:
            #TODO implement it
            pass
        self.res = tf.matmul(self.input, self.w, name="multiplying_weight_matrix")
        self.flops += 2*nin_*size-size

        if bias:
            self.bias = True
            self.b = tf.get_variable(shape=[size],
                                initializer=tf.constant_initializer(value=0.0, dtype="float32"),
                                name="bias",
                                trainable=True)
            self.nbparams += int(size)
            self.res = tf.add(self.res, self.b, name="adding_bias")
            self.flops += size # vectors addition of size "size"

        if relu:
            self.res = tf.nn.relu(self.res, name="applying_relu")
            self.flops += size  # we consider relu of requiring 1 comptuiation (one max)

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
            m_init, v_init = sess.run(tf.nn.moments(tf.matmul(self.input, self.w), [0]))
            # pdb.set_trace()
            sess.run(tf.assign(self.g, scale_init/tf.sqrt(v_init + 1e-10), name="weigth_normalization_init_g"))
            if self.bias:
                sess.run(tf.assign(self.b, -m_init*scale_init, name="weigth_normalization_init_b"))

class NNFully:
    def __init__(self, input, outputsize, layersizes=(100,), weightnorm=False, bias = True):
        """
        Most classical form of neural network,
        It takes intput as input, add hidden layers of sizes in layersizes.
        Then add 
        Infos:
            - len(layersizes) is the number of hidden layer
            - layersizes[i] is the size of the ith hidden layer
            - for a linear network, call with layersizes=[]
        :param input; the tensorflow input node
        :param outputsize: size of the output of this layer
        :param layersizes: the sizes of each layer (length= number of layer)
        :param weightnorm: do you want to use weightnormalization ? (see https://arxiv.org/abs/1602.07868)
        :param bias: do you want to add a bias in your computation
        """

        # TODO
        reuse = False
        graphtoload = None
        bias = True

        self.nb_layer = len(layersizes)
        self.layers = []

        z = input
        #hidden layers
        for i, ls in enumerate(layersizes):
            # if size of input = layer size, we still apply guided dropout!
            with tf.variable_scope("dense_layer_{}".format(i)):
                new_layer = DenseLayer(input=z, size=ls, relu=True, bias=bias, guided_dropout_mask=None,
                                       weight_normalization=weightnorm)
            self.layers.append(new_layer)
            z = new_layer.res

        # output layer
        self.output = None
        with tf.variable_scope("last_dense_layer", reuse=reuse) as scope:
            self.output = DenseLayer(input=z, size=outputsize, relu=False, bias=bias,
                                     guided_dropout_mask=None, weight_normalization=weightnorm)
            self.pred = tf.identity(self.output.res, name="output")

    def getnbparam(self):
        """
        :return: the number of trainable parameters of the neural network build 
        """
        res = 0
        for el in self.layers:
            res += el.nbparams
        res += self.output.nbparams
        return res

    def getflops(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition. Results are given for a minibatch of 1 example.
        :return: the number of flops of the neural network build 
        """
        res = 0
        for el in self.layers:
            res += el.flops
        res += self.output.flops
        return res

    def initwn(self, sess):
        """
        Initialize the weights for weight normalization
        :param sess: a tensorflow session
        :return: 
        """
        for el in self.layers:
            el.initwn(sess=sess)
        self.output.initwn(sess=sess)

# class ResidualBlock:
#     def __init__(self, input, params, bias=True, relu=True, numlayer="", k=None, reuse=None, graphtoload=None):
#         """Compute the output of a fully connected layer with 'resNet'
#         W.Relu(W_1.X+bias)+X
#         NB : output same size ass input
#         prev : the previous tensorflow object
#         nIn : size of input
#         nLayer : number of layer between the residual and the input
#         nbVert : number of neurons per 'hidden' layer
#         biais : add bias"""
#
#         # bellow: first resNet implementation
#         # self.relu = relu
#         # with tf.name_scope("resblock"+numlayer) as scope:
#         #     self.w1 = ReluLayer(input, params, bias, numlayer="resblock"+numlayer+"w1", k=k)  # Relu(W_1.X+bias)
#         #     parw = BrickParams(nin=params.nout, nout=params.nin)
#         #     self.W = WeightBias(self.w1.out, parw, False, numlayer="resblock"+numlayer+"w", k=None)
#         #
#         #     if relu:
#         #         self.out = tf.nn.relu(tf.add(self.W.z, input, name="directconn"), name="nonlinearity")
#         #     else:
#         #         self.out = tf.add(self.W.z, input, name="directconn")
#
#         # new resNet implementation (http://link.springer.com/chapter/10.1007/978-3-319-46493-0_38)
#         # https://arxiv.org/pdf/1603.05027.pdf
#         with tf.variable_scope("resblock" + numlayer, reuse=reuse):
#             relu_imp = tf.nn.relu(input, name="relu_input")
#             self.w1 = ReluLayer(relu_imp, params, bias, numlayer="resblock" + numlayer + "w1", k=k,
#                                 reuse=reuse, graphtoload=graphtoload)  # Relu(W_1.X+bias)
#             parw = BrickParams(nin=params.nout, nout=params.nin)
#             with tf.variable_scope('layer' + "resblock" + numlayer + "w", reuse=reuse):
#                 self.W = WeightBias(self.w1.out, parw, False, numlayer="resblock" + numlayer + "w",
#                                     k=None, graphtoload=graphtoload)
#             self.out = tf.add(self.W.z, input, name="directconn")
#
#     def getnbparam(self):
#         return self.w1.getnbparam() + self.W.getnbparam()
#
#     def initlayerwn(self, sess, ph, data):
#         with tf.variable_scope("init_wn_resblok"):
#             tmp = self.w1.initlayerwn(sess, ph=ph, data=data)
#             np.maximum(tmp, 0, tmp)  # the output of layers are relu
#             data_tmp = copy.deepcopy(data)
#             data_tmp[0] = tmp
#             tmp = self.W.initlayerwn(sess=sess, ph=ph, data=data_tmp)
#         return tmp + data[0]
#
# class MicrosoftResNet(TFNeuralNet):
#     def __init__(self, par, ph, paramTypes=BrickParams, defineloss=True, netname="", reuse=None, graphtoload=None):
#         """ Definition of the tensorflow graph
#         for the "resNet" architecture
#         1. Scale the input to have the proper size
#         2. Stack residual block
#         """
#         TFNeuralNet.__init__(self, par, ph)
#
#         z = self.ph.input
#         n = self.ph.nIn
#         n_out = self.ph.nOut
#         self.layers = []
#         self.x_prime = None
#         # scale inputs using linear model (if sizes does not match)
#         if self.ph.nIn != self.ph.nOut:
#             # number of inputs differs from number of output, I scale once, and afterwards use standard ResNet
#             parX = BrickParams(nin=self.ph.nIn, nout=self.ph.nOut, weightnorm=par.params["weightnorm"])
#             with tf.variable_scope("scaling", reuse=reuse):
#                 self.x_prime = WeightBias(input=z, params=parX, bias=False, numlayer="_scale", graphtoload=graphtoload)
#             z = self.x_prime.z
#         else:
#             z = self.ph.input
#         parL = paramTypes(nin=n_out, nout=self.layer_size, weightnorm=par.params["weightnorm"])
#         # stack residual layers
#         for i in range(self.nb_layer-1):
#             if self.ph.k is not None:
#                 # Guided Dropout Case
#                 if self.ph.samemat:
#                     k = self.ph.k
#                 else:
#                     k = self.ph.k[i, :, :]
#             else:
#                 k = None
#             new_layer = ResidualBlock(input=z, params=parL, bias=True, numlayer=str(i), k=k,
#                                       reuse=reuse, graphtoload=graphtoload)
#             self.layers.append(new_layer)
#             z = new_layer.out
#         parL = BrickParams(nin=n_out, nout=self.layer_size, weightnorm=par.params["weightnorm"])
#         # the last layer (output of the network)
#         self.output = ResidualBlock(input=z,
#                                     params=parL,
#                                     bias=True,
#                                     relu=False,
#                                     numlayer="_last",
#                                     reuse=reuse,
#                                     graphtoload=graphtoload)
#         # the predictions
#         with tf.variable_scope('Prediction'):
#             self.pred = self.output.out
#
#         if defineloss:
#             # define loss function
#             with tf.variable_scope('Loss'):
#                 if "l2lambda" in par.params:
#                     self.loss_fun = tf.nn.l2_loss(self.ph.output - self.getoutput(), name="l2_loss")
#                     for lay in self.layers:
#                         self.loss_fun = tf.add(self.loss_fun,
#                                                par.params["l2lambda"]*tf.add(tf.nn.l2_loss(lay.attrs.w),
#                                                                              tf.nn.l2_loss(lay.attrs.b)))
#                 else:
#                     self.loss_fun = tf.nn.l2_loss(self.ph.output - self.getoutput(), name="l2_loss")
#
#             # define error
#             with tf.variable_scope('Error'):
#                 self.error = tf.add(self.pred, -self.ph.output, name=netname+"error_diff")
#                 self.error_abs = tf.abs(self.error)
#                 self.l1_avg = tf.reduce_mean(self.error_abs, name=netname+"l1_avg")
#                 self.l2_avg = tf.reduce_mean(self.error * self.error, name=netname+"l2_avg")
#                 self.l_max = tf.reduce_max(self.error_abs, name=netname+"l_max")
#
#             # add loss as a summary for training
#             sum0 = tf.summary.scalar(netname+"loss", self.loss_fun)
#             sum1 = tf.summary.scalar(netname+"l1_avg", self.l1_avg)
#             sum2 = tf.summary.scalar(netname+"l2_avg", self.l2_avg)
#             sum3 = tf.summary.scalar(netname+"loss_max", self.l_max)
#
#             self.mergedsummaryvar = tf.summary.merge([sum0, sum1, sum2, sum3])
#
#             tf.add_to_collection(NAMESAVEDTFVARS, self.loss_fun)
#             tf.add_to_collection(NAMESAVEDTFVARS, self.pred)
#             tf.add_to_collection(NAMESAVEDTFVARS, self.l1_avg)
#             tf.add_to_collection(NAMESAVEDTFVARS, self.l2_avg)
#             tf.add_to_collection(NAMESAVEDTFVARS, self.l_max)
#
#             tf.add_to_collection("LOSSFUNResNet"+netname, self.loss_fun)
#             tf.add_to_collection("OUTPUTResNet"+netname, self.getoutput())
#
#     def getoutput(self):
#         return self.output.out
#
#     def getlossfun(self):
#         return self.loss_fun
#
#     def getlayersize(self):
#         return self.layer_size
#
#     def run(self, sess, batch_in, batch_out, toberun=None):
#         """ sess : tensorflow session
#         batch_in : input data for current minibatch
#         batch_out : output data for current minibatch
#         toberun : tensorflow 'stuff' that need to be run
#         """
#         if toberun is None:
#             toberun = self.getlossfun()
#         feed_dict = {self.ph.input: batch_in, self.ph.output: batch_out}
#         return sess.run(toberun, feed_dict=feed_dict)
#
#     def initwn(self, sess, data):
#         # init weight normalization with resnet
#         with tf.variable_scope("init_wn_resnet"):
#             if self.ph.nIn != self.ph.nOut:
#                 tmp = self.x_prime.initlayerwn(sess, ph=self.ph, data=data)
#                 np.maximum(tmp, 0, tmp)  # the output of layers are relu
#                 data[0] = tmp
#             datatmp = copy.deepcopy(data)
#             for id, layer in enumerate(self.layers):
#                 # print(data[-1].shape)
#                 if len(data[-1].shape) >=3:
#                     # in this case there is different matrices for each layer
#                     datatmp[-1] = data[-1][id, :, :]
#                 tmp = layer.initlayerwn(sess, ph=self.ph, data=[datatmp[0], 0, datatmp[-1]] )
#                 np.maximum(tmp, 0, tmp)  # the output of layers are relu
#                 datatmp[0] = tmp
#             tmp = self.output.initlayerwn(sess, ph=self.ph, data=datatmp)
#         return tmp
#
#     def getnbparam(self):
#         if self.ph.nIn != self.ph.nOut:
#             res = self.x_prime.getnbparam()
#         else:
#             res = 0
#         for layer in self.layers:
#             res += layer.getnbparam()
#         res += self.output.getnbparam()
#         return res

if __name__ == "__main__":
    # TODO : code some sort of test for guided dropout stuff
    input = np.ndarray([1.,2.,3.,4.,5.],shape=[5,1],dtype="float32")
    output = np.ndarray([10.,10.,10.],shape=[5,1],dtype="float32")
