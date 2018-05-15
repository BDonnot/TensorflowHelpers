import copy
import pdb

import numpy as np
import tensorflow as tf

DTYPE_USED=tf.float32

class DenseLayer:
    def __init__(self, input, size, relu=False, bias=True, weight_normalization=False,
                 keep_prob=None, layernum=0):
        """
        for weight normalization see https://arxiv.org/abs/1602.07868
        for counting the flops of operations see https://mediatum.ub.tum.de/doc/625604/625604
        :param input: input of the layer 
        :param size: layer size (number of outputs units)
        :param relu: do you use relu ?
        :param bias: do you add bias ?
        :param guided_dropconnect_mask: tensor of the mask matrix  #TODO 
        :param weight_normalization: do you use weight normalization (see https://arxiv.org/abs/1602.07868)
        :param keep_prob: a scalar tensor for dropout layer (None if you don't want to use it)
        :param layernum: number of layer (this layer in the graph)
        """
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
                                initializer=tf.contrib.layers.xavier_initializer(dtype=DTYPE_USED, uniform=False),
                                # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weights_matrix:0"),
                                trainable=True)  # weight matrix
            self.nbparams += int(nin_ * size)

            if weight_normalization:
                self.weightnormed = True
                self.g = tf.get_variable(shape=[size],
                                    name="weight_normalization_g",
                                    initializer=tf.constant_initializer(value=1.0, dtype=DTYPE_USED),
                                    # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/weight_normalization_g:0"),
                                    trainable=True)
                self.nbparams += int(size)
                self.scaled_matrix = tf.nn.l2_normalize(self.w_, dim=0, name="weight_normalization_scaled_matrix")
                self.flops += size*(2*nin_-1) # clomputation of ||v|| (size comptuation of inner product of vector of size nin_)
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
                                         initializer=tf.constant_initializer(value=0.0, dtype=DTYPE_USED),
                                         # initializer=tf.get_default_graph().get_tensor_by_name(tf.get_variable_scope().name+"/bias:0"),
                                         name="bias",
                                         trainable=True)
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
                res = tf.nn.dropout(self.res, keep_prob=keep_prob, name="applying_dropout")
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



class ResidualBlock:
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

        self.input = input
        self.weightnormed = weight_normalization

        self.res = input
        self.flops = 0
        self.nbparams = 0

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
                self.nbparams += self.first_layer.nbparams
                self.res = self.first_layer.res

            # treating "-> W_2 * . "
            self.second_layer = None
            with tf.variable_scope("resBlock_second_layer"):
                self.second_layer = layerClass(self.res, int(input.get_shape()[1]), relu=False, bias=bias,
                                               weight_normalization=weight_normalization,
                                               keep_prob=None,
                                               **kwardslayer)
                self.flops += self.second_layer.flops
                self.nbparams += self.second_layer.nbparams

            # treating "-> X + ."

            if keep_prob is not None:
                #TODO : copy pasted from DenseLayer
                tmp = tf.nn.dropout(self.second_layer.res, keep_prob=keep_prob, name="applying_dropout")
                # we consider that generating random number count for 1 operation
                self.flops += size  # generate the "size" real random numbers
                self.flops += size  # building the 0-1 vector of size "size" (thresholding "size" random values)
                self.flops += size  # element wise multiplication with res
            else:
                tmp = self.second_layer.res

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


class DenseBlock:
    def __init__(self, input, relu=False, bias=True, weight_normalization=False,
                 keep_prob=None, nblayer=2, layernum=0, size=0,
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
        self.input = input
        self.weightnormed = weight_normalization
        size = int(input.get_shape()[1])
        self.res = input
        self.flops = 0
        self.nbparams = 0
        self.layers = []

        with tf.variable_scope("dense_block_{}".format(layernum)):
            for i in range(nblayer):
                tmp_layer = layerClass(self.res, size, relu=True, bias=bias,
                                       weight_normalization=weight_normalization,
                                       keep_prob=None, layernum=i,
                                       **kwardslayer)
                self.flops += tmp_layer.flops
                self.nbparams += tmp_layer.nbparams
                self.res = tmp_layer.res
                for l in self.layers:
                    self.res = self.res + l.res
                    self.flops += size
                self.layers.append(tmp_layer)

            if relu:
                self.res = tf.nn.relu(self.res, name="applying_relu")
                self.flops += size  # we consider relu of requiring 1 computation per number (one max)

            if keep_prob is not None:
                #TODO : copy pasted from DenseLayer
                self.res = tf.nn.dropout(self.res, keep_prob=keep_prob, name="applying_dropout")
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


class NNFully:
    def __init__(self, input, outputsize, layersizes=(), weightnorm=False, bias=True,
                 layerClass=DenseLayer, kwardslayer={}, resizeinput=False, name=None,
                 output_nonlin=tf.identity):
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
        :param layersizes: iterable, the sizes of each hidden layer (length: number of layer)
        :param weightnorm: do you want to use weightnormalization ? (see https://arxiv.org/abs/1602.07868)
        :param bias: do you want to add a bias in your computation
        :param layerClass: type of layer you want to use (DenseLayer, ResidualBlock or DenseBlock for example)
        :param kwardslayer: keyword arguments forwarded when building the layer networks compatible with class "layerClass"
        :param resizeinput: do you want to scale, prior to do any computation, your input to have the same size as your output
        :param name: a name (optional) for this 
        :param output_nonlin: non linearity at the output of the neural network
        
        if kwardslayer is a dcitionnary, the same "kwardslayer" will be used for alllayer. If you want specific
        "kwardslayer" per layer, use a list of the same size of "layersizes"
        """

        self.params_added = 0
        self.flop_added = 0
        name = name+"_" if name is not None else ""
        # TODO if outputsize != input.get_shape()[1] ?
        # TODO remove that for standard network without residual or dense block !!!
        if resizeinput:
            if outputsize != input.get_shape()[1]:
                # scaling the input linearly to have the proper size,
                # only if sizes does not match
                input_propersize = DenseLayer(input=input, size=outputsize, relu=False,
                                              bias=False, weight_normalization=False,
                                              keep_prob=None, layernum=name+"scaling_proper_input_size")
                self.params_added += input_propersize.nbparams
                self.flop_added += input_propersize.flops
                z = input_propersize.res
            else:
                z = input
        else:
            z = input

        self.nb_layer = len(layersizes)
        self.layers = []

        #hidden layers
        for i, ls in enumerate(layersizes):
            if isinstance(kwardslayer, type({})):
                tmp_kw = kwardslayer
            elif isinstance(kwardslayer, type([])):
                tmp_kw = kwardslayer[i]
            else:
                msg = "NNFully: wrong \"kwardslayer\" type argument speficified. You provided {} "
                msg += "of type {} but only {} (same arguments for every layer) "
                msg +="and {} (different argument for each layer) are understood."
                raise RuntimeError(msg.format(kwardslayer, type(kwardslayer), type({}), type([])))
            new_layer = layerClass(input=z, size=ls, relu=True, bias=bias,
                                       weight_normalization=weightnorm, layernum=name+str(i), **tmp_kw)
            self.layers.append(new_layer)
            z = new_layer.res

        # output layer
        self.output = None
        self.pred = None
        # with tf.variable_scope("last_dense_layer", reuse=reuse):
        self.output = DenseLayer(input=z, size=outputsize, relu=False, bias=bias,
                                 weight_normalization=weightnorm,
                                 layernum=name+"last", keep_prob=None)
        self.pred = output_nonlin(self.output.res, name="output")

    def getnbparam(self):
        """
        :return: the number of trainable parameters of the neural network build 
        """
        res = self.params_added
        for el in self.layers:
            res += el.nbparams
        res += self.output.nbparams
        return res

    def getflop(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition. Results are given for a minibatch of 1 example.
        :return: the number of flops of the neural network build 
        """
        res = self.flop_added
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

if __name__ == "__main__":
    # TODO : code some sort of test for guided dropout stuff
    input = np.ndarray([1.,2.,3.,4.,5.],shape=[5,1],dtype="float32")
    output = np.ndarray([10.,10.,10.],shape=[5,1],dtype="float32")
