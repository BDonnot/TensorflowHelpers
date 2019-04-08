import tensorflow as tf

from .Layers import DenseLayer

# TODO refactor it: it should be a "Graph" object. There is something weird between this and the layers
# TODO alternatively, ResidualBlock and VaeBlock etc could also be kind of NNFUlly :-/
# TODO really really weird

class NNFully:
    def __init__(self, input, outputsize, layersizes=(), weightnorm=False, bias=True,
                 layerClass=DenseLayer, kwardslayer={}, resizeinput=False, name=None,
                 output_nonlin=tf.identity, resizeoutput=True):
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
        self.resize_input = resizeinput
        self.resize_output = resizeoutput
        name = name + "_" if name is not None else ""
        # TODO if outputsize != input.get_shape()[1] ?
        # TODO remove that for standard network without residual or dense block !!!
        # pdb.set_trace()
        if resizeinput:
            if outputsize != input.get_shape()[1]:
                # scaling the input linearly to have the proper size,
                # only if sizes does not match
                input_propersize = DenseLayer(input=input, size=outputsize, relu=False,
                                              bias=False, weight_normalization=False,
                                              keep_prob=None, layernum=name + "scaling_proper_input_size")
                self.input_propersize = input_propersize
                self.params_added += input_propersize.get_nb_params()
                self.flop_added += input_propersize.get_flops()
                z = input_propersize.get_res()
            else:
                z = input
        else:
            z = input

        self.nb_layer = len(layersizes)
        self.layers = []

        # hidden layers
        for i, ls in enumerate(layersizes):
            if isinstance(kwardslayer, type({})):
                tmp_kw = kwardslayer
            elif isinstance(kwardslayer, type([])):
                tmp_kw = kwardslayer[i]
            else:
                msg = "NNFully: wrong \"kwardslayer\" type argument speficified. You provided {} "
                msg += "of type {} but only {} (same arguments for every layer) "
                msg += "and {} (different argument for each layer) are understood."
                raise RuntimeError(msg.format(kwardslayer, type(kwardslayer), type({}), type([])))

            try:
                new_layer = layerClass(input=z, size=ls, relu=True, bias=bias,
                                       weight_normalization=weightnorm, layernum=name + str(i), **tmp_kw)
            except TypeError as e:
                msg = "tensorflowHelper.ANN: Impossible to build layer of class {} with arguments provided. "
                msg += "Have you checked all keys of \"kwardslayer\" ({}) where keyword argument of that class ?"
                print(msg.format(layerClass, tmp_kw.keys()))
                print(e)
                raise e

            self.layers.append(new_layer)
            z = new_layer.get_res()

        # output layer
        self.output = None
        self.pred = None
        if self.resize_output:
            # with tf.variable_scope("last_dense_layer", reuse=reuse):
            self.output = DenseLayer(input=z, size=outputsize, relu=False, bias=bias,
                                     weight_normalization=weightnorm,
                                     layernum=name + "last", keep_prob=None)
            self.pred = output_nonlin(self.output.get_res(), name="output")
        else:
            self.output = self.layers[-1]
            self.pred = output_nonlin(self.output.get_res(), name="output")

    def getnbparam(self):
        """
        :return: the number of trainable parameters of the neural network build
        """
        res = self.params_added
        for el in self.layers:
            res += el.get_nb_params()
        if self.resize_output:
            res += self.output.get_nb_params()
        return res

    def getflop(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition. Results are given for a minibatch of 1 example.
        :return: the number of flops of the neural network build
        """
        res = self.flop_added
        for el in self.layers:
            res += el.get_flops()
        if self.resize_output:
            res += self.output.get_flops()
        return res

    def initwn(self, sess):
        """
        Initialize the weights for weight normalization
        :param sess: a tensorflow session
        :return:
        """
        for el in self.layers:
            el.initwn(sess=sess)
        if self.resize_output:
            self.output.initwn(sess=sess)

    def tell_epoch(self, sess, epochnum):
        if self.resize_input:
            self.input_propersize.tell_epoch(sess, epochnum)
        for el in self.layers:
            el.tell_epoch(sess, epochnum)
        if self.resize_output:
            self.output.tell_epoch(sess, epochnum)

    def start_train(self, sess):
        if self.resize_input:
            self.input_propersize.start_train(sess)
        for el in self.layers:
            el.start_train(sess)
        if self.resize_output:
            self.output.start_train(sess)

    def start_test(self, sess):
        if self.resize_input:
            self.input_propersize.start_test(sess)
        for el in self.layers:
            el.start_test(sess)
        if self.resize_output:
            self.output.start_test(sess)

    def add_loss(self, previous_loss):
        loss = previous_loss
        if self.resize_input:
            loss = self.input_propersize.add_loss(loss)
        for el in self.layers:
            loss = el.add_loss(loss)
        if self.resize_output:
            loss = self.output.add_loss(loss)
        return loss

    def startexp(self, sess):
        if self.resize_input:
            self.input_propersize.start_exp(sess)
        for el in self.layers:
            el.start_exp(sess)
        if self.resize_output:
            self.output.start_exp(sess)