import pdb

import tensorflow as tf

from .DefANN import NNFully

class ExpGraphOneXOneY:
    def __init__(self, data, var_x_name="input", var_y_name="output", nnType=NNFully, argsNN=(), kwargsNN={}):
        """The base class for every 'Graph' subclass to be use in with Experiment.

        /!\ This class works only with one input variable, and one output variable. /!\

        Basically, this should represent the neural network.
        :param data: the dictionnary of input tensor data (key=name, value=tensorflow tensor)
        :param var_x_name: the name of the input variable
        :param var_y_name: the name of the output variable
        :param nnType: the type of neural network to use
        :param args forwarded to the initializer of neural network
        :param kwargsNN: key word arguments forwarded to the initializer of neural network
        """

        self.data = data  # the dictionnary of data pre-processed as produced by an ExpData instance
        self.outputname = (var_y_name,)  # name of the output variable, should be one of the key of self.data
        self.inputname = (var_x_name,)

        self.nn = nnType(
            input=data[var_x_name],
            outputsize=int(data[var_y_name].get_shape()[1]),
            *argsNN,
            **kwargsNN)
        self.vars_out = {self.outputname[0]: self.nn.pred}

        self.mergedsummaryvar = None
        self.loss = None

    def getnbparam(self):
        """
        :return:  the number of total free parameters of the neural network"""
        return self.nn.getnbparam()

    def getflop(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition.
        Results are given for a minibatch of 1 example for a single forward pass.
        :return: the number of flops of the neural network build 
        """
        return self.nn.getflop()

    def getoutput(self):
        """
        :return: a dictionnray corresponding to the output variables. keys; variables names, values: the tensor of the forward pass
        """
        return self.vars_out

    def run(self, sess, toberun):
        """
        Use the tensorflow session 'sess' to run the graph node 'toberun' with data 'data'
        :param sess: a tensorflow session
        :param toberun: a node in the tensorflow computation graph to be run...
        :return:
        """
        return sess.run(toberun)

    def init(self, mergedsummaryvar, loss):
        """
        Assign the summary 'mergedsummaryvar' for easier access
        :param mergedsummaryvar: the summary of everything to be save by tensorboard
        :param loss: the loss tensor use for training
        :return:
        """
        self.mergedsummaryvar = mergedsummaryvar
        self.loss = loss

    def initwn(self, sess):
        """
        Initialize the weights for weight normalization
        :param sess: a tensorflow session
        :return: 
        """
        self.nn.initwn(sess=sess)

    def get_true_output_dict(self):
        """
        :return: the output data dictionnary. key: varname, value=true value of this data
        """
        return {self.outputname[0]: self.data[self.outputname[0]]}

    def get_input_size(self):
        """

        :return: the number of columns (variables) in input
        """
        return int(self.data[self.inputname[0]].shape[1])

    def get_output_size(self):
        """

        :return: the number of columns (variables) in output
        """
        return int(self.data[self.outputname[0]].shape[1])

    def run_with_feeddict(self, sess, toberun, data=None):
        """
        Use the tensorflow session 'sess' to run the graph node 'toberun' with data 'data'
        :param sess: a tensorflow session
        :param toberun: a node in the tensorflow computation graph to be run...
        :param data: the data set to be used.
        :return:
        """
        fd = {}
        for id, el in self.phs:  # TODO Here !!!
            fd[el] = data[id]
        return sess.run(toberun, feed_dict=fd)


class ExpGraph(ExpGraphOneXOneY):
    def __init__(self, data, var_x_name={"input"}, var_y_name={"output"}, nnType=NNFully, argsNN=(), kwargsNN={}):
        """The base class for every 'Graph' subclass to be use in with Experiment.

        This class works can deal with multiple input/output.
        By default, it concatenate all the input variables in one vector, and does the same for the output.
        For example, the underlying neural network will not see the difference between the data type.
        
        The value for each can be retrieve with standard methods:
        - self.get_true_output_dict()
        - self.vars_out
        - self.vars_in

        Basically, this should represent the neural network.
        :param data: the dictionnary of input tensor data (key=name, value=tensorflow tensor)
        :param var_x_name: iterable: the names of all the input variables
        :param var_y_name: iterable: the name of  all the output variables
        :param nnType: the type of neural network to use
        :param args forwarded to the initializer of neural network
        :param kwargsNN: key word arguments forwarded to the initializer of neural network
        """

        self.data = data  # the dictionnary of data pre-processed as produced by an ExpData instance
        self.outputname = var_y_name  # name of the output variable, should be one of the key of self.data
        self.inputname = var_x_name
        self.data = data

        # dictionnary of "ground truth" data
        self.true_dataY = {k : self.data[k] for k in self.outputname}

        # 1. build the input layer
        # self.input = tf.zeros(shape=(None, 0), dtype=tf.float32)
        self.dimin = {}  # to memorize which data goes where
        prev = 0
        tup = tuple()
        for el in self.inputname:
            tup += (self.data[el],)
            this_size = int(self.data[el].get_shape()[1])
            self.dimin[el] = (prev, prev+this_size)
            prev += this_size
        self.input = tf.concat(tup, axis=1, name="input_concatenantion")

        # 2. build the output layer
        self.dimout = {}  # to memorize which data goes where
        prev = 0
        tup = tuple()
        for el in self.outputname:
            tup += (self.data[el],)
            this_size = int(self.data[el].get_shape()[1])
            self.dimout[el] = (prev, prev+this_size)
            prev += this_size
        self.output = tf.concat(tup, axis=1, name="output_concatenantion")

        # 3. build the neural network
        self.nn = nnType(
            input=self.input,
            outputsize=int(self.output.get_shape()[1]),
            *argsNN,
            **kwargsNN)

        # 4. build structure to retrieve the right information from the concatenated one's
        self.vars_out = {} # dictionnary of output of the NN
        for varn in self.outputname:
            be, en = self.dimout[varn]
            self.vars_out[varn] = self.nn.pred[:, be:en]
        self.vars_in = {}
        for varn in self.inputname:
            be, en = self.dimin[varn]
            self.vars_in[varn] = self.input[:, be:en]

        # 5. create the fields summary and loss that will be created in ExpModel and assign via "self.init"
        self.mergedsummaryvar = None
        self.loss = None

    def get_true_output_dict(self):
        """
        :return: the output data dictionnary. key: varname, value=true value of this data
        """
        return self.true_dataY

    def get_input_size(self):
        """
        :return: the number of columns (variables) in input
        """
        return int(self.input.get_shape()[1])

    def get_output_size(self):
        """
        :return: the number of columns (variables) in output
        """
        return int(self.output.get_shape()[1])