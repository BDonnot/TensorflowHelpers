import pdb

import tensorflow as tf

from .ANN import NNFully

class ExpGraphOneXOneY:
    def __init__(self, data, var_x_name="input", var_y_name="output", nnType=NNFully, argsNN=(), kwargsNN={},
                 ):
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
    def __init__(self, data, var_x_name={"input"}, var_y_name={"output"}, nnType=NNFully, argsNN=(), kwargsNN={},
                 spec_encoding={}):
        """
        This class can deal with multiple input/output.
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
        :param spec_encoding: any specification for changing variable (callable)
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
        for el in sorted(self.inputname):
            if el in spec_encoding:
                tup += (spec_encoding[el](self.data[el]),)
            else:
                tup += (self.data[el],)
            this_size = int(tup[-1].get_shape()[1])
            self.dimin[el] = (prev, prev+this_size)
            prev += this_size
        self.input = tf.concat(tup, axis=1, name="input_concatenantion")

        # 2. build the output layer
        self.dimout = {}  # to memorize which data goes where
        prev = 0
        tup = tuple()
        for el in sorted(self.outputname):
            tup += (self.data[el],)
            this_size = int(self.data[el].get_shape()[1])
            self.dimout[el] = (prev, prev+this_size)
            prev += this_size
        self.output = tf.concat(tup, axis=1, name="output_concatenantion")

        # 3. build the neural network
        self.nn = nnType(input=self.input,
                         outputsize=int(self.output.get_shape()[1]),
                         *argsNN,
                         **kwargsNN)

        # 4. build structure to retrieve the right information from the concatenated one's
        self.vars_out = {} # dictionnary of output of the NN
        for varn in sorted(self.outputname):
            be, en = self.dimout[varn]
            self.vars_out[varn] = self.nn.pred[:, be:en]
        self.vars_in = {}
        for varn in sorted(self.inputname):
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


class ComplexGraph(ExpGraphOneXOneY):
    def __init__(self, data,
                 outputsize,
                 sizes,
                 var_x_name={"input"}, var_y_name={"output"},
                 nnType=NNFully, argsNN=(), kwargsNN={},
                 encDecNN=NNFully, args_enc=(), kwargs_enc={},
                 args_dec=(), kwargs_dec={},
                 kwargs_enc_dec=None,
                 spec_encoding={},
                 has_vae=False,
                 latent_dim_size=None,
                 latent_hidden_layers=(),
                 latent_keep_prob=None):
        """
        This class can deal with multiple input/output.
        It will first "encode" with a neural network of type "encDecNN" for each input.
        Then concatenate all the outputs to feed the "main" neural network of type "nnType"
        Afterwards, information goes through a decoding process, with neural network of type "encDecNN"

        The value for each can be retrieve with standard methods:
        - self.get_true_output_dict()
        - self.vars_out
        - self.vars_in

        Basically, this should represent the neural network.
        
        This class can have a variational part after the intermediate NN of the "ComplexGraph" model
        The "decoder" of the "ComplexGraph" use the a variational input.
        To compute the latent space, code derived from "https://github.com/tegg89/VAE-Tensorflow" have been used.
        
        :param data: the dictionnary of input tensor data (key=name, value=tensorflow tensor)
        :param var_x_name: iterable: the names of all the input variables
        :param var_y_name: iterable: the name of  all the output variables
        :param nnType: the type of neural network to use
        :param args forwarded to the initializer of neural network
        :param kwargsNN: key word arguments forwarded to the initializer of neural network
        :param encDecNN: class to use to build the neural networks for encoding / decoding
        :param args_enc:
        :param kwargs_enc: 
        :param args_dec:
        :param kwargs_dec: 
        :param kwargs_enc_dec: 
        :param sizes: the size output by the encoder for each input variable. Dictionnary with key: variable names, value: size
        :param outputsize: the output size for the intermediate / main neural network
        :param has_vae: do you want to add a variationnal auto encoder (between the output of the intermediate neural network and the decoders) 
        :param latent_dim_size: the size of the latent space (int)
        :param latent_hidden_layers: the number of hidden layers of the latent space (ordered iterable of integer)
        :param latent_keep_prob: keep probability for regular dropout for the building of the latent space (affect only the mean)
        """

        self.data = data  # the dictionnary of data pre-processed as produced by an ExpData instance
        self.outputname = var_y_name  # name of the output variable, should be one of the key of self.data
        self.inputname = var_x_name
        self.data = data


        if kwargs_enc_dec is not None:
            # TODO make this usage deprecated
            # TODO raise an error if kwargs_enc or kwargs_dec not "empty"
            kwargs_enc = kwargs_enc_dec
            kwargs_dec = kwargs_enc_dec

        # dictionnary of "ground truth" data
        self.true_dataY = {k: self.data[k] for k in self.outputname}
        self.true_dataX = {k: self.data[k] for k in self.inputname}
        self.size_in = 0
        for _,v in self.true_dataX.items():
            self.size_in += int(v.get_shape()[1])

        # 1. build the encodings neural networks
        self.outputEnc = {}
        self.encoders = {}
        self._buildencoders(sizes, spec_encoding, encDecNN, args_enc, kwargs_enc)

        # self.input = tf.zeros(shape=(None, 0), dtype=tf.float32)
        tup = tuple()
        for el in sorted(self.inputname):
            tup += (self.outputEnc[el],)
        self.enc_output = tf.concat(tup, axis=1, name="encoder_output_concatenantion")

        # 3. build the neural network
        self.nn = None
        self._buildintermediateNN(nnType=nnType, argsNN=argsNN, input=self.enc_output, outputsize=outputsize, kwargsNN=kwargsNN)

        # 4. add a variational component if needed
        self.has_vae = has_vae
        if self._have_latent_space():
            self.latent_z = None
            self.latent_dim_size = None # will be set to the proper value (eg not None in self._build_latent_space)
            self._build_latent_space(latent_dim_size, latent_hidden_layers, latent_keep_prob)
            inputdec = self.latent_z
        else:
            self.kld = None
            inputdec = self.nn.pred

        # 5. build the decodings neural networks
        self.outputDec = {}
        self.decoders = {}
        self.size_out = 0
        self._builddecoders(encDecNN, args_dec, kwargs_dec, inputdec=inputdec)

        # 6. build structure to retrieve the right information from the concatenated one's
        self.vars_out = self.outputDec  # dictionnary of output of the NN
        self.vars_in = self.true_dataX

        # 7. create the fields summary and loss that will be created in ExpModel and assign via "self.init"
        self.mergedsummaryvar = None
        self.loss = None


    def initwn(self, sess):
        """
        Initialize the weights for weight normalization
        :param sess: a tensorflow session
        :return: 
        """
        for _,v in self.encoders.items():
            v.initwn(sess=sess)
        self.nn.initwn(sess=sess)
        for _,v in self.decoders.items():
            v.initwn(sess=sess)
            # pdb.set_trace()

    def get_true_output_dict(self):
        """
        :return: the output data dictionnary. key: varname, value=true value of this data
        """
        return self.true_dataY

    def get_input_size(self):
        """
        :return: the number of columns (variables) in input
        """
        return self.size_in

    def get_output_size(self):
        """
        :return: the number of columns (variables) in output
        """
        return self.size_out

    def getnbparam(self):
        """
        :return:  the number of total free parameters of the neural network"""
        res = 0
        for _,v in self.encoders.items():
            res += v.getnbparam()
        res += self.nn.getnbparam()
        for _,v in self.decoders.items():
            res += v.getnbparam()
        return res

    def getflop(self):
        """
        flops are computed using formulas in https://mediatum.ub.tum.de/doc/625604/625604
        it takes into account both multiplication and addition.
        Results are given for a minibatch of 1 example for a single forward pass.
        :return: the number of flops of the neural network build 
        """
        res = 0
        for _,v in self.encoders.items():
            res += v.getflop()
        res += self.nn.getflop()
        for _,v in self.decoders.items():
            res += v.getflop()
        return res

    def _buildencoders(self, sizes, spec_encoding, encDecNN, args_enc, kwargs_enc):
        """
        Build the encoder networks
        :param sizes: 
        :param spec_encoding: 
        :param encDecNN: 
        :param args_enc: 
        :param kwargs_enc: 
        :return: 
        """
        with tf.variable_scope("ComplexGraph_encoding"):
            for varname in sorted(self.inputname):
                with tf.variable_scope(varname):
                    size_out = sizes[varname]
                    if varname in spec_encoding:
                        input_tmp=spec_encoding[varname](self.data[varname])
                    else:
                        input_tmp = self.data[varname]
                    tmp = encDecNN(*args_enc,
                                   input=input_tmp,
                                   outputsize=size_out,
                                   **kwargs_enc)
                    self.encoders[varname] = tmp
                    self.outputEnc[varname] = tmp.pred
                    
    def _builddecoders(self, encDecNN, args_dec, kwargs_dec, inputdec=None):
        """
        Build the decoder networks
        :param sizes: 
        :param encDecNN: 
        :param args_dec: 
        :param kwargs_dec: 
        :return: 
        """
        with tf.variable_scope("ComplexGraph_decoding"):
            if inputdec is None:
                inputdec = self.nn.pred
            for varname in sorted(self.outputname):
                with tf.variable_scope(varname):
                    # size_out = sizes[varname]
                    tmp = encDecNN(*args_dec,
                                   input=inputdec,
                                   outputsize=int(self.data[varname].get_shape()[1]),
                                   **kwargs_dec)
                    self.decoders[varname] = tmp
                    self.outputDec[varname] = tmp.pred
                    self.size_out += int(tmp.pred.get_shape()[1])

    def _buildintermediateNN(self, nnType, argsNN, input, outputsize, kwargsNN):
        """
        
        :param nnType:
        :param argsNN: 
        :param input: 
        :param outputsize: 
        :param kwargsNN: 
        :return: 
        """
        self.nn = nnType(*argsNN,
                         input=input,
                         outputsize=outputsize,
                         **kwargsNN)

    def _build_latent_space(self, latent_dim_size, latent_hidden_layers, latent_keep_prob):
        with tf.variable_scope("ComplexGraph_variational"):
            # inspired from "https://github.com/tegg89/VAE-Tensorflow/blob/master/model.py"
            if latent_dim_size is None:
                latent_dim_size = int(self.nn.pred.get_shape()[1])
            self.latent_dim_size = latent_dim_size
            # build the mean
            self.mu_vae = NNFully(input=self.nn.pred, outputsize=latent_dim_size, resizeinput=False,
                                  layersizes=latent_hidden_layers, kwardslayer={"keep_prob": latent_keep_prob},
                                  name="mu_vae")

            # We build the var of the VAE
            self.logstd_vae = NNFully(input=self.nn.pred, outputsize=latent_dim_size, resizeinput=False,
                                  layersizes=latent_hidden_layers,
                                  name="logstd_vae")
            # sample a N(0,1) same shape as log std
            self.epsilon = tf.random_normal(tf.shape(self.logstd_vae.pred), name='epsilon')
            # get the true std (eg take the exponential
            self.std_vae = tf.exp(.5 * self.logstd_vae.pred)

            # compute the latent variable
            self.latent_z = self.mu_vae.pred + tf.multiply(self.std_vae, self.epsilon)
            # TODO I may have a problem here. kl divergence must be added example by example in a minibatch...
            self.kld = -.5 * tf.reduce_sum(1. + self.logstd_vae.pred - tf.pow(self.mu_vae.pred, 2) - tf.exp(self.logstd_vae.pred),
                                           # reduction_indices=1,
                                           name="kl_divergence")

    def _have_latent_space(self):
        return self.has_vae

    def init(self, mergedsummaryvar, loss):
        """
        Assign the summary 'mergedsummaryvar' for easier access
        :param mergedsummaryvar: the summary of everything to be save by tensorboard
        :param loss: the loss tensor use for training (reconstruction loss) : I need to add the KL-divergence loss if vae is used
        :return:
        """
        self.mergedsummaryvar = mergedsummaryvar
        sqrt_dim_size = tf.sqrt(float(self.latent_dim_size))
        if self._have_latent_space():
            self.loss = tf.add(loss, 1/sqrt_dim_size*self.kld)
        else:
            self.loss = loss
