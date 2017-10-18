import os
import time
import re
import copy
import sys
import json
import socket
import datetime
import shutil
import random
import logging
import pdb

import numpy as np
# from tqdm import tqdm
import tensorflow as tf

from .DefANN import NNFully
from .DefDataHandler import ExpData

TRAINING_COLLECTION_NAME = "train_op"
NAMESAVEDTFVARS = 'savedvars'
NAME_PH_SAVED = "placeholder"

class Path:
    def __init__(
            self,
            path_IEEE=None,
            path_scripts=None,
            path_data=None,
            path_save=None,
            path_save_data=None):
        self.path_IEEE = path_IEEE
        self.path_scripts = path_scripts
        self.path_data = path_data
        self.path_save = path_save
        self.path_save_data = path_save_data
        if path_scripts is not None:
            self.path_logger = os.path.join(path_scripts, "PyHades2", "log")
        else:
            self.path_logger = path_scripts

    def initfromdit(self, dicts, name):
        dico = dicts[name]
        if "path_IEEE" in dico.keys():
            self.path_IEEE = dico["path_IEEE"]
        if "path_scripts" in dico.keys():
            self.path_scripts = dico["path_scripts"]
        if "path_data" in dico.keys():
            self.path_data = dico["path_data"]
        if "path_save" in dico.keys():
            self.path_save = dico["path_save"]
        if "path_save_data" in dico.keys():
            self.path_save_data = dico["path_save_data"]
        if "path_logger" in dico.keys():
            self.path_logger = dico["path_logger"]


class TFWriters:
    def __init__(self, trainname, valname, minibatchname, saver=None):
        """Handle different kind of writer for usage in tensorflow
           By default save 3 infos:
            -minibatch: the last loss computed
            -train; error on the whole training set
            -validation: error on the whole validation set
        :param trainname: the name displayed for the "training" writer
        :param valname: the name displayed for the "valname" writer
        :param minibatchname: the name displayed for the "minibatch" writer
        """

        # save the minibatch info
        self.minibatchname = minibatchname
        self.minibatchwriter = tf.summary.FileWriter(
            minibatchname, graph=tf.get_default_graph())

        # save the training set info
        self.trainname = trainname
        self.trainwriter = tf.summary.FileWriter(
            trainname, graph=tf.get_default_graph())

        # save the validation set info
        self.valname = valname
        self.valwriter = tf.summary.FileWriter(
            valname, graph=tf.get_default_graph())

        # saver of the graph
        self.saver = saver if saver is not None else tf.train.Saver(
            max_to_keep=None)

        # TODO idee : mettre en argument une liste (nomsaver / dataset), et quand on appelerait 'save' ca calculerait,
        # TODO pour chaque element de cette liste, l'erreur sur le datset et le
        # TODO sauvegarderait au bon endroit.


class ExpLogger:
    MAXNUMBATCHPEREPOCH = int(1e6)  # static member

    def __init__(
            self,
            path,
            params,
            name_model,
            logger,
            nameSaveLearning,
            saver,
            num_savings,
            num_savings_minibatch,
            epochsize,
            saveEachEpoch):
        """
        Logger that will log the data in tensorboard as well as in a log file, to be used in the Experiment class.
        It will also have the possibility to save and restore tensorflow models
        :param path: the complete path of the experiment
        :param params:
        :param name_model:
        :param logger: (put None if you don't want it): text logger to use
        :param nameSaveLearning: path where tensorflow data will be stored (for use in tensorboard)
        :param saver: (put None if you don't want it): tensorflow saver to use
        :param num_savings: number of times training and validation set error will be computed and logged
        :param num_savings_minibatch: number of times error for the minibatches will be saved during the learning
        :param epochsize: size of the epoch (1 epoch = 1 pass through all the training set)
        :param saveEachEpoch: do you want to force saving of error for the whole validation set (and training set) at each end of epoch
        :param filesavenum: an addition file where some data will be store (time and number of minibatches by default)
        """

        # logging with text files (usefull when you don't want to / can't use tensorboard
        # (minimal stuff logged here)
        logsuffix = "logger.log"

        self.params = params
        if not os.path.exists(path):
            os.mkdir(path)
        self.loggername = os.path.join(os.path.split(path)[0], logsuffix)
        if logger is None:
            self.logger = logging.getLogger("{}{}".format(params, time.time()))
            self.logger.setLevel(logging.INFO)
            self.formatter = logging.Formatter(
                '%(asctime)s :: %(levelname)s :: %(message)s')
            self.file_handler = logging.FileHandler(
                self.loggername, encoding="utf-8")
            self.file_handler.setLevel(logging.INFO)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
        else:
            self.logger = logger

        # tensorflow wirter / saver / restorer
        self.tfwriter = TFWriters(
            trainname=os.path.join(
                path, "Train"), valname=os.path.join(
                path, "Val"), minibatchname=os.path.join(
                path, "Minibatch"), saver=saver)

        self.path_saveinfomini = os.path.join(path, "minibatch.count")
        self.filesavenum = open(self.path_saveinfomini, "a")
        self.params = params
        self.epochsize = epochsize
        self.saveEachEpoch = saveEachEpoch

    def logtf(self, minibatchnum, graph, data, sess, forcesaving=False):
        """
        Compute the validation set error, and the training set error and the minibatch error, and store them in tensorboard
        :param minibatchnum: current number of minibatches proccessed
        :param graph: an object of class 'ExpGraph' or one of its derivatives
        :param data: an object of class 'ExpData' or one of its derivatives
        :param sess: a tensorflow session to execute the code
        :param forcesaving: do you want to save this data absolutely ?
        :return:
        """

        global_step = int(
            minibatchnum *
            1. /
            self.epochsize *
            self.MAXNUMBATCHPEREPOCH)

        computed = False
        error_nan = False
        loss_ = np.NaN

        if forcesaving or (
                minibatchnum %
                self.params.save_minibatch_loss == 0):
            computed = True
            # save the minibatch info during training
            summary, loss_ = graph.run(
                sess, toberun=[
                    graph.mergedsummaryvar, graph.loss])
            self.tfwriter.minibatchwriter.add_summary(summary, global_step)

            if self.filesavenum is not None:
                self.filesavenum.write(
                    "{}::{}\n".format(
                        minibatchnum,
                        datetime.datetime.now()))
            self.logger.info(
                "Last seen l2 error after {} minibatches : {}".format(
                    minibatchnum, loss_))
            error_nan = not np.isfinite(loss_)
            if error_nan:
                # do not bother to compute training and validation set error if
                # the loss is nan.
                return computed, True, loss_

        if forcesaving or (
                self.saveEachEpoch and minibatchnum %
                self.epochsize == 0) or (
                minibatchnum %
                self.params.save_loss == 0):
            computed = True
            error_nan, loss_ = data.computetensorboard(
                sess=sess, writers=self, graph=graph, xval=global_step, minibatchnum=minibatchnum)

        return computed, error_nan, loss_

    def logtf_with_feeddict(
            self,
            minibatchnum,
            graph,
            data,
            data_minibatch,
            sess,
            forcesaving=False):
        """
        Compute the validation set error, and the training set error and the minibatch error, and store them in tensorboard
        :param minibatchnum: current number of minibatches proccessed
        :param graph: an object of class 'ExpGraph' or one of its derivatives
        :param data: an object of class 'ExpData' or one of its derivatives
        :param data_minibatch: #TODO
        :param sess: a tensorflow session to execute the code
        :param forcesaving: do you want to save this data absolutely ?
        :return:
        """

        global_step = int(
            minibatchnum *
            1. /
            self.epochsize *
            self.MAXNUMBATCHPEREPOCH)

        computed = False
        error_nan = False
        loss_ = np.NaN

        if forcesaving or (
                minibatchnum %
                self.params.save_minibatch_loss == 0):
            if data_minibatch is None:
                return computed, error_nan, loss_

            computed = True
            # save the minibatch info during training
            summary, loss_ = graph.run(
                sess, toberun=[
                    graph.mergedsummaryvar, graph.getloss], data=data_minibatch)
            self.tfwriter.minibatchwriter.add_summary(summary, global_step)

            if self.filesavenum is not None:
                self.filesavenum.write(
                    "{}::{}\n".format(
                        minibatchnum,
                        datetime.datetime.now()))
            self.logger.info(
                "Last seen l2 error after {} minibatches : {}".format(
                    minibatchnum, loss_))
            error_nan = not np.isfinite(loss_)
            if error_nan:
                # do not bother to compute training and validation set error if
                # the loss is nan.
                return computed, True, loss_

        if forcesaving or (
                self.saveEachEpoch and minibatchnum %
                self.epochsize == 0) or (
                minibatchnum %
                self.params.save_loss == 0):
            computed = True
            error_nan, loss_ = data.computetensorboard(
                sess=sess, writers=self, graph=graph, xval=global_step, minibatchnum=minibatchnum)

        return computed, error_nan, loss_

    def normalize_loss(self, loss, nsample):
        """
        normalize the loss depending on the number of samples
        :param loss: the loss computed that will be normalized
        :param nsample: the number of samples for which this is computed
        :return: the normalized loss.
        """
        return loss  # TODO normalize it properly!

    def info(self, str):
        """
        :param str: what you want to log (info level)
        :return:
        """
        self.logger.info(str)

    def savetf(self, sess, fullpath, global_step=None):
        """
        Save the current tensorflow graph
        :param sess: the tensorflow session
        :param fullpath: the full path of where the computation graph is saved
        :param global_step: the total number of training steps computed
        :return:
        """
        self.tfwriter.saver.save(sess, fullpath, global_step=global_step)

    def restoretf(self, sess, path=None, name="ModelTrained_best"):
        """
        Restore the tensorflow graph previously saved
        :param sess: the current tensorflow session
        :param path: the path of the Experience have been save (not the path where the saver operates!)
        :param name: (optionnal) name of the model to restore
        :return:
        """
        if path is None:
            path = self.params.path_saver
        self.tfwriter.saver.restore(sess, os.path.join(path, name))


class ExpGraph:
    def __init__(self, data, var_x_name="input", var_y_name="output", nnType=NNFully, argsNN=(), kwargsNN={}):
        """The base class for every 'Graph' subclass to be use in with Experiment.
        This class works only with one input variable, and one output variable.
        Basically, this should represent the neural network.
        :param data: the dictionnary of input tensor data (key=name, value=tensorflow tensor)
        :param var_x_name: the name of the input variable
        :param var_y_name: the name of the output variable
        :param nnType: the type of neural network to use
        :param args forwarded to the initializer of neural network
        :param kwargsNN: key word arguments forwarded to the initializer of neural network
        """

        self.data = data  # the dictionnary of data pre-processed as produced by an ExpData instance
        self.outputname = var_y_name  # name of the output variable, should be one of the key of self.data
        self.intputname = var_x_name

        self.nn = nnType(
            input=data[var_x_name],
            outputsize=int(data[var_y_name].get_shape()[1]),
            *argsNN,
            **kwargsNN)
        self.vars_out = self.nn.pred
        self.data = data

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
        :return: The "last" node of the graph, that serves as output
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

    def get_true_output_tensor(self):
        """
        :return: the output data tensor (target of the optimizer)
        """
        return self.data[self.outputname]

    def get_input_size(self):
        """
        
        :return: the number of columns (variables) in input
        """
        return int(self.data[self.intputname].shape[1])

    def get_output_size(self):
        """

        :return: the number of columns (variables) in input
        """
        return int(self.data[self.outputname].shape[1])

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

class ExpSaverParam:
    def __init__(self,
                 path=".",
                 pathdata=".",
                 params=None,
                 logger=None,
                 name_exp="MyExp",
                 saver=None,
                 num_savings=100,
                 num_savings_minibatch=500,
                 num_savings_model=20,
                 epochsize=1,
                 saveEachEpoch=False,
                 minibatchnum=0,
                 batch_size=1,
                 num_epoch=1):
        """
        Save the loss (full training / test set) each "self.save_loss" minibatches
        Save the loss (minibatch) each "self.num_savings_minibatch" minibatches
        Save the model each "self.save_model" minibatches

        :param path: the path where the experiment should be stored
        :param pathdata: The path where the data is stored
        :param params: the parameters of the experiments (learning rate etc.)
        :param logger: the text logger to use (default: None)
        :param name_exp: The name you want your experiment to have
        :param saver: the tensorflow saver obejct to use (None if not)
        :param num_savings: the number of time you save the error on total training set and total validation set
        :param num_savings_minibatch: the number of times you save the error on the last minibatches computed
        :param num_savings_model: number of times the models are saved
        :param epochsize: the size of an epoch in terms of minibatches
        :param saveEachEpoch: do you want to force the savings at each epoch
        :param minibatchnum: the number of minibatches computed
        :param batch_size: the size of one training minibatch
        :param num_epoch: the number of epoch to run
        # :param name_model: the model name
        """
        self.path = path
        self.pathdata = pathdata
        self.params = params
        self.logger = logger
        self.nameSaveLearning = name_exp
        self.saver = saver
        self.num_savings = num_savings
        self.save_loss = 0
        self.num_savings_minibatch = num_savings_minibatch
        self.save_minibatch_loss = 0
        self.num_savings_model = num_savings_model
        self.save_model = 0
        self.epochsize = epochsize
        self.saveEachEpoch = saveEachEpoch
        self.name_model = name_exp
        self.minibatchnum = minibatchnum
        self.batch_size = batch_size
        self.name_exp_with_path = os.path.join(self.path, self.name_model)
        self.path_saver = os.path.join(self.name_exp_with_path, "TFInfo")
        self.num_epoch = num_epoch

    def initsizes(self, nrows):
        """
        Initialize, with the dataset sizes, the proper number of savings
        :param nrows: total number of rows of the training dataset
        :return:
        """
        self.epochsize = nrows // self.batch_size  # number of minibatches per epoch
        total_minibatches = self.epochsize * self.num_epoch  # total number of minibatches
        # save the loss each "self.save_loss" minibatches
        self.save_loss = total_minibatches // self.num_savings
        self.save_minibatch_loss = total_minibatches // self.num_savings_minibatch
        self.save_model = total_minibatches // self.num_savings_model


class ExpModel:
    def __init__(self,
                 exp_params,
                 data, graph,
                 lossfun=tf.nn.l2_loss,
                 optimizerClass=tf.train.AdamOptimizer, optimizerkwargs={},
                 netname=""):
        """ init the model with hyper-parameters etc
        add the loss and the optimizer to the pure definition of the neural network.
        The NN is defined by ExpGraph.
        Here the saver / restorer is defined as well.
        :param exp_params: an object of class ExpSaverParam
        # :param input: a list of input placeholder, obtained from ExpData  #TODO
        # :param output: a list of outputs placeholder, obtained from ExpData  #TODO
        :param data: an object of class ExpData or one of its derivative. The data used for the computation
        :param graph: an object of class ExpGraph or one of its derivative. The NN used for the computation.
        :param lossfun: the loss function to use
        :param optimizerClass: the class optimizer to use (tf.train.optimizer)
        :param optimizerkwargs: the key-words arguments to build the optimizer. You can pass the learning rate here.
        # :param graphType: the type of "ExpGraph" to use
        # :param pars: the dictionnary of hyper parameters of the mdoels
        """

        self.lossfun = lossfun
        self.optimizerClass = optimizerClass
        self.exp_params = exp_params

        # 1. store the graph and the data class
        self.graph = graph
        self.data = data
        self.exp_params.initsizes(self.data.getnrows())

        # 2. build some important node: the inference, loss and optimizer node
        self.inference = tf.identity(graph.getoutput(), name="inference")
        true_output_tensor = self.graph.get_true_output_tensor()
        self.loss = None
        with tf.variable_scope("training_loss"):
            self.loss = self.lossfun(
                self.inference-true_output_tensor,
                name="loss")

        self.optimize=None
        with tf.variable_scope("optimizer"):
            self.optimize = optimizerClass(
                **optimizerkwargs).minimize(loss=self.loss, name="optimizer")

        # 3. build the summaries that will be stored
        with tf.variable_scope("summaries"):
            self.error = tf.add(self.inference, -true_output_tensor, name=netname + "error_diff")
            self.error_abs = tf.abs(self.error, name=netname + "error_abs")
            self.l1_avg = tf.reduce_mean( self.error_abs, name=netname + "l1_avg")
            self.l2_avg = tf.reduce_mean(self.error * self.error, name=netname + "l2_avg")
            self.l_max = tf.reduce_max(self.error_abs, name=netname + "l_max")

            # add loss as a summary for training
            sum0 = tf.summary.scalar(netname + "loss", self.loss)
            sum1 = tf.summary.scalar(netname + "l1_avg", self.l1_avg)
            sum2 = tf.summary.scalar(netname + "l2_avg", self.l2_avg)
            sum3 = tf.summary.scalar(netname + "loss_max", self.l_max)

            self.mergedsummaryvar = tf.summary.merge([sum0, sum1, sum2, sum3])

            tf.add_to_collection(NAMESAVEDTFVARS, self.loss)
            tf.add_to_collection(NAMESAVEDTFVARS, self.inference)
            tf.add_to_collection(NAMESAVEDTFVARS, self.l1_avg)
            tf.add_to_collection(NAMESAVEDTFVARS, self.l2_avg)
            tf.add_to_collection(NAMESAVEDTFVARS, self.l_max)

            tf.add_to_collection("LOSSFUNFully" + netname, self.loss)
            tf.add_to_collection("OUTPUTFully" + netname, self.inference)

            self.graph.init(self.mergedsummaryvar, self.loss)

        # 4. create the saver object
        self.explogger = ExpLogger(
            path=exp_params.path_saver,
            params=exp_params,
            logger=exp_params.logger,
            nameSaveLearning=exp_params.nameSaveLearning,
            saver=exp_params.saver,
            num_savings=exp_params.num_savings,
            num_savings_minibatch=exp_params.num_savings_minibatch,
            epochsize=exp_params.epochsize,
            saveEachEpoch=exp_params.saveEachEpoch,
            name_model=exp_params.name_model)

    def getlossfun(self):
        """
        :return : the loss function of the computation graph created"""
        return self.lossfun

    def run(self, sess):
        """
        Run one single minibatch.
        If it is time, store the error made on that minibatch
        If it is time, store the error made on the training / validation set
        :param sess: a tensorflow session to execute the computation
        :return:
        """
        newepoch = False  # TODO
        timedata = 0  # time to load the data cannot be computed with this method

        # 1. train the model
        # TODO optim: make in one pass optimizer and storing minibatch info if
        # any
        beg__ = datetime.datetime.now()
        self.graph.run(sess, toberun=self.optimize)
        end__ = datetime.datetime.now()
        tmp = end__ - beg__
        timeTrain = tmp.total_seconds()  # ellapsed time training the model, in seconds

        # 2. store informations # TODO optim make it in one single pass above!
        beg__ = datetime.datetime.now()
        losscomputed, is_error_nan, valloss = self.explogger.logtf(
            minibatchnum=self.exp_params.minibatchnum, graph=self.graph, data=self.data, sess=sess)

        end__ = datetime.datetime.now()
        tmp = end__ - beg__
        timesaving = tmp.total_seconds()  # ellapsed time saving data, in seconds

        self.exp_params.minibatchnum += 1
        return newepoch, is_error_nan, losscomputed, valloss, timedata, timeTrain, timesaving

    def saveTrainedNet(self, sess, valloss, minibatchnum):
        """
        Save the network if the performance on validation set improve
        :param sess: the tensorflow session to use
        :param valloss: the list of loss on validation set
        :param minibatchnum: the current number of minibatches computed
        """

        if len(valloss) >= 2:
            if valloss[-2] >= valloss[-1]:
                #  we saw an improvment on validation set, we save it
                self.explogger.savetf(
                    sess,
                    os.path.join(
                        self.exp_params.path_saver,
                        "ModelTrained_best"))
        else:
            # force saving at the beginning
            self.explogger.savetf(
                sess,
                os.path.join(
                    self.exp_params.path_saver,
                    "ModelTrained_best"))

        #  we save at each iterations
        if minibatchnum // self.exp_params.save_model:
            self.explogger.savetf(
                sess,
                os.path.join(
                    self.exp_params.path_saver,
                    "ModelTrained"),
                global_step=minibatchnum)

    def restoreTrainedNet(self, sess):  # TODO
        pass

    def run_withfeed_dict(self, sess): #TODO in another class. Here it is for class with tensorflow dataset
        """
        Run one single minibatch.
        If it is time, store the error made on that minibatch
        If it is time, store the error made on the training / validation set
        :param sess: a tensorflow session to execute the computation
        :return:
        """
        # 1. get the data
        beg__ = datetime.datetime.now()
        newepoch, * \
            data_minibatch = self.data.nextminibatch(self.exp_params.batch_size)
        end__ = datetime.datetime.now()
        tmp = end__ - beg__
        timedata = tmp.total_seconds()  # ellapsed time getting the data, in seconds

        # 2. train the model
        # TODO optim: make in one pass optimizer and storing minibatch info if
        # any
        beg__ = datetime.datetime.now()
        self.graph.run(sess, toberun=self.optimize, data=data_minibatch)
        end__ = datetime.datetime.now()
        tmp = end__ - beg__
        timeTrain = tmp.total_seconds()  # ellapsed time training the model, in seconds

        # 3. store informations # TODO optim make it in one single pass above!
        beg__ = datetime.datetime.now()
        losscomputed, is_error_nan, valloss = self.explogger.logtf(minibatchnum=self.exp_params.minibatchnum,
                                                                   graph=self.graph,
                                                                   data=self.data,
                                                                   data_minibatch=data_minibatch,
                                                                   sess=sess)

        end__ = datetime.datetime.now()
        tmp = end__ - beg__
        timesaving = tmp.total_seconds()  # ellapsed time saving data, in seconds

        return newepoch, is_error_nan, losscomputed, valloss, timedata, timeTrain, timesaving


class ExpParams:
    def __init__(self, epochnum=1, batch_size=50):
        self.num_epoch = epochnum
        self.batchsize = batch_size


class Exp:
    def __init__(self,
                 parameters,
                 dataClass=ExpData, dataargs=(), datakwargs={},
                 graphType=ExpGraph, graphargs=(), graphkwargs={},
                 modelType=ExpModel, modelargs=(), modelkwargs={}
                 ):
        """
        Build the experience.
        Meaning that it will
        first build the data from what you give (dataClass, dataargs and datakwargs)
        Then build the neural networks with ExpGraph, graphargs, graphkwargs
        Then build the comptuation graph (add the loss, savers etc.) : modelType, modelargs, modelkwargs
        Then open a tensorflow session
        :param parameters:
        :param dataClass: the class
        :param dataargs:
        :param datakwargs:
        :param graphType:
        :param graphargs:
        :param graphkwargs:
        :param modelType:
        :param modelargs:
        :param modelkwargs:
        """

        self.parameters = parameters
        self.path = self.parameters.name_exp_with_path  # path of the experience
        if not os.path.exists(self.path):
            print(
                "I creating the path {} for the current experiment".format(
                    self.path))
            os.mkdir(self.path)
        else:
            print("The path {} already exists".format(self.path))

        # TODO
        self.startfromscratch = True  # do I start the experiments from scratch

        # 1. load the data
        self.data = None
        with tf.variable_scope("datareader"):
            self.data = dataClass(
                pathdata=parameters.pathdata,
                *dataargs,
                **datakwargs)
        # self.inputs, self.inputsize = self.data.getinputs()
        # self.outputs, self.outputsize = self.data.getoutputs()

        # 2. build the graph
        self.graph = None
        with tf.variable_scope("neuralnetwork"):
            self.graph = graphType(data=self.data.getdata(),
                                   *graphargs, **graphkwargs)

        # 3. add the loss, optimizer and saver
        self.model = modelType(
            exp_params=self.parameters,
            data=self.data,
            graph=self.graph,
            *modelargs,
            **modelkwargs)

        # 4. create the tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # 5. defines other quantities needed for additionnal data
        self.timedata = 0  # time to get the data
        self.trainingsteps = 0  # total number of training steps computed
        # self.minibatchnum = 0 # total number of minibatches computed ( #TODO
        # is this the same ?)
        self.timeTrain = 0  # total time spent to train the model
        self.timesaving = 0  # ellapsed time saving data, in seconds
        self.valloss = []  # the loss on the validation set

    def start(self):  # TODO log beginning, and graph restoring
        """
        Run the entire experiments.
        :return:
        """
        is_error_nan = False

        # log the beginning
        self.logbeginning()

        # 1. init the variables
        self.sess.run(tf.global_variables_initializer())

        # 3. init the data
        self.data.init(self.sess)

        # 2. init the weight normalization, is needed
        self.graph.initwn(self.sess)

        # 4. launch the computation
        for epochnum in range(self.parameters.num_epoch):
            is_error_nan = self.runallminibatchesthisepoch()
            if is_error_nan:
                break

        #  log the end of the experiment
        self.logend(is_error_nan)

    def runallminibatchesthisepoch(self):
        """
        Run all the minibatches of a single epoch
        :return: True if the model diverge (output nan) False otherwise
        """
        is_error_nan = False
        batch_size = self.parameters.batch_size
        newepoch = False
        while not newepoch:
            self.trainingsteps += 1
            # call the training function of the model
            newepoch, is_error_nan, losscomputed, valloss, timedata, timeTrain, timesaving = self.model.run(
                self.sess)
            self.timedata += timedata
            self.timeTrain += timeTrain
            self.timesaving += timesaving

            if losscomputed:
                self.valloss.append(valloss)
                self.saveTrainedNet()

            if is_error_nan:
                break

        return is_error_nan

    def logbeginning(self):
        """Log everything at the beginning of an experiment"""
        self.model.explogger.info('Beginning of the experiment')

        self.model.explogger.info("Experience with more traditional data.")
        self.model.explogger.info(
            "Size of train : nsample {}, size_X {},size_Y {} ".format(
                self.data.getnrows(),
                self.graph.get_input_size(),
                self.graph.get_output_size()))
        self.model.explogger.info(
            "Size of validation set : nsample {}, size_X {},size_Y {} ".format(
                self.data.getnrowsval(),
                self.graph.get_input_size(),
                self.graph.get_output_size()))

        dict_summary = {}
        dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
        dict_summary["flop"] = "{}".format(self.graph.getflop())
        dict_summary["get_input_size"] = "{}".format(self.graph.get_input_size())
        dict_summary["get_output_size"] = "{}".format(self.graph.get_output_size())
        self.writesummaryExp(dict_summary)

    def logend(self, is_error_nan):
        """Log everything at the end of an experiment
        :param is_error_nan: is the error seen Nan (eg has the model diverged ?
        :return
        """
        if is_error_nan:
            self.model.explogger.info("Nan error encounter, stopping training")

        dict_summary = {}
        # force saving of everything at the end of the computation
        computed, error_nan, valloss = self.model.explogger.logtf(minibatchnum=self.parameters.minibatchnum,
                                                                  graph=self.graph,
                                                                  data=self.data,
                                                                  sess=self.sess,
                                                                  forcesaving=True)

        if computed:
            self.valloss.append(valloss)
            self.saveTrainedNet()

        self.transitComp, self.error_l1 = self.data.computelasterror(
            sess=self.sess, graph=self.graph, logger=self.model.explogger, params=self.model.parameters, dict_summary=dict_summary)

        dict_summary["training_time"] = self.timeTrain
        dict_summary["training_steps"] = self.trainingsteps
        dict_summary["data_getting_time"] = self.timedata
        dict_summary["data_saving_time"] = self.timesaving
        # dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
        # dict_summary["flops"] = "{}".format(self.graph.getflops())

        self.writesummaryExp(dict_summary)

    def saveTrainedNet(self):
        """
        Save the neural network and
        :return:
        """
        # TODO handle this cases!
        if not self.startfromscratch:
            with open(os.path.join(self.path, "summary.json"), "r") as f:
                dict_summary = json.load(f)
            dict_summary["training_time"] += self.timeTrain
            dict_summary["training_steps"] += self.trainingsteps
            dict_summary["data_getting_time"] += self.timedata
            dict_summary["data_saving_time"] += self.timesaving
            # dict_summary["flops"] = "{}".format(self.graph.getflops())
            # dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
            # dict_summary["l1_val_loss"] += "{}".format(self.vallos[-1])
        else:
            dict_summary = {}
            dict_summary["training_time"] = self.timeTrain
            dict_summary["training_steps"] = self.trainingsteps
            dict_summary["data_getting_time"] = self.timedata
            dict_summary["data_saving_time"] = self.timesaving
            dict_summary["l1_val_loss"] = "{}".format(self.valloss[-1])
            # dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
            # dict_summary["flops"] = "{}".format(self.graph.getflops())
            # dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())

        self.writesummaryExp(dict_summary)
        self.model.saveTrainedNet(
            sess=self.sess,
            minibatchnum=self.parameters.minibatchnum,
            valloss=self.valloss)

    def writesummaryExp(self, dict_summary):
        """
        Update the files located at os.path.join(self.path, "summary.json")
        with the information in dict_summary
        :param dict_summary: [dictionnary] new informations for updating the reference file
        :return: 
        """
        full_path = os.path.join(self.path, "summary.json")
        if os.path.exists(full_path):
            # get back previous item
            with open(full_path, "r") as f:
                dict_saved = json.load(f)
        else:
            dict_saved = {}
        with open(full_path, "w") as f:
            for key, val in dict_summary.items():
                dict_saved[key] = val
            json.dump(dict_saved, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    from DataHandler import OnlineData, StoredData, OverFitData
    path_IEEE = "/save"
    path_scripts = "/home/bdonnot/Documents/RHades2"
    path_scripts = "/home/benjamin/Documents/RHades2"
    paths = Path(
        path_data=os.path.join(
            path_IEEE,
            "ieee"),
        path_IEEE=path_IEEE,
        path_save=os.path.join(
            path_scripts,
            "PyHades2",
            "modelSaved"),
        path_scripts=path_scripts,
        path_save_data=os.path.join(
            path_scripts,
            "PyHades2",
            "BestResults"))
    fileName = "ieee30_ADN.xml"

    os.environ["GSR_TRACE_CONF"] = os.path.join(
        paths.path_scripts, "dep/config/Trace.conf")
    os.environ["GSR_PATH"] = os.path.join(paths.path_scripts, "dep/config/etc")

    tested_params = [
        {'batch_size': 150, 'nb_layer': 7, 'learning_rate': 0.001, 'layer_size': 10},
        {'nb_layer': 9, 'layer_size': 30, 'batch_size': 15, 'learning_rate': 0.0004},
        {'nb_layer': 6, 'layer_size': 45, 'batch_size': 15, 'learning_rate': 0.1},
        {'nb_layer': 6, 'layer_size': 35, 'batch_size': 100, 'learning_rate': 0.01},
        {'nb_layer': 6, 'layer_size': 25, 'batch_size': 5, 'learning_rate': 0.01},
        {'nb_layer': 9, 'layer_size': 25, 'batch_size': 100, 'learning_rate': 0.1},
        {'nb_layer': 5, 'layer_size': 40, 'batch_size': 40, 'learning_rate': 0.0002},
        {'nb_layer': 1, 'layer_size': 35, 'batch_size': 50, 'learning_rate': 0.001},
        {'nb_layer': 4, 'layer_size': 30, 'batch_size': 20, 'learning_rate': 0.0008},
        {'nb_layer': 5, 'layer_size': 30, 'batch_size': 40, 'learning_rate': 0.03},
        {'nb_layer': 9, 'layer_size': 40, 'batch_size': 10, 'learning_rate': 0.01},
        {'nb_layer': 9, 'layer_size': 40, 'batch_size': 50, 'learning_rate': 0.0002},
        {'nb_layer': 9, 'layer_size': 20, 'batch_size': 15, 'learning_rate': 0.0006},
        {'nb_layer': 9, 'layer_size': 10, 'batch_size': 50, 'learning_rate': 0.0006},
        {'nb_layer': 5, 'layer_size': 15, 'batch_size': 50, 'learning_rate': 0.0008},
        {'nb_layer': 8, 'layer_size': 15, 'batch_size': 1, 'learning_rate': 0.0001},
        {'nb_layer': 3, 'layer_size': 35, 'batch_size': 100, 'learning_rate': 0.0006},
        {'nb_layer': 5, 'layer_size': 50, 'batch_size': 5, 'learning_rate': 0.0001},
        {'nb_layer': 1, 'layer_size': 20, 'batch_size': 15, 'learning_rate': 0.0004},
        {'nb_layer': 6, 'layer_size': 60, 'batch_size': 100, 'learning_rate': 0.03},
        {'nb_layer': 4, 'layer_size': 30, 'batch_size': 15, 'learning_rate': 0.0008},
        {'nb_layer': 7, 'layer_size': 60, 'batch_size': 20, 'learning_rate': 0.03},
        {'nb_layer': 4, 'layer_size': 45, 'batch_size': 20, 'learning_rate': 0.001},
        {'nb_layer': 9, 'layer_size': 30, 'batch_size': 40, 'learning_rate': 0.0001},
        {'nb_layer': 1, 'layer_size': 35, 'batch_size': 5, 'learning_rate': 0.001},
        {'nb_layer': 7, 'layer_size': 50, 'batch_size': 20, 'learning_rate': 0.001},
        {'nb_layer': 3, 'layer_size': 15, 'batch_size': 1, 'learning_rate': 0.0006}
    ]

    random.seed(42)

    ONLINE = False
    if ONLINE:
        import reseauADN
        num_core = 4
        n_eval = 1000
        fIn = os.path.join(paths.path_data, fileName)
        net = reseauADN.Reseau_Base(fIn)
        data = OnlineData(net, num_core=num_core)
        # generate (or get) a testing set
        # that will be used during the experiment
        # to compute the error
        plouf = data.valdata(size=n_eval, last=False)
        plouf = data.traindata(size=n_eval, last=False)
    else:
        data = StoredData(
            os.path.join(
                path_scripts,
                "PyHades2",
                "ampsdatareal"))
        # data = StoredData(os.path.join(path_scripts, "PyHades2", "intensityData"))
        # data = OverFitData(os.path.join(path_scripts, "PyHades2", "ampsdatareal"),
        #                   size=50)

    for par in tested_params[14:25]:
        params = ParamsIEEE()
        params.init_fromdict(par)
        print(params)
        # print(par)
        exp = OldExperience(paths=paths,
                            params=params,
                            graphType=MicrosoftResNet,
                            # TFGraphFully,#TFGraphFullyWithDec,#MicrosoftResNet,
                            data=data,
                            js=20,
                            ks=300,  # 180000,#it seems enough #ks=3000000, #to compare with old methods
                            # experimentaly, ks = 180 000 is enough when MicrosoftResNet/AdamOptimizer is used. Often the error afterwards increase
                            # ks is deprecated, use js (equiv to epochnum instead)
                            online=ONLINE,
                            num_eval_error=1,
                            num_eval_last_error=250,
                            fileName=fileName,
                            optimizer=tf.train.AdamOptimizer)  # tf.train.RMSPropOptimizer)

        # print(TFGraphFully.__name__)
        exp.run_withoutLr(startAgain=True, computeAll=True)  # ,oldmethod=True)
