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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda *i, **kwargs: i[0]  # pylint:disable=invalid-name

import tensorflow as tf

from .ANN import NNFully
from .DataHandler import ExpData
from .Graph import ExpGraphOneXOneY

TRAINING_COLLECTION_NAME = "train_op"
NAMESAVEDTFVARS = 'savedvars'
NAME_PH_SAVED = "placeholder"

#TODO do not save if just restoring the model for predictions only. Also make sure
#TODO that realoading a reloaded situation end up leading the one choosen (and not the original one)
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
    def __init__(self, path, trainname, valname, minibatchname, saver=None, othersaver={}):
        """Handle different kind of writer for usage in tensorflow
           By default save 3 infos:
            -minibatch: the last loss computed
            -train; error on the whole training set
            -validation: error on the whole validation set
        :param path: where the files are stored
        :param trainname: the name displayed for the "training" writer
        :param valname: the name displayed for the "valname" writer
        :param minibatchname: the name displayed for the "minibatch" writer
        :param othersaver: names of other saver to use. These names should match names given in datasets
        """

        # save the minibatch info
        self.minibatchname = minibatchname
        self.minibatchwriter = tf.summary.FileWriter(
            os.path.join(path, minibatchname)
            , graph=tf.get_default_graph())

        # save the training set info
        self.trainname = trainname
        self.trainwriter = tf.summary.FileWriter(
            os.path.join(path, trainname),
            graph=tf.get_default_graph())

        # save the validation set info
        self.valname = valname
        self.valwriter = tf.summary.FileWriter(
            os.path.join(path, valname),
            graph=tf.get_default_graph())

        #other savers
        self.othersavers = {}
        for savername in othersaver:
            self.othersavers[savername] = tf.summary.FileWriter(os.path.join(path, savername),
                                                           graph=tf.get_default_graph())
        # TODO idee : mettre en argument une liste (nomsaver / dataset), et quand on appelerait 'save' ca calculerait,
        # TODO pour chaque element de cette liste, l'erreur sur le datset et le
        # TODO sauvegarderait au bon endroit.

        # saver of the graph
        self.saver = saver if saver is not None else tf.train.Saver(
            max_to_keep=None)


class ExpLogger:
    MAXNUMBATCHPEREPOCH = int(1e6)  # static member

    def __init__(
            self,
            path,
            params,
            name_model,
            logger,
            nameSaveLearning,
            num_savings,
            num_savings_minibatch,
            epochsize,
            saveEachEpoch,
            otherinfo = {},
            saver=None
    ):
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
        :param otherinfo: names forwarded to TFWriters for other info to save (use this if you want to report error on a test set for example)
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
        self.tfwriter = TFWriters(path=path,
                                  trainname="Train",
                                  valname="Val",
                                  minibatchname="Minibatch",
                                  saver=saver,
                                  othersaver=otherinfo)
        self.otherinfo = otherinfo
        self.path_saveinfomini = os.path.join(path, "minibatch.count")
        self.filesavenum = open(self.path_saveinfomini, "a")
        self.params = params
        self.epochsize = epochsize
        self.saveEachEpoch = saveEachEpoch

    def logtf(self, minibatchnum, graph, data, sess, forcesaving=False):
        """
        Compute the validation set error, and the training set error and the minibatch error, and store them in tensorboard
        :param minibatchnum: current number of minibatches proccessed
        :param graph: an object of class 'ExpGraphOneXOneY' or one of its derivatives
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
            # compute error on training set and validation set (mandatory)
            error_nan, loss_ = data.computetensorboard(
                sess=sess, writers=self, graph=graph, xval=global_step, minibatchnum=minibatchnum)
            # compute error on the other datasets
            for savername in self.otherinfo:
                data.computetensorboard_annex(sess=sess, writers=self, graph=graph, xval=global_step,
                                              minibatchnum=minibatchnum, name=savername)

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
        :param graph: an object of class 'ExpGraphOneXOneY' or one of its derivatives
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

    def close(self):
        self.filesavenum.close()

class ExpParam:
    def __init__(self,
                 path=".",
                 pathdata=".",
                 params=None,
                 logger=None,
                 name_exp="MyExp",
                 num_savings=100,
                 num_savings_minibatch=500,
                 num_savings_model=20,
                 # epochsize=1,
                 saveEachEpoch=False,
                 # minibatchnum=0,
                 batch_size=1,
                 num_epoch=1,
                 continue_if_exists=False,
                 showtqdm=False):
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
        # :param epochsize: the size of an epoch in terms of minibatches
        :param saveEachEpoch: do you want to force the savings at each epoch
        # :param minibatchnum: the number of minibatches computed
        :param batch_size: the size of one training minibatch
        :param num_epoch: the number of epoch to run
        :param continue_if_exists: if the folder exists, stop do not run the expriment
        :param showtqdm: if False, deactivate display of progress bar
        """
        self.path = path
        self.pathdata = pathdata
        self.params = params
        self.logger = logger
        self.nameSaveLearning = name_exp
        # self.saver = saver
        self.num_savings = num_savings
        self.save_loss = 0
        self.num_savings_minibatch = num_savings_minibatch
        self.save_minibatch_loss = 0
        self.num_savings_model = num_savings_model
        self.save_model = 0
        self.epochsize = 0
        self.saveEachEpoch = saveEachEpoch
        self.name_model = name_exp
        self.minibatchnum = 0
        self.batch_size = batch_size
        self.name_exp_with_path = os.path.join(self.path, self.name_model)
        self.path_saver = os.path.join(self.name_exp_with_path, "TFInfo")
        self.num_epoch = num_epoch
        self.total_minibatches = 0
        self.continue_if_exists = continue_if_exists
        self.showtqdm = showtqdm

    def initsizes(self, nrows):
        """
        Initialize, with the dataset sizes, the proper number of savings
        :param nrows: total number of rows of the training dataset
        :return:
        """
        #TODO display warnings when the number should be set to 1
        self.epochsize = round(nrows/self.batch_size)  # number of minibatches per epoch
        # pdb.set_trace()
        if self.epochsize == 0:
            self.epochsize = 1
        self.total_minibatches = self.epochsize * self.num_epoch  # total number of minibatches
        # save the loss each "self.save_loss" minibatches
        self.save_loss = round(self.total_minibatches / self.num_savings)
        if self.save_loss == 0:
            self.save_loss = 1
        self.save_minibatch_loss = round(self.total_minibatches / self.num_savings_minibatch)
        if self.save_minibatch_loss == 0:
            self.save_minibatch_loss = 1
        self.save_model = round(self.total_minibatches / self.num_savings_model)
        if self.save_model == 0:
            self.save_model = 1


class ExpModel:
    def __init__(self,
                 exp_params,
                 data, graph,
                 lossfun=tf.nn.l2_loss,
                 optimizerClass=tf.train.AdamOptimizer, optimizerkwargs={},
                 netname="",
                 otherinfo={}):
        """ init the model with hyper-parameters etc
        add the loss and the optimizer to the pure definition of the neural network.
        The NN is defined by ExpGraphOneXOneY.
        Here the saver / restorer is defined as well.
        :param exp_params: an object of class ExpParam
        :param data: an object of class ExpData or one of its derivative. The data used for the computation
        :param graph: an object of class ExpGraphOneXOneY or one of its derivative. The NN used for the computation.
        :param lossfun: the loss function to use
        :param optimizerClass: the class optimizer to use (tf.train.optimizer)
        :param optimizerkwargs: the key-words arguments to build the optimizer. You can pass the learning rate here.
        :param otherinfo: an iterable: the names of all the other dataset for which errors will be computed (for example Test dataset)
        """
        self.lossfun = lossfun
        self.optimizerClass = optimizerClass
        self.exp_params = exp_params

        # 1. store the graph and the data class
        self.graph = graph
        self.data = data
        self.exp_params.initsizes(self.data.getnrows())

        # 2. build some important node: the inference, loss and optimizer node
        with tf.variable_scope("inference"):
            self.inference = {k: tf.identity(v, name="{}".format(k)) for k,v in graph.getoutput().items()}
            true_output_dict = self.graph.get_true_output_dict()
        self.loss = None

        with tf.variable_scope("training_loss"):
            self.losses = {k: self.lossfun(self.inference[k]-true_output_dict[k], name="training_loss_{}".format(k)) for k in self.inference.keys()}
            self.loss = tf.constant(0., dtype=tf.float32)
            for _, l in self.losses.items():
                # TODO capability of having ponderated loss!
                self.loss = self.loss + l

        self.optimize=None
        with tf.variable_scope("optimizer"):
            self.optimize = optimizerClass(
                **optimizerkwargs).minimize(loss=self.loss, name="optimizer")

        # 3. build the summaries that will be stored
        with tf.variable_scope("summaries"):
            self.error = {}
            self.error_abs = {}
            self.error_abs = {}
            self.l1_avg = {}
            self.l2_avg = {}
            self.l_max = {}
            li_summaries = []
            for var_output_name in self.inference.keys():
                k = var_output_name
                with tf.variable_scope("{}".format(var_output_name)):
                    self.error[k] = tf.add(self.inference[k], -true_output_dict[k], name="{}_{}_error_diff".format(netname, k))
                    self.error_abs[k] = tf.abs(self.error[k],  name="{}_{}_error_abs".format(netname, k))
                    self.l1_avg[k] = tf.reduce_mean( self.error_abs[k],  name="{}_{}_l1_avg".format(netname, k))
                    self.l2_avg[k] = tf.reduce_mean(self.error[k] * self.error[k],  name="{}_{}_l2_avg".format(netname, k))
                    self.l_max[k] = tf.reduce_max(self.error_abs[k],  name="{}_{}_l_max".format(netname, k))

                    # add loss as a summary for training
                    sum0 = tf.summary.scalar(netname + "loss_{}_{}".format(netname, k), self.losses[k])
                    sum1 = tf.summary.scalar(netname + "l1_avg_{}_{}".format(netname, k), self.l1_avg[k])
                    sum2 = tf.summary.scalar(netname + "l2_avg_{}_{}".format(netname, k), self.l2_avg[k])
                    sum3 = tf.summary.scalar(netname + "loss_max_{}_{}".format(netname, k), self.l_max[k])

                    li_summaries += [sum0, sum1, sum2, sum3]

                    tf.add_to_collection(NAMESAVEDTFVARS, self.losses[k])
                    tf.add_to_collection(NAMESAVEDTFVARS, self.inference[k])
                    tf.add_to_collection(NAMESAVEDTFVARS, self.l1_avg[k])
                    tf.add_to_collection(NAMESAVEDTFVARS, self.l2_avg[k])
                    tf.add_to_collection(NAMESAVEDTFVARS, self.l_max[k])

                    tf.add_to_collection("LOSSFUNFully" + netname, self.losses[k])
                    tf.add_to_collection("OUTPUTFully" + netname, self.inference[k])
            self.mergedsummaryvar = tf.summary.merge(li_summaries)
        self.graph.init(self.mergedsummaryvar, self.loss)

        # 4. create the saver object (at the end, because each node of the graph must be saved)
        self.explogger = ExpLogger(
            path=exp_params.path_saver,
            params=exp_params,
            logger=exp_params.logger,
            nameSaveLearning=exp_params.nameSaveLearning,
            num_savings=exp_params.num_savings,
            num_savings_minibatch=exp_params.num_savings_minibatch,
            epochsize=exp_params.epochsize,
            saveEachEpoch=exp_params.saveEachEpoch,
            name_model=exp_params.name_model,
            otherinfo=otherinfo)

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
        # print(self.exp_params.minibatchnum)
        newepoch = self.exp_params.minibatchnum % self.exp_params.epochsize == 0
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
        if minibatchnum % self.exp_params.save_model == 0:
            # pdb.set_trace()
            self.explogger.savetf(
                sess,
                os.path.join(
                    self.exp_params.path_saver,
                    "ModelTrained"),
                global_step=minibatchnum)

    def restoreTrainedNet(self, sess):  # TODO
        pass

    def checkreloaded(self, sess):
        """
        Check that the model properly reloaded the weights
        :param sess: 
        :return: 
        """
        self.explogger.logtf(minibatchnum=0, graph=self.graph, data=self.data, sess=sess, forcesaving=True)

    def computelasterror(self, sess, dict_summary=None):
        """
        Compute and store in the writers (tensorflow and text logger the last information about the experiment)
        :param sess: a tensorflow session
        :param dict_summary: the summary dictionnary to save other informations
        :return: 
        """

        #TODO for now getpred takes a lot of RAM, maybe it is not necessary and it can be optimized
        predicted, orig = self.data.getpred(sess=sess, graph=self.graph, varsname=self.graph.outputname)
        if self.explogger.logger is not None:
            self.explogger.logger.info("_______________________________")
            self.explogger.logger.info("Computing error for \"{}\" dataset".format("validation"))
        for varname in orig.keys():
            self.logfinalerror(varname, pred=predicted[varname], true=orig[varname], dict_summary=dict_summary)
        for dsn, ds in self.data.otherdatasets.items():
            predicted, orig = self.data.getpred(sess=sess, graph=self.graph,
                                                varsname=self.graph.outputname, dataset_name=dsn)
            if self.explogger.logger is not None:
                self.explogger.logger.info("_______________________________")
                self.explogger.logger.info("Computing error for \"{}\" dataset".format(dsn))
            for varname in orig.keys():
                self.logfinalerror(varname, pred=predicted[varname], true=orig[varname],
                                   dict_summary=dict_summary, dataset_name=dsn)

    def logfinalerror(self, varname, pred, true, dict_summary=None, dataset_name="Val", logger=None):
        """
        Log the final error (eg at the end of training) meaning that:
            - comptue the error for the whole validation set (using pred and true array)
            - log it in text file self.explogger.logger
            - update the info in dict_summary
        :param varname: the name of the variable concerned
        :param pred: the predicted value for this varibale (numpy array)
        :param true: the true value for this variable (numpy array)
        :param dict_summary: a dictionary for logging the error
        :param logger: None for logger of self.explogger, another logger otherwise
        :return: 
        """
        error = pred - true
        mean_abs_error = np.mean(np.abs(error))
        mean_abs_val = np.mean(np.abs(true))
        max_abs_val = np.max(np.abs(true))

        if logger is None:
            logger = self.explogger.logger
            
        if logger is not None:
            logger.info("Final MAE (mean abs error) for {} : {:.3f} ({:.1f}%)".format(
                varname, mean_abs_error, mean_abs_error / mean_abs_val * 100))
        max_error = np.max(np.abs(error))
        if logger is not None:
            logger.info("Max MAE error for {} : {:.3f} ({:.3f} -- {:.1f}%)".format(
                varname, max_error, max_error / mean_abs_val, max_error / max_abs_val * 100))
            
        a = np.full(shape=(1, true.shape[1]), fill_value=1.0)
        threshold = np.maximum(np.mean(np.abs(true), axis=0) * 1e-3, a)
        mean_rel_error = np.mean(np.abs(error[np.abs(true) >= threshold]) /
                                 np.abs(true[np.abs(true) >= threshold]))
        if logger is not None:
            logger.info("Final MAPE for {} : {:.3f}% ".format(
                varname, mean_rel_error * 100))

        max_rel_error = np.max(np.abs(error[np.abs(true) >= threshold] /
                                      true[np.abs(true) >= threshold]))
        if logger is not None:
            logger.info("Final max MAPE for {} : {:.3f}% ".format(
                varname, max_rel_error * 100))
        b = np.percentile(np.abs(true), 90, axis=0).reshape((1, true.shape[1]))
        threshold = np.maximum(a, b)
        mean_abs_error_high = np.mean(np.abs(error[np.abs(true) >= threshold])).astype(np.float32)
        if logger is not None:
            logger.info("Final MAE (when abs true_values >= q_90) for {} : {:.3f} ".format(
                varname, mean_abs_error_high))

        mean_rel_error_high = np.mean(np.abs(error[np.abs(true) >= threshold] /
                                             true[np.abs(true) >= threshold]))
        if logger is not None:
            logger.info("Final MAPE (abs values >= q_90) for {} : {:.3f}% ".format(
                varname, 100*mean_rel_error_high))

        max_rel_error_high = np.max(np.abs(error[np.abs(true) >= threshold] /
                                           true[np.abs(true) >= threshold]))
        if logger is not None:
            logger.info("Final max MAPE (abs values >= q_90) for {} : {:.3f}% ".format(
                varname, max_rel_error_high * 100))

            logger.info("Mean (abs) value val set for {} : {:.3f}".format(varname, mean_abs_val))
            logger.info("Std value val set for {} : {:.3f}".format(varname, np.std(true)))
            logger.info("Max (abs) value val set for {} : {:.3f}".format(varname, max_abs_val))

        if dict_summary is not None:
            dict_summary[varname + "_mean_abs_error"] = "{}".format(mean_abs_error)
            dict_summary[varname + "_mean_abs_error_pct"] = "{}".format(mean_abs_error / mean_abs_val * 100)

            dict_summary[varname + "_max_abs_error"] = "{}".format(max_error)
            dict_summary[varname + "_max_abs_error_pct"] = "{}".format(max_error / max_abs_val * 100)

            dict_summary[varname + "_mean_rel_error"] = "{}".format(mean_rel_error)
            dict_summary[varname + "_max_rel_error"] = "{}".format(max_rel_error)

            dict_summary[varname + "_mean_value"] = "{}".format(mean_abs_val)
            dict_summary[varname + "_max_value"] = "{}".format(max_abs_val)

            dict_summary[varname + "_mean_abs_error_high"] = "{}".format(mean_abs_error_high)
            dict_summary[varname + "_max_rel_error_high"] = "{}".format(max_rel_error_high)
            # if params is not None:
            #     dict_summary[varname + "_params"] = "{}".format(params[varname])

        if logger is not None:
            dummy_pred = np.mean(true, axis=0)
            error = dummy_pred - true
            mean_abs_error = np.mean(np.abs(error))
            logger.info("Dummy pred (mean by dimension) MAE error (val set) for {} : {:.3f}".format(
                varname, mean_abs_error))
            threshold = np.maximum(np.mean(np.abs(true), axis=0) * 1e-3, a)
            mean_rel_error = np.mean(np.abs(error[np.abs(true) >= threshold]) /
                                     np.abs(true[np.abs(true) >= threshold]))
            logger.info("Dummy pred (mean by dimension) MAPE (val set) for {} : {:.3f}%".format(
                varname, mean_rel_error * 100))

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

    def close(self):
        self.explogger.close()


class ExpSaverParam(ExpParam):
    def __init__(self, *args, **kwargs):
        ExpParam.__init__(self, *args, **kwargs)
        print("W: ExpSaverParam: DEPRECATED, use ExpParam instead")

class Exp:
    def __init__(self,
                 parameters,
                 dataClass=ExpData, dataargs=(), datakwargs={},
                 graphType=ExpGraphOneXOneY, graphargs=(), graphkwargs={},
                 modelType=ExpModel, modelargs=(), modelkwargs={},
                 otherdsinfo={},
                 startfromscratch=False
                 ):
        """
        Build the experience.
        Meaning that it will
        first build the data from what you give (dataClass, dataargs and datakwargs)
        Then build the neural networks with ExpGraphOneXOneY, graphargs, graphkwargs
        Then build the comptuation graph (add the loss, savers etc.) : modelType, modelargs, modelkwargs
        Then open a tensorflow session
        :param parameters: the experiment parameters (object of class ExpParam)
        :param dataClass: the class used for building the data handler (ExpData or one of its derivatives -- pass the class, not an object)
        :param dataargs: arguments used to build the data handler
        :param datakwargs: key word arguments used to build the data handler
        :param graphType: the class used for building the computation graph (neural network), for example "ExpGraph" -- pass the class not an object
        :param graphargs: arguments used to build the computation graph
        :param graphkwargs: key word arguments used to build the computation graph
        :param modelType: the type of model abstraction (eg neural network + optimizer + saver) you want to use (for example ExpModel) -- pass the class, not an object
        :param modelargs: arguments used to build the model abstraction
        :param modelkwargs: key word arguments used to build the model abstraction
        :param otherdsinfo: other dataset info, used when you want to save other information during training (dictionnary with key: name of the dataset, value = dictionnary with key = [argsdata, kwargsdata] and value the arguments used to build the data handler for these other datasets (see dataargs and datakwargs)
        :param startfromscratch: do you start the experiment from scratch, or do you want to reload a previous experiment made.
        """

        self.parameters = parameters
        self.path = self.parameters.name_exp_with_path  # path of the experience
        if not os.path.exists(self.path):
            print(
                "I creating the path {} for the current experiment".format(
                    self.path))
            os.mkdir(self.path)
        else:
            if self.parameters.continue_if_exists:
                print("The path {} already exists".format(self.path))
            else:
                str_ = "The path \"{}\" already exists"
                str_ += "If you still want to continue you can pass \"continue_if_exists=True\""
                str_ += " when you build the parameters of the experiments (object of class ExpParam)"
                raise RuntimeError(str_.format(self.path))


        self.startfromscratch = startfromscratch  # do I start the experiments from scratch
        # save the parameters of the experiments
        if self.startfromscratch:
            self.path_saveparamexp = os.path.join(self.path, "parameters")
            if not os.path.exists(self.path_saveparamexp):
                os.mkdir(self.path_saveparamexp)
            with open(os.path.join(self.path_saveparamexp, "parameters.json"), "w") as f:
                json.dump(parameters.__dict__, fp=f, sort_keys=True, indent=4)
            self._saveinfos(name="data_related",classType=dataClass,
                           args=dataargs, kwargs=datakwargs)
            self._saveinfos(name="graph_related",classType=graphType,
                           args=graphargs, kwargs=graphkwargs)
            self._saveinfos(name="model_related",classType=modelType,
                           args=modelargs, kwargs=modelkwargs)

        # tf.reset_default_graph()
        self.data = None
        self.graph = None

        # with tf.Graph.as_default() as g:
        # 1. load the data
        with tf.variable_scope("datareader"):
            self.data = dataClass(
                *dataargs,
                batch_size=parameters.batch_size,
                pathdata=parameters.pathdata,
                path_exp=self.path,
                otherdsinfo=otherdsinfo,
                **datakwargs)
        # 2. build the graph or load the graph
        with tf.variable_scope("neuralnetwork"):
            self.graph = graphType(*graphargs,
                                   data=self.data.getdata(),
                                   **graphkwargs
                                   )

        # 3. add the loss, optimizer and saver
        self.model = modelType(exp_params=self.parameters,
                                data=self.data,
                                graph=self.graph,
                                otherinfo=otherdsinfo.keys(),
                                *modelargs,
                                **modelkwargs)

        # 4. create the tensorflow session
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        # self.parameters.saver = tf.train.Saver()
        self.sess = tf.Session(config=self.config)

        # 5. defines other quantities needed for additionnal data
        self.timedata = 0  # time to get the data
        self.trainingsteps = 0  # total number of training steps computed
        self.timeTrain = 0  # total time spent to train the model
        self.timesaving = 0  # ellapsed time saving data, in seconds
        self.valloss = []  # the loss on the validation set

        self.is_initialized = False

    def _saveinfos(self, name, classType, args, kwargs):
        """
        Save some information about data. Not to use outside the class
        :param name: 
        :param classType: 
        :param args: 
        :param kwargs: 
        :return: 
        """
        with open(os.path.join(self.path_saveparamexp, "{}.json".format(name)), "w") as f:
            dd = {}
            dd["classType"] = str(classType)
            dd["args"] = [str(el) for el in args]
            dd.update({str(k): str(v) for k, v in kwargs.items()})
            json.dump(dd, fp=f, sort_keys=True, indent=4)

    def _initialize(self):
        """
        Initialize the model, should not be used from outside the class
        :return: 
        """
        # pdb.set_trace()
        if self.is_initialized:
            return
        # 1. init the variables
        if self.startfromscratch:
            # log the beginning
            self.logbeginning()
            self.sess.run(tf.global_variables_initializer())
            with open(os.path.join(self.path_saveparamexp, "parameters.json"), "w") as f:
                json.dump(self.parameters.__dict__, fp=f, sort_keys=True, indent=4)
        else:
            # self.parameters.saver = tf.train.Saver()
            # get the value of the best variable saved
            # self.parameters.saver.restore(self.sess, os.path.join(self.path, "TFInfo", "ModelTrained_best"))
            self.model.explogger.tfwriter.saver.restore(self.sess, os.path.join(self.path, "TFInfo", "ModelTrained_best"))
        # pdb.set_trace()

        # 3. init the data and graph
        self.data.init(self.sess)
        self.graph.startexp(self.sess)

        if not self.startfromscratch:
            # check that the model didn't do anything stupid while reloading
            self.model.checkreloaded(self.sess)
        # pdb.set_trace()
        # 2. init the weight normalization, if needed
        if self.startfromscratch:
            self.graph.initwn(self.sess)
        # pdb.set_trace()
        self.is_initialized = True


    def start(self):  # TODO log beginning, and graph restoring
        """
        Run the entire experiments.
        :return:
        """

        self._initialize()
        is_error_nan = False

        # 4. launch the computation
        with tqdm(total=self.parameters.num_epoch, desc="Epoch", disable=not self.parameters.showtqdm) as pbar:
            for epochnum in range(self.parameters.num_epoch):
                is_error_nan = self.runallminibatchesthisepoch()
                self.graph.tell_epoch(self.sess, epochnum=epochnum)
                pbar.update(1)
                if is_error_nan:
                    break

        #  log the end of the experiment
        self.logend(is_error_nan)
        # pdb.set_trace()

    def runallminibatchesthisepoch(self):
        """
        Run all the minibatches of a single epoch
        :return: True if the model diverge (output nan) False otherwise
        """
        is_error_nan = False
        newepoch = False
        with tqdm(total=self.parameters.epochsize, desc="Minibatches", disable=not self.parameters.showtqdm) as pbar:
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

                pbar.update(1)
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
        for dsname, ds in self.data.otherdatasets.items():
            self.model.explogger.info(
                "Size of {} set : nsample {}, size_X {},size_Y {} ".format(dsname,
                                                                           ds.nrows,
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

        self.model.computelasterror(sess=self.sess, dict_summary=dict_summary)

        dict_summary["training_time"] = self.timeTrain
        dict_summary["training_steps"] = self.trainingsteps
        dict_summary["data_getting_time"] = self.timedata
        dict_summary["data_saving_time"] = self.timesaving

        self.writesummaryExp(dict_summary)

    def saveTrainedNet(self):
        """
        Save the neural network and update the information in dict_summary on disk
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
        else:
            dict_summary = {}
            dict_summary["training_time"] = self.timeTrain
            dict_summary["training_steps"] = self.trainingsteps
            dict_summary["data_getting_time"] = self.timedata
            dict_summary["data_saving_time"] = self.timesaving
            dict_summary["l1_val_loss"] = "{}".format(self.valloss[-1])

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
            try:
                # get back previous item
                with open(full_path, "r") as f:
                    dict_saved = json.load(f)
            except:
                # someone put a non json file at the given place, or the json is corrupted
                dict_saved = {}
        else:
            dict_saved = {}
        with open(full_path, "w") as f:
            for key, val in dict_summary.items():
                dict_saved[key] = val
            json.dump(dict_saved, f, sort_keys=True, indent=4)

    def getpred(self, dsname=None, includeinput=False):
        """
        :param dsnam: the name of the dataset you want to get the error from (none=validation dataset)
        :param includeinput: include the "prediction" for input data also
        :return: 2 dictionnaries: one containing the prediction, the second the true values
        """
        self._initialize()
        varname = self.graph.outputname
        if includeinput:
            varname = varname | self.graph.inputname
        return self.data.getpred(self.sess, self.graph, varname, dataset_name=dsname)

    def __enter__(self):
        # kill and delete the previous session (unused, but just to be sure)
        self.sess.close()
        del self.sess
        # recreate a proper session that will be used and delete in __exit__
        self.sess = tf.Session(config=self.config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()
        del self.sess
        self.model.close()
        # tf.reset_default_graph()