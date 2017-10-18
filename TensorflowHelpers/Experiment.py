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


class ExpDataReader:
    def __init__(self, train, batch_size):
        """Read the usefull data for the Experience to run
        Store both input and outputs
        :param train: if True this data reader concerns the training set
        :param batch_size: number of data to read each time the iterator is called
        """
        self.dataX = np.zeros(
            (0, 0), dtype=np.float32)  # main data set (eg the training set)
        # main data set (eg the training set)
        self.dataY = np.zeros((0, 0), dtype=np.float32)
        self.dataset = tf.contrib.data.Dataset.from_tensor_slices(
            (self.dataX, self.dataY))

        self.mX = tf.constant(0.)
        self.mY = tf.constant(0.)
        self.sdX = tf.constant(1.)
        self.sdY = tf.constant(0.)

    def nrowsX(self):
        """
        :return: The number of rows / examples of the input data set
        """
        return self._nrows(self.dataX)

    def _nrowsY(self):
        """
        :return: The number of rows / examples of the output data set
        """
        return self._nrows(self.dataY)

    def _ncols(self, array):
        """
        :param array: the concerned "array"
        :return: The number of columns / variables of the data set  "array"
        """
        return array.shape[1]

    def _nrows(self, array):
        """
        :param array: the concerned "array"
        :return: The number of  rows / examples of the data set  "array"
        """
        return array.shape[0]

    def ncolsX(self):
        """
        :return: The number of columns / variables of the input data set
        """
        return self._ncols(self.dataX)

    def ncolsY(self):
        """
        :return: The number of columns / variables of the output data set
        """
        return self._ncols(self.dataY)

    def init(self, sess):
        """
        Initialize the datasets (if need to be done)
        :return:
        """
        pass


class ExpDataReaderNPY(ExpDataReader):  # TODO! does not work!!!
    def __init__(self, dataX=None, dataY=None, filenames=None, pathdata="."):
        """
        Read from npy files or take a numpy array as input.
        Be carefull, only one of 'data' or 'filename' must be provided.
        If data is provided, the data will not be copied.
        :param dataX: numpy array (or equivalent) for the input data
        :param dataY: numpy array (or equivalent) for the output data
        :param filenames: filenamaes of npy files where data are stored
        :param pathdata: path where data are stored
        """
        raise RuntimeError("Do not use ExpDatReaderNPY yet!")
        ExpDataReader.__init__(self)
        if (filenames is None or len(filenames) == 0) and (
                dataX is None or dataY is None):
            raise RuntimeError(
                "ExpDataReaderNPY.__init__: you must provide at least a numpy array in dataX / dataY or a filename")

        if (filenames is not None or len(filenames) == 2) and (
                dataX is not None and dataY is not None):
            raise RuntimeError(
                "ExpDataReaderNPY.__init__: you must provide only one 'dataX / dataY' or 'filename' fields")

        if dataX is not None and dataY is None:
            raise RuntimeError(
                "ExpDataReaderNPY.__init__: you should provide both dataX and dataY or none of them.")
        if len(filenames) != 2:
            raise RuntimeError(
                "ExpDataReaderNPY.__init__: you should provide 2 names in filenames: the name of the input data and the one ouput data in filenames.")

        if filenames is not None:
            if re.match(".*\\.npy$", filenames[0]):
                self.dataX = np.load(os.path.join(pathdata, filenames[0]))
                self.dataY = np.load(os.path.join(pathdata, filenames[1]))
            elif re.match(".*\\.csv$", filenames[0]):
                print("ExpDataReaderNPY.__init__: you provided csv data. Data will be parsed with pandas, and only numeric variables will be kept")
                import pandas as pd
                numerics = [
                    'int16',
                    'int32',
                    'int64',
                    'float16',
                    'float32',
                    'float64']

                self.dataX = np.array(
                    pd.read_csv(
                        os.path.join(
                            pathdata,
                            filenames[0]),
                        sep=";").select_dtypes(
                        include=numerics))
                self.dataY = np.array(
                    pd.read_csv(
                        os.path.join(
                            pathdata,
                            filenames[1]),
                        sep=";").select_dtypes(
                        include=numerics))
            else:
                raise RuntimeError(
                    "ExpDataReaderNPY.__init__Impossible to read data of type {} -- make sure your file is a npy or a csv (sep=\";\") data or use another data reader class".format(filenames[0]))
        else:
            self.dataX = dataX
            self.dataY = dataY

        if self.nrowsX() != self._nrowsY():
            raise RuntimeError(
                "ExpDataReaderNPY.__init__: input data set and output data set have not the same number of rows.")
        if self.ncolsX() == 0 or self.ncolsY() == 0:
            raise RuntimeError(
                "ExpDataReaderNPY.__init__: there are no columns in the data provided. Check that the data are numeric")

        self.features_placeholder = tf.placeholder(
            self.dataX.dtype, self.dataX.shape)
        self.labels_placeholder = tf.placeholder(
            self.dataY.dtype, self.dataY.shape)

        self.dataset = tf.contrib.data.Dataset.from_tensor_slices(
            (self.features_placeholder, self.labels_placeholder))
        # [Other transformations on `dataset`...]
        # dataset = ...
        self.iterator = self.dataset.make_initializable_iterator()
        self.input, self.output = self.iterator.get_next()

    def init(self, sess):
        """
        Initialize the operator
        :param sess:
        :return:
        """
        sess.run(
            self.iterator.initializer,
            feed_dict={
                self.features_placeholder: self.dataX,
                self.labels_placeholder: self.dataY})


class ExpCSVDataReader(ExpDataReader):
    def __init__(self, train, batch_size, pathdata=".",
                 filenames=("X.csv", "Y.csv"), sizes=(1, 1), num_thread=4,
                 mX=None, sdX=None, mY=None, sdY=None):
        """
        :param train: if true concern the training set
        :param batch_size: number of data to unpack each time
        :param pathdata: path where the data are stored
        :param filenames: names of the files with input data and output data
        :param sizes: number of columns of the data in X and Y
        :param num_thread: number of thread to read the data
        :param mX: mean of X data set (used for validation instead -- of recomputing the mean)
        :param mY: mean of Y data set (used for validation instead -- of recomputing the mean)
        :param sdX: standard deviation of X data set (used for validation -- instead of recomputing the std)
        :param sdY: standard deviation of Y data set (used for validation -- instead of recomputing the std)
        """
        ExpDataReader.__init__(self, train=train, batch_size=batch_size)
        self.sizes = sizes

        # TODO optimization: do not parse the file if you nrows (training set parsed 2 times)
        # count the number of lines
        if (mX is None) or (sdX is None):
            fun_process = self._normalize
        else:
            fun_process = self._countlines
        mX_, sdX_, self.nrows = fun_process(
            path=pathdata, size=sizes[0], fn=filenames[0])
        if (mX is None) or (sdX is None):
            fun_process = self._normalize
        else:
            fun_process = self._countlines
        mY_, sdY_, tmpN = fun_process(
            path=pathdata, size=sizes[1], fn=filenames[1])

        self.mX = tf.convert_to_tensor(
            mX_, name="mean_X", dtype=tf.float32) if mX is None else mX
        self.sdX = tf.convert_to_tensor(
            sdX_, name="std_X", dtype=tf.float32) if sdX is None else sdX
        self.mY = tf.convert_to_tensor(
            mY_, name="mean_Y", dtype=tf.float32) if mY is None else mY
        self.sdY = tf.convert_to_tensor(
            sdY_, name="std_Y", dtype=tf.float32) if sdY is None else sdY

        if tmpN != self.nrows:
            raise RuntimeError(
                "ExpCSVDataReader: the files {} and {} does not count the same number of lines".format(
                    sizes[0], sizes[1]))

        self.dataX = tf.contrib.data.TextLineDataset(
            [os.path.join(pathdata, filenames[0])]).skip(1).map(
            lambda line: self._parse_function(line, size=sizes[0], m=self.mX, std=self.sdX),
            num_threads=num_thread,
            output_buffer_size=num_thread * 5
        )
        self.dataY = tf.contrib.data.TextLineDataset(
            [os.path.join(pathdata, filenames[1])]).skip(1).map(
            lambda line: self._parse_function(line, size=sizes[1], m=self.mY, std=self.sdY),
            num_threads=num_thread,
            output_buffer_size=num_thread * 5
        )

        self.dataset = tf.contrib.data.Dataset.zip((self.dataX, self.dataY))
        if train:
            self.dataset = self.dataset.repeat(-1)
            self.dataset = self.dataset.shuffle(buffer_size=10000)
        else:
            self.dataset = self.dataset.repeat(1)
        self.dataset = self.dataset.batch(batch_size=batch_size)

    def _normalize(self, path, size, fn):
        """
        Compute some statistics of the file fn.
        fn should be a csv file with a coma separator, copntaining only float objects, with a single header line.
        :param path: the path where the file is
        :param size: dimension of the file (number of columns)
        :param fn: the file name
        :return: the mean, the standard deviation, and the number of rows
        """
        count = 0
        acc = np.zeros(shape=(size))
        acc2 = np.zeros(shape=(size))
        with open(os.path.join(path, fn)) as fp:
            fp.readline()  # do not parse the header
            for (count, li) in enumerate(fp, 1):
                spl = li.split(";")
                acc += [float(el) for el in spl]
                acc2 += [float(el) * float(el) for el in spl]
        acc /= count
        acc2 /= count
        m = acc
        std = np.sqrt(acc2 - acc * acc)
        std[std <= 1e-3] = 1.
        return m, std, count

    def _countlines(self, path, size, fn):
        """
        Compute the number of rows of the file fn.
        fn should be a csv file with a coma separator, copntaining only float objects, with a single header line.
        :param path: the path where the file is
        :param size: dimension of the file (number of columns)
        :param fn: the file name
        :return: the mean, the standard deviation, and the number of rows
        """
        count = 0
        with open(os.path.join(path, fn)) as fp:
            fp.readline()  # do not count the header
            for (count, li) in enumerate(fp, 1):
                pass
        return 0., 1., count

    def _nrows(self, array):
        """
        :param array: unused
        :return: the number of rows in the training set
        """
        return self.nrows

    def _parse_function(self, csv_row, size, m, std):
        """
        Read the data in the csv format
        :param csv_row: the lines to parse
        :param size: the number of columns
        :param m: the mean over the whole data set
        :param std: the standar deviation over the whole data set
        :return:
        """
        # TODO make a cleaner version of preprocessing!
        record_defaults = [[0.0] for _ in range(size)]
        row = tf.decode_csv(
            csv_row,
            record_defaults=record_defaults,
            field_delim=";")
        row = row - m
        row = row / std
        return row

    def _ncols(self, array):
        """
        :param array: tensorflow dataset object
        :return: the number of variables of a parsed array
        """
        # pdb.set_trace()
        return int(array.output_shapes[0])


class ExpData:
    def __init__(self, batch_size=50, sizemax=int(1e4),
                 pathdata=".",
                 classData=ExpDataReader,
                 argsTdata=(), kwargsTdata={},
                 argsVdata=(), kwargsVdata={},
                 ):
        """ The base class for every 'data' subclasses, depending on the problem
        :param batch_size: the size of the minibatch
        :param pathdata: the path where data are stored
        :param sizemax: maximum size of data chunk that will be "fed" to the computation graph
        :param classData: the class of data (should derive from 'ExpDataReader')
        :param argsTdata: default arguments to build an instance of class 'expDataReader' (build the training data set)
        :param kwargsTdata: keywords arguments to build an instance of class 'expDataReader' (build the training data set)
        :param argsVdata: default arguments to build an instance of class 'expDataReader' (build the validation data set)
        :param kwargsVdata: keywords arguments to build an instance of class 'expDataReader' (build the validation data set)
        """
        # the data for training (fitting the models parameters)
        self.trainingData = classData(
            pathdata=pathdata,
            *argsTdata,
            **kwargsTdata,
            train=True,
            batch_size=batch_size)
        # get the values of means and standard deviation of the training set,
        # to be use in the others sets
        mX = self.trainingData.mX
        mY = self.trainingData.mY
        sdX = self.trainingData.sdX
        sdY = self.trainingData.sdY
        # the data for training (only used when reporting error on the whole
        # set)
        self.trainData = classData(
            pathdata=pathdata,
            *argsTdata,
            **kwargsTdata,
            train=False,
            batch_size=sizemax,
            mX=mX,
            mY=mY,
            sdX=sdX,
            sdY=sdY)
        # the data for validation set (fitting the models hyper parameters --
        # only used when reporting error on the whole set)
        self.valData = classData(
            pathdata=pathdata,
            *argsVdata,
            **kwargsVdata,
            train=False,
            batch_size=sizemax,
            mX=mX,
            mY=mY,
            sdX=sdX,
            sdY=sdY)

        self.size_in = self.trainData.ncolsX()
        self.size_out = self.trainData.ncolsY()
        self.sizemax = sizemax

        self.iterator = tf.contrib.data.Iterator.from_structure(
            self.trainData.dataset.output_types, self.valData.dataset.output_shapes)
        self.next_input, self.next_output = self.iterator.get_next(
            name="true_data")

        self.training_init_op = self.iterator.make_initializer(
            self.trainingData.dataset)
        self.train_init_op = self.iterator.make_initializer(
            self.trainData.dataset)
        self.validation_init_op = self.iterator.make_initializer(
            self.valData.dataset)

    def getnrows(self):
        """
        :return: Number of row of the training set
        """
        return self.trainData.nrowsX()

    def fancydescrption(self):
        """
        :return: A description for an instance of this data type
        """
        return "Standard data type"

    def gettype(self):
        """
        :return: the type of data this is.
        """
        return "ExpData"

    def getsizeoutX(self):
        """
        :return: the dimension of input vector (number of variables)
        """
        return self.trainData.ncolsX()

    def getsizeoutY(self):
        """
        :return: the dimension of output vector (number of variables to predict)
        """
        return self.trainData.ncolsY()

    def getnrowsval(self):
        """
        :return: Number of row of the validation set
        """
        return self.valData.nrowsX()

    def computetensorboard(
            self,
            sess,
            graph,
            writers,
            xval,
            minibatchnum,
            sum=False):
        """
        Compute and log (using writers) the errors on the training set and validation set
        Return the error ON THE VALIDATION SET
        :param sess: a tensorflow session
        :param graph: an object of class ExpGraph
        :param writers: an object of class ExpWriter
        :param xval: the index value of the tensorboard run
        :param minibatchnum: the number of minibatches computed
        :param sum: if true make the sum of both losses, other return the error ON THE VALIDATION SET
        :return: THE VALIDATION SET (except if sum is true)
        """
        valloss = np.NaN
        # switch the reader to the the "train" dataset for reporting the
        # training error
        sess.run(self.train_init_op)
        error_nan, trainloss = self.computetensorboard_aux(
            sess=sess, graph=graph, writers=writers, xval=xval, dataset=self.trainData, minibatchnum=minibatchnum, train=True)
        # switch the reader to the the "validation" dataset for reporting the
        # training error
        sess.run(self.validation_init_op)
        if not error_nan:
            error_nan, valloss = self.computetensorboard_aux(
                sess=sess, graph=graph, writers=writers, xval=xval, dataset=self.valData, minibatchnum=minibatchnum, train=False)

        if not sum:
            res = valloss
        else:
            res = trainloss + valloss
        sess.run(self.training_init_op)
        return error_nan, res

    def computetensorboard_aux(
            self,
            sess,
            graph,
            dataset,
            writers,
            xval,
            minibatchnum,
            train=True):
        """
        Compute the error on a whole data set.
        Report the results in the text logger and in tensorboard.
        Chunk of data of size at most 'self.sizemax' are fed in one "chunk"
        :param sess: the tensorflow session
        :param graph: the ExpGraph to be used for the comptuation
        :param dataset: the dataset from which data are read
        :param writers: the Expwriters to be used
        :param xval: the 'x value' to be written in tensorboard
        :param minibatchnum: the current number of minibatches
        :param train: does it concern the training set
        :return:
        """
        n = dataset.nrowsX()
        acc_loss = 0.
        error_nan = False
        # pdb.set_trace()
        while True:
            # TODO GD here : check that the "huge batch" concern the same
            # disconnected quad
            try:
                summary, loss_ = graph.run(
                    sess, toberun=[
                        graph.mergedsummaryvar, graph.loss])
                if train:
                    writers.tfwriter.trainwriter.add_summary(summary, xval)
                else:
                    writers.tfwriter.valwriter.add_summary(summary, xval)
                acc_loss += loss_
                error_nan = not np.isfinite(loss_)
                if error_nan:
                    break
            except tf.errors.OutOfRangeError:
                break
        # pdb.set_trace()
        name = "Train" if train else "Validation"
        writers.logger.info(
            "{} l2 error after {} minibatches : {}".format(
                name, minibatchnum, acc_loss))
        return error_nan, acc_loss

    def computelasterror(self, sess, graph, logger, params, dict_summary, valSet=True, varsY=None):  # TODO
        """
        :param sess: a tensorflow session
        :param graph: an object of type "ExpGraph" or one of its derivatives
        :param logger: an object of class "ExpLogger" or one of its derivatives  #TODO ExpLogger
        :param params: an object of class "ExpParams" or one of its derivatives  #TODO ExpParams
        :param dict_summary: the summary to update, which will be written byt the "Experiment" class
        :return: #TODO complete here a viable template of this function
        """
        return None, None

    def combinerescompute(self, res, curr, toberun, indx=None):
        """
        Combines the previous results (in res) and the new one (in curr) to get the new one
        :param res: previous results
        :param curr: last results computed
        :param toberun: what have been run
        :param indx: on which part of the data
        :return:
        """
        if toberun is None:
            res += np.float32(curr)  # TODO false in case of mean for example!
        else:
            res[indx, :] = curr
        return res

    def initres(self, dataset, size=0):
        """
        Init the results structure to store the results of a forward pass on the whole 'dataset'
        :param dataset: the data for which the resutls will be computed
        :param size: the size you want (by default dataset.nrowsX()) [unused in this version]
        :return:
        """
        init = np.ndarray(
            (dataset.nrowsX(),
             dataset.ncolsY()),
            dtype="float32")
        return init

    def getpred(self, sess, graph, train=True):
        """
        return the prediction made by 'graph' on the data
        :param sess: a tensorflow sessions
        :param graph: an object of class ExpGraph, representing a comptuation graph / neural network
        :param train: do you want to output the prediction on the traininset or on the validation set
        :return:
        """

        dataset = self.trainData if train else self.valData
        n = dataset.nrowsX()
        n = int(n)

        res = self.initres(dataset=dataset, size=0)

        tmp = int(0)
        self.comptime = 1.0
        while tmp < n:
            limit = int(np.min([tmp + self.sizemax, n]))
            indx = [int(el) for el in range(tmp, limit)]

            beg__ = datetime.datetime.now()
            _, *dat = self.gettfbatch(newepoch=True,
                                      dataset=dataset, indxs=indx)
            end__ = datetime.datetime.now()
            curr = graph.run(sess, toberun=graph.getoutput(), data=dat)
            tmpTime = end__ - beg__
            self.comptime += tmpTime.total_seconds()

            res = self.combinerescompute(
                res=res, curr=curr, toberun=graph.getoutput(), indx=indx)
            tmp = limit
        return res

    def getinputs(self):
        """
        :return: The tensor of input data as well as its size
        """
        return self.next_input, int(self.trainData.dataset.output_shapes[0][1])

    def getoutputs(self):
        """
        :return: The tensor of input data as well as its size
        """
        return self.next_output, int(
            self.trainData.dataset.output_shapes[1][1])

    def init(self, sess):
        """
        initialize the data if needed
        :param sess:
        """
        sess.run(self.training_init_op)

    def nextminibatch(self, minibatch_size=100):  # TODO
        """
        Return a new minibatch of training data
        :param minibatch_size: the size of the minibatch you want to get
        :return:
        """
        if minibatch_size == 0:
            raise RuntimeError(
                "StoredData.nextminibatch : you must ask at least one example")
        if minibatch_size > self.getnrows():
            raise RuntimeError(
                "StoredData.nextminibatch :not enough data to have a minibatch this big")
        return 0.0

    def gettfbatch(self, indxs, dataset, newepoch=True):  # TODO
        """
        Return the data indexed by indxs of the training set (if train=True) or of the validation set (if not)
        :param train:
        :param indxs:
        :return:
        """
        res = 0.
        # res = dataset.get(indxs)
        return newepoch, res

    def computetensorboard_aux_with_feeddict(
            self,
            sess,
            graph,
            dataset,
            writers,
            xval,
            minibatchnum,
            train=True):
        """
        Compute the error on a whole data set.
        Report the results in the text logger and in tensorboard.
        Chunk of data of size at most 'self.sizemax' are fed in one "chunk"
        :param sess: the tensorflow session
        :param graph: the ExpGraph to be used for the comptuation
        :param dataset: the dataset from which data are read
        :param writers: the Expwriters to be used
        :param xval: the 'x value' to be written in tensorboard
        :param minibatchnum: the current number of minibatches
        :param train: does it concern the training set
        :return:
        """
        n = dataset.nrowsX()
        n = int(n)
        tmp = int(0)
        acc_loss = 0.
        error_nan = False
        while tmp < n:
            limit = int(np.min([tmp + self.sizemax, n]))
            indx = [el for el in range(tmp, limit)]
            # TODO GD here : check that the "huge batch" concern the same
            # disconnected quad
            _, *dat = self.gettfbatch(newepoch=True,
                                      dataset=dataset, indxs=indx)

            summary, loss_ = graph.run(
                sess, toberun=[
                    graph.mergedsummaryvar, graph.getloss], data=dat)
            if train:
                writers.tfwriter.trainwriter.add_summary(summary, xval)
            else:
                writers.tfwriter.valwriter.add_summary(summary, xval)
            acc_loss += loss_
            error_nan = not np.isfinite(loss_)
            if error_nan:
                break
            tmp = limit
        name = "Train" if train else "Validation"
        writers.logger.info(
            "{} l2 error after {} minibatches : {}".format(
                name, minibatchnum, acc_loss))
        return error_nan, acc_loss


class ExpGraph:
    def __init__(self, input, output, nnType=NNFully, argsNN=(), kwargsNN={}):
        """The base class for every 'Graph' subclass to be use in with Experiment.
        Basically, this represent the neural network.
        :param inputs: the input of the graph (node in a tf computation graph + its size)
        :param nnType: the type of neural network to use
        :param args forwarded to the initializer of neural network
        :param kwargsNN: key word arguments forwarded to the initializer of neural network
        """
        # TODO make the class NNFully directly a descendant of this class!
        # no need to make self.nn here!
        self.nn = nnType(
            input=input[0],
            outputsize=output[1],
            *argsNN,
            **kwargsNN)
        self.vars_out = self.nn.pred

        self.mergedsummaryvar = None
        self.loss = None

    def getnbparam(self):
        """Return the number of total free parameters of the neural network"""
        return self.nn.getnbparam()

    def getoutput(self):
        """
        :return: The "last" node of the graph, that serves as output
        """
        return self.vars_out

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
        with tf.variable_scope("training_loss"):
            self.loss = self.lossfun(
                data.getoutputs()[0] -
                self.inference,
                name="loss")
        with tf.variable_scope("optimizer"):
            self.optimize = optimizerClass(
                **optimizerkwargs).minimize(loss=self.loss, name="optimizer")

        # 3. build the summaries that will be stored
        with tf.variable_scope("summaries"):
            self.error = tf.add(
                self.inference, -data.getoutputs()[0], name=netname + "error_diff")
            self.error_abs = tf.abs(self.error)
            self.l1_avg = tf.reduce_mean(
                self.error_abs, name=netname + "l1_avg")
            self.l2_avg = tf.reduce_mean(
                self.error * self.error, name=netname + "l2_avg")
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
        print(
            "{:%H:%M:%S.%f} minibatchnum {}".format(
                datetime.datetime.now(),
                self.exp_params.minibatchnum))

        return newepoch, is_error_nan, losscomputed, valloss, timedata, timeTrain, timesaving

    def run_withfeed_dict(self, sess):
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


class ExpParams:
    def __init__(self, epochnum=1, batch_size=50):
        self.num_epoch = epochnum
        self.batchsize = batch_size


class Exp:
    def __init__(self,
                 parameters,
                 dataClass=ExpData, dataargs=(), datakwargs={},
                 graphType=ExpGraph, graphargs=(), graphkwargs={},
                 modelType=ExpModel, modelargs=(), modelkwargs={},

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
        self.inputs, self.inputsize = self.data.getinputs()
        self.outputs, self.outputsize = self.data.getoutputs()

        # 2. build the graph
        self.graph = None
        with tf.variable_scope("neuralnetwork"):
            self.graph = graphType(input=(self.inputs, self.inputsize),
                                   output=(self.outputs, self.outputsize),
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

        # 6. create the path

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
                self.data.getsizeoutX(),
                self.data.getsizeoutY()))
        self.model.explogger.info(
            "Size of validation set : nsample {}, size_X {},size_Y {} ".format(
                self.data.getnrowsval(),
                self.data.getsizeoutX(),
                self.data.getsizeoutY()))

        dict_summary = {}
        dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
        dict_summary["flops"] = "{}".format(self.graph.get())
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
        dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
        dict_summary["flops"] = "{}".format(self.graph.get())

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
            dict_summary["flops"] = "{}".format(self.graph.get())
            dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
            # dict_summary["l1_val_loss"] += "{}".format(self.vallos[-1])
        else:
            dict_summary = {}
            dict_summary["training_time"] = self.timeTrain
            dict_summary["training_steps"] = self.trainingsteps
            dict_summary["data_getting_time"] = self.timedata
            dict_summary["data_saving_time"] = self.timesaving
            dict_summary["l1_val_loss"] = "{}".format(self.valloss[-1])
            dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())
            dict_summary["flops"] = "{}".format(self.graph.get())
            dict_summary["nb_params"] = "{}".format(self.graph.getnbparam())

        self.writesummaryExp(dict_summary)
        self.model.saveTrainedNet(
            sess=self.sess,
            minibatchnum=self.parameters.minibatchnum,
            valloss=self.valloss)

    def writesummaryExp(self, dict_summary):
        with open(os.path.join(self.path, "summary.json"), "r") as f:
            dict_saved = json.load(f)

        with open(os.path.join(self.path, "summary.json"), "w") as f:
            for key, val in dict_summary.items():
                dict_saved[key] = val
            json.dump(dict_summary, f, sort_keys=True, indent=4)


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
