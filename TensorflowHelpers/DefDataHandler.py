import numpy as np
import tensorflow as tf
import datetime
import os

import pdb

#TODO make this class more compliant with the other version
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

        self.ms = {"input": tf.constant(0.), "output": tf.constant(0.)}
        self.sds = {"input": tf.constant(1.), "output": tf.constant(1.)}

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

    def _nrows(self, array):
        """
        :param array: the concerned "array"
        :return: The number of  rows / examples of the data set  "array"
        """
        return array.shape[0]

    def init(self, sess):
        """
        Initialize the datasets (if need to be done)
        :return:
        """
        pass

#TODO refactor ExpCSVDataReader and ExpTFrecordDataReader
# TODO at least _nrows, _ncols and _shape_properly are copy paste.
# TODO beside the logic is exactly the same!
class ExpCSVDataReader(ExpDataReader):
    def __init__(self, train, batch_size, pathdata=".",
                 filenames={"input": "X.csv", "output": "Y.csv"},
                 sizes={"input":1, "output":1},
                 num_thread=4,
                 ms=None, sds=None):
        """
        :param train: if true concern the training set
        :param batch_size: number of data to unpack each time
        :param pathdata: path where the data are stored
        :param filenames: names of the files with input data and output data [should be a 2 keys dictionnaries with keys "input" and "output"]
        :param sizes: number of columns of the data in X and Y [should be a 2 keys dictionnaries with keys "input" and "output"]
        :param other_dataset: other files (same format as
        :param num_thread: number of thread to read the data
        :param ms: means of X data set (used for validation instead -- of recomputing the mean)
        :param sds: standard deviation of Y data set (used for validation -- instead of recomputing the std)
        """
        ExpDataReader.__init__(self, train=train, batch_size=batch_size)
        self.sizes = sizes

        # TODO optimization: do not parse the file if you nrows (training set parsed 2 times)
        # count the number of lines
        if (ms is None) or (sds is None):
            fun_process = self._normalize
        else:
            fun_process = self._countlines

        ms_, sds_, self.nrows = fun_process(path=pathdata, fns=filenames, sizes=sizes)
        self.ms = self._shape_properly(ms_) if ms is None else ms
        self.sds = self._shape_properly(sds_) if sds is None else sds

        self.dataX = tf.contrib.data.TextLineDataset(
            [os.path.join(pathdata, filenames[0])]).skip(1).map(
            lambda line: self._parse_function(line, size=sizes[0], m=self.ms["input"], std=self.sds["input"]),
            num_threads=num_thread,
            output_buffer_size=num_thread * 5
        )
        self.dataY = tf.contrib.data.TextLineDataset(
            [os.path.join(pathdata, filenames[1])]).skip(1).map(
            lambda line: self._parse_function(line, size=sizes[1], m=self.ms["output"], std=self.sds["output"]),
            num_threads=num_thread,
            output_buffer_size=num_thread * 5
        )

        self.dataset = tf.contrib.data.Dataset.zip({"input": self.dataX, "output": self.dataY})
        if train:
            self.dataset = self.dataset.repeat(-1)
            self.dataset = self.dataset.shuffle(buffer_size=10000)
        else:
            self.dataset = self.dataset.repeat(1)
        self.dataset = self.dataset.batch(batch_size=batch_size)



    def _normalize(self, path, sizes, fns):
        """
        Compute some statistics of the files fns.
        :param path: the path where data are located
        :param sizes: a dictionnary (with keys ["input", "output"]) containing the number of rows of the datasets
        :param fns: a dictionnary (with keys ["input", "output"]) containing the names of the datasets
        :return: 
        """
        mX, sdX, nrowsX = self._normalize_aux(path, size=sizes["input"], fn=fns["input"])
        mY, sdY, nrowsY = self._normalize_aux(path, size=sizes["output"], fn=fns["output"])
        if nrowsX != nrowsY:
            error_str = "ExpCSVDataReader._normalize: The files {} and {} (located at {}) "
            error_str += "does not count the same number of lines."
            raise RuntimeError(error_str.format(fns["input"], fns["output"], path))

        ms = {"input": mX, "output": mY}
        sds = {"input": sdX, "output": sdY}
        return ms, sds, nrowsX

    def _normalize_aux(self, path, size, fn):
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
        count += 1
        acc /= count
        acc2 /= count
        m = acc
        std = np.sqrt(acc2 - acc * acc)
        std[std <= 1e-3] = 1.
        return m, std, count

    def _countlines(self, path, sizes, fns):
        """
        Compute the number of lines of both files fns (and check they match).
        each file in fns should be a csv file with a semi-colon separator, copntaining only float objects, with a single header line.
        :param path: the path where the file is
        :param size: dimension of the file (number of columns)
        :param fns: the file name
        :return: the mean, the standard deviation, and the number of rows
        """
        cX = self._countlines_aux(path, fn=fns["input"])
        cY = self._countlines_aux(path, fn=fns["input"])
        if cX != cY:
            error_str = "ExpCSVDataReader._normalize: The files {} and {} (located at {}) "
            error_str += "does not count the same number of lines."
            raise RuntimeError(error_str.format(fns["input"], fns["output"], path))

        ms = {"input": 0., "output": 0.}
        sds = {"input": 1., "output": 1.}
        return ms, sds, cX

    def _countlines_aux(self, path, fn):
        """
        Compute the number of rows of the file fn.
        fn should be a csv file with a coma separator, copntaining only float objects, with a single header line.
        :param path: the path where the file is located
        :param fn: the file name
        :return: the mean, the standard deviation, and the number of rows
        """
        count = 0
        with open(os.path.join(path, fn)) as fp:
            fp.readline()  # do not count the header
            for (count, li) in enumerate(fp, 1):
                pass
        return count

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

    def _nrows(self, array):
        """
        :param array: unused
        :return: the number of rows in the training set
        """
        return self.nrows

    def _shape_properly(self, ms):
        """
        Transform a dictionnary of numpy array in a dictionnary of tensorflow tensor
        :param ms: 
        :return: 
        """
        return {k: tf.convert_to_tensor(v, name="mean_{}".format(k), dtype=tf.float32) for k, v in ms.items()}

class ExpTFrecordsDataReader(ExpDataReader):
    def __init__(self, train, batch_size, pathdata=".",
                 filename="data.tfrecord",
                 vars={"input", "output"},
                 sizes={"input":1, "output":1},
                 num_thread=4,
                 ms=None, sds=None):
        """
        ms (and sds) should be None or dictionnaries with at least the keys in vars, and tensorflow float32 tensors as values
        
        :param train: if true concern the training set
        :param batch_size: number of data to unpack each time
        :param pathdata: path where the data are stored
        :param filenames: names of the files with input data and output data
        :param sizes: number of columns of the data in X and Y
        :param num_thread: number of thread to read the data
        :param ms: means of data set (used for validation instead -- of recomputing the mean). 
        :param sds: standard deviation of data set (used for validation instead -- of recomputing the mean)
        """
        #TODO handle case where there are multiple tfrecords !

        ExpDataReader.__init__(self, train=train, batch_size=batch_size)
        self.sizes = sizes
        self.num_thread = num_thread
        self.batch_size = batch_size

        # TODO optimization: do not parse the file if you nrows (training set parsed 2 times)
        # count the number of lines
        if (ms is None) or (sds is None):
            fun_process = self._normalize
        else:
            fun_process = self._countlines

        ms_, sds_, self.nrows = fun_process(
            path=pathdata, fn=filename, sizes=sizes)

        self.ms = self._shape_properly(ms_) if ms is None else ms
        self.sds = self._shape_properly(sds_) if sds is None else sds

        self.dataset = tf.contrib.data.TFRecordDataset(
            [os.path.join(pathdata, filename)]).map(
            lambda line: self._parse_function(example_proto=line, sizes=sizes, ms=self.ms, stds=self.sds),
            num_threads=num_thread,
            output_buffer_size=num_thread * 5
        )
        if train:
            self.dataset = self.dataset.repeat(-1)
            self.dataset = self.dataset.shuffle(buffer_size=10000)
        else:
            self.dataset = self.dataset.repeat(1)
        self.dataset = self.dataset.batch(batch_size=batch_size)

    def _countlines(self, path, fn, sizes):
        """
        :param path: the path where data are located
        :param fn: the file name
        :return: the number of lines of the files (must iterate through it line by line) 
        """
        ms = {el: 0. for el in sizes}
        sds = {el: 0. for el in sizes}
        nb = 0
        fn_ = os.path.join(path, fn)
        for nb, record in enumerate(tf.python_io.tf_record_iterator(fn_)):
            pass
        # don't forget to add 1 because python start at 0!
        return ms, sds, nb+1

    def _parse_function(self, example_proto, sizes, ms, stds):
        """
        :param example_proto: 
        :param sizes: 
        :param ms: 
        :param stds: 
        :return: 
        """
        features = {k: tf.FixedLenFeature((val,), tf.float32, default_value=[0.0 for _ in range(val)])
                    for k, val in sizes.items()}
        parsed_features = tf.parse_single_example(example_proto, features)
        for k in sizes.keys():
            parsed_features[k] = parsed_features[k] - ms[k]
            parsed_features[k] = parsed_features[k]/stds[k]

        return parsed_features

    def _normalize(self, path, fn, sizes):
        """
        Compute some statistics of the file fn.
        fn should be a csv file with a coma separator, copntaining only float objects, with a single header line.
        side effect: will created some node in the tensorflow graph just to compute these statistics
        
        :param path: the path where the file is
        :param sizes: dimension of the file (number of columns) per type of data (dictionnary)
        :param fn: the file name
        :return: the mean, the standard deviation, and the number of rows
        """

        acc = { k:np.zeros(shape=(v)) for k,v in sizes.items() }
        acc2 = { k:np.zeros(shape=(v)) for k,v in sizes.items() }

        with tf.variable_scope("datareader_compute_means_vars"):
            ms = {k:tf.constant(0.0, name="fake_means") for k,_ in sizes.items()}
            sds = {k:tf.constant(1.0, name="fake_stds") for k,_ in sizes.items()}
            dataset = tf.contrib.data.TFRecordDataset(
                [os.path.join(path, fn)]).map(
                lambda line: self._parse_function(example_proto=line, sizes=sizes, ms=ms, stds=sds),
                num_threads=self.num_thread,
                output_buffer_size=self.num_thread * 5
            ).repeat(1)
            iterator = dataset.make_one_shot_iterator()
            parsed_features = iterator.get_next(name="fake_iterator")

            # I need a session to parse the features
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # from tqdm import tqdm
            count = 0
            with tf.Session(config=config) as sess:
                while True:
                    count += 1
                    try:
                        pf = sess.run(parsed_features)
                        for k in sizes.keys():
                            vect = pf[k]
                            acc[k] += vect
                            acc2[k] += vect*vect
                    except tf.errors.OutOfRangeError:
                        break
            acc = {k: v/count for k,v in acc.items()}
            acc2 = {k: v/count for k,v in acc2.items()}

            ms = acc
            stds = {k: np.sqrt(acc2[k] - v * v) for k,v in acc.items()}
            for k,v in stds.items():
                stds[k][stds[k] <= 1e-3] = 1.0
        return ms, stds, count

    def _shape_properly(self, ms):
        """
        Transform a dictionnary of numpy array in a dictionnary of tensorflow tensor
        :param ms: 
        :return: 
        """
        return {k: tf.convert_to_tensor(v, name="mean_{}".format(k), dtype=tf.float32) for k, v in ms.items()}

    def _nrows(self, array):
        """
        :param array: unused
        :return: the number of rows in the training set
        """
        return self.nrows

class ExpData:
    def __init__(self, batch_size=50, sizemax=int(1e4),
                 pathdata=".",
                 classData=ExpDataReader,
                 sizes={"input":1, "output":1},
                 argsTdata=(), kwargsTdata={},
                 argsVdata=(), kwargsVdata={},
                    otherdsinfo = {}
                 ):
        """ The base class for every 'data' subclasses, depending on the problem
        :param batch_size: the size of the minibatch
        :param pathdata: the path where data are stored
        :param sizemax: maximum size of data chunk that will be "fed" to the computation graph
        :param classData: the class of data (should derive from 'ExpDataReader')
        :param sizes: the sizes (number of columns) of each dataset
        :param argsTdata: default arguments to build an instance of class 'expDataReader' (build the training data set)
        :param kwargsTdata: keywords arguments to build an instance of class 'expDataReader' (build the training data set)
        :param argsVdata: default arguments to build an instance of class 'expDataReader' (build the validation data set)
        :param kwargsVdata: keywords arguments to build an instance of class 'expDataReader' (build the validation data set)
        :param otherdsinfo : dictionnaries of keys = dataset name, values = dictionnaries of keys: "argsdata" : tuple, kwargsdata: dictionnaries
        """
        # the data for training (fitting the models parameters)
        self.trainingData = classData(
            pathdata=pathdata,
            *argsTdata,
            **kwargsTdata,
            sizes=sizes,
            train=True,
            batch_size=batch_size)
        # get the values of means and standard deviation of the training set,
        # to be use in the others sets
        self.ms = self.trainingData.ms
        self.sds = self.trainingData.sds
        # the data for training (only used when reporting error on the whole
        # set)
        self.trainData = classData(pathdata=pathdata,
                                   *argsTdata,
                                   **kwargsTdata,
                                   sizes=sizes,
                                   train=False,
                                   batch_size=sizemax,
                                   ms=self.ms,
                                   sds=self.sds)
        # the data for validation set (fitting the models hyper parameters --
        # only used when reporting error on the whole set)
        self.valData = classData(pathdata=pathdata,
                                 *argsVdata,
                                 **kwargsVdata,
                                 sizes=sizes,
                                 train=False,
                                 batch_size=sizemax,
                                 ms=self.ms,
                                 sds=self.sds)
        self.sizemax = sizemax # size maximum of a "minibatch" eg the maximum number of examples that will be fed
        # at once for making a single forward computation

        self.iterator = tf.contrib.data.Iterator.from_structure(
            output_types=self.trainingData.dataset.output_types,
            output_shapes=self.trainingData.dataset.output_shapes)

        self.true_data = self.iterator.get_next(
            name="true_data")

        self.training_init_op = self.iterator.make_initializer(
            self.trainingData.dataset)
        self.train_init_op = self.iterator.make_initializer(
            self.trainData.dataset)
        self.validation_init_op = self.iterator.make_initializer(
            self.valData.dataset)

        self.otherdatasets = {}
        self.otheriterator_init = {}
        for otherdsname, values in otherdsinfo.items():
            self.otherdatasets[otherdsname] = classData(pathdata=pathdata,
                                                        *values["argsdata"],
                                                        **values["kwargsdata"],
                                                        sizes=sizes,
                                                        train=False,
                                                        batch_size=sizemax,
                                                        ms=self.ms,
                                                        sds=self.sds
                                                        )
            self.otheriterator_init[otherdsname] = self.iterator.make_initializer(self.otherdatasets[otherdsname].dataset)

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
        # TODO why is it in data ?
        valloss = np.NaN
        # switch the reader to the the "train" dataset for reporting the
        # training error
        sess.run(self.train_init_op)
        error_nan, trainloss = self.computetensorboard_aux(
            sess=sess, graph=graph, writer=writers.tfwriter.trainwriter, xval=xval,
            minibatchnum=minibatchnum, train=True, name="Train", textlogger=writers.logger)
        # switch the reader to the the "validation" dataset for reporting the
        # training error
        sess.run(self.validation_init_op)
        if not error_nan:
            error_nan, valloss = self.computetensorboard_aux(
                sess=sess, graph=graph, writer=writers.tfwriter.valwriter, xval=xval,
                minibatchnum=minibatchnum, train=False, name="Validation", textlogger=writers.logger)

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
            writer,
            textlogger,
            name,
            xval,
            minibatchnum,
            train=True):
        """
        Compute the error on a whole data set.
        Report the results in the text logger and in tensorboard.
        Chunk of data of size at most 'self.sizemax' are fed in one "chunk"
        :param sess: the tensorflow session
        :param graph: the ExpGraph to be used for the comptuation
        :param writer: the tensorflow writer to use
        :param textlogger: the text logger to use to store the information
        :param name: the name displayed on the file logger
        :param xval: the 'x value' to be written in tensorboard
        :param minibatchnum: the current number of minibatches
        :param train: does it concern the training set
        :return:
        """
        # TODO why is it in data ?
        acc_loss = 0.
        error_nan = False
        while True:
            # TODO GD here : check that the "huge batch" concern the same
            # TODO disconnected quad
            try:
                summary, loss_ = graph.run(
                    sess, toberun=[
                        graph.mergedsummaryvar, graph.loss])
                if train:
                    writer.add_summary(summary, xval)
                else:
                    writer.add_summary(summary, xval)
                acc_loss += loss_
                error_nan = not np.isfinite(loss_)
                if error_nan:
                    break
            except tf.errors.OutOfRangeError:
                break
        # name = "Train" if train else "Validation"
        textlogger.info(
            "{} l2 error after {} minibatches : {}".format(
                name, minibatchnum, acc_loss))
        return error_nan, acc_loss

    def getpred(self, sess, graph, varsname):
        """
        :param sess: a tensorflow session
        :param graph: an object of class 'ExpGraph' or one of its derivatives
        :param the variable for which you want to do the comptuation
        :return: the prediction for the validation test (rescaled, and "un preprocessed" (eg directly comparable to the original data) ) 
        :return: the original data
        :return: the predictions takes the form of a dictionnary k: name, value: value predicted
        """
        #TODO why is it in data ?
        res = {k: np.zeros(shape=(self.valData.nrowsX(), int(self.ms[k].shape[0]))) for k in varsname}
        orig = {k: np.zeros(shape=(self.valData.nrowsX(), int(self.ms[k].shape[0]))) for k in varsname}
        sess.run(self.validation_init_op)
        previous = 0
        sds = sess.run(self.sds)
        ms = sess.run(self.ms)
        while True:
            # TODO GD here : check that the "huge batch" concern the same
            # TODO disconnected quad
            try:
                #TODO optim: do not compute irrelevant data, eg data not in varsname
                # TODO it is done when calling self.true_data
                preds = graph.run(sess, toberun=[graph.vars_out, self.true_data] )
                size = 0
                for k in res.keys():
                    # getting the prediction
                    tmp = preds[0][k]
                    size = tmp.shape[0]
                    # rescale it ("un preprossed it")
                    tmp = tmp*sds[k]+ms[k]
                    # storing it in res
                    res[k][previous:(previous+size), :] = tmp

                    tmp = preds[1][k]
                    # rescale it ("un preprossed it")
                    tmp = tmp * sds[k] + ms[k]
                    # storing it in res
                    orig[k][previous:(previous + size), :] = tmp

                previous += size
            except tf.errors.OutOfRangeError:
                break
        sess.run(self.training_init_op)
        return res, orig

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

    def getdata(self):
        """
        :return: the data read and parsed dictionnary with: key=name, values=the associated tensor
        """
        return self.true_data

    def init(self, sess):
        """
        initialize the data if needed
        :param sess:
        """
        sess.run(self.training_init_op)

    def computetensorboard_annex(self, sess, writers, graph, xval, minibatchnum, name):
        """
        Will compute the error on the data "referenced" by "name", and store it using the TFWriters "writers"
        :param sess: a tensorflow session
        :param graph: an object of class ExpGraph
        :param writers: an object of class ExpWriter
        :param xval: the index value of the tensorboard run
        :param minibatchnum: the number of minibatches computed
        :param name: the name of the dataset you want the error from
        :return: nothing
        """
        if not name in self.otheriterator_init:
            error_str = "ExpData.computetensorboard_annex you ask to compute the error on the dataset name \"{}\""
            error_str += " but it does not exists.\nMake sure to have passed the proper \"otherdsinfo\" arguments"
            error_str += " when you build your ExpData. For example :\n"
            error_str += "otherdsinfo={{ \"{}\":{{\"argsdata\":(), \"kwargsdata\": {{ \"filename\": "
            error_str += "\"test_example.tfrecord\" }} }} }} }} \n"
            error_str += "if you want to link the dataset with data coming from \"test_example.tfrecord\" to the name \"{}\""
            raise RuntimeError(error_str.format(name, name, name))
        sess.run(self.otheriterator_init[name])
        self.computetensorboard_aux(
            sess=sess, graph=graph, writer=writers.tfwriter.othersavers[name], xval=xval,
            minibatchnum=minibatchnum, train=False, name=name, textlogger=writers.logger)
        sess.run(self.training_init_op)

if __name__=="__main__":
    test_tf_nb_lines = ExpTFrecordsDataReader(train=True, batch_size=1)
    print("The number of lines computed is {}".format(test_tf_nb_lines._countlines(path="/home/bdonnot/Documents/PyHades2/tfrecords_118_5000",
                                                                                   fn="neighbours-test.tfrecord",
                                                                                   sizes={"prod_q": 54})))
    print("Test lines parsing {}".format(test_tf_nb_lines._normalize(path="/home/bdonnot/Documents/PyHades2/tfrecords_118_5000",
                                                                     fn="neighbours-test.tfrecord",
                                                                     sizes={"prod_q":54})))