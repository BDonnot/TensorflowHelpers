import random
import datetime
import os

import pdb

import numpy as np
import tensorflow as tf

from .ANN import DTYPE_USED, DTYPE_NPY

# TODO READER: make them behave equally well, for now only TFRecordReader can preprocess for example
# TODO: READERS: correct the but when encountering nan's or infinite value in all reader classes
# TODO: have a better implementation of preprocessing function

class ExpDataReader:
    ms_tensor=True # is the 'ms' field a tensor or a numpy array
    def __init__(self, train, batch_size, fun_preprocess=lambda x: x):
        """Read the usefull data for the Experience to run
        Store both input and outputs
        :param train: if True this data reader concerns the training set
        :param batch_size: number of data to read each time the iterator is called
        """
        self.dataX = np.zeros(
            (0, 0), dtype=DTYPE_NPY)  # main data set (eg the training set)
        # main data set (eg the training set)
        self.dataY = np.zeros((0, 0), dtype=DTYPE_NPY)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.dataX, self.dataY))

        self.ms = {"input": tf.constant(0., dtype=DTYPE_USED), "output": tf.constant(0., dtype=DTYPE_USED)}
        self.sds = {"input": tf.constant(1., dtype=DTYPE_USED), "output": tf.constant(1., dtype=DTYPE_USED)}

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

    def _shape_properly(self, ms, name):
        """
        Transform a dictionnary of numpy array in a dictionnary of tensorflow tensor
        :param ms: 
        :param name: 
        :return: 
        """
        return {k: tf.convert_to_tensor(v, name="{}_{}".format(name, k), dtype=DTYPE_USED) for k, v in ms.items()}

# TODO refactor ExpCSVDataReader and ExpTFrecordDataReader
# TODO at least _nrows, _ncols and _shape_properly are copy paste.
# TODO beside the logic is exactly the same!
class ExpCSVDataReader(ExpDataReader):
    def __init__(self, train, batch_size, pathdata=".",
                 filename={"input": "X.csv", "output": "Y.csv"},
                 sizes={"input":1, "output":1},
                 num_thread=4, donnotcenter={},
                 fun_preprocess=lambda x: x,
                 ms=None, sds=None):
        """
        :param train: if true concern the training set
        :param batch_size: number of data to unpack each time
        :param pathdata: path where the data are stored
        :param filenames: names of the files with input data and output data [should be a  dictionnary with keys as sizes]
        :param sizes: number of columns of the data in X and Y [should be a 2 keys dictionnaries with keys "input" and "output"]
        :param other_dataset: other files (same format as
        :param num_thread: number of thread to read the data
        :param donnotcenter: iterable: variable that won't be centered/reduced
        :param ms: means of X data set (used for validation instead of recomputing the mean)
        :param sds: standard deviation of Y data set (used for validation instead of recomputing the std)
        """
        ExpDataReader.__init__(self, train=train, batch_size=batch_size)
        self.sizes = sizes

        # TODO optimization: do not parse the file if you nrows (training set parsed 2 times)
        # count the number of lines
        if (ms is None) or (sds is None):
            fun_process = self._normalize
        else:
            fun_process = self._countlines

        ms__, sds__, self.nrows = fun_process(path=pathdata, fns=filename, sizes=sizes)

        ms_ = {}
        sds_ = {}
        for k in ms__.keys():
            if k in donnotcenter:
                ms_[k] = np.zeros(ms__[k].shape, dtype=DTYPE_NPY)
                sds_[k] = np.ones(sds__[k].shape, dtype=DTYPE_NPY)
            else:
                ms_[k] = ms__[k]
                sds_[k] = sds__[k]

        self.ms = self._shape_properly(ms_, name="means") if ms is None else ms
        self.sds = self._shape_properly(sds_, name="stds") if sds is None else sds

        self.datasets = {}

        for el in sizes.keys():
            self.datasets[el] = tf.data.TextLineDataset(
                [os.path.join(pathdata, filename[el])]).skip(1).map(
                lambda line: self._parse_function(line, size=sizes[el], m=self.ms[el], std=self.sds[el]),
                num_parallel_calls=num_thread
            ).prefetch(num_thread * 5)

        self.dataset = tf.data.Dataset.zip(self.datasets)
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

        ms = {}
        sds = {}
        nrows = None
        prev_files = set()
        for key in sizes.keys():
            m, sd, nrows_tmp = self._normalize_aux(path, size=sizes[key], fn=fns[key])
            ms[key] = m
            sds[key] = sd
            if nrows is not None:
                if nrows_tmp != nrows:
                    error_str = "ExpCSVDataReader._normalize: The files {} and {} (located at {}) "
                    error_str += "does not count the same number of lines."
                    raise RuntimeError(error_str.format(fns["input"], prev_files, path))
            prev_files.add(fns[key])

        # mX, sdX, nrowsX = self._normalize_aux(path, size=sizes["input"], fn=fns["input"])
        # mY, sdY, nrowsY = self._normalize_aux(path, size=sizes["output"], fn=fns["output"])
        # if nrowsX != nrowsY:
        #     error_str = "ExpCSVDataReader._normalize: The files {} and {} (located at {}) "
        #     error_str += "does not count the same number of lines."
        #     raise RuntimeError(error_str.format(fns["input"], fns["output"], path))
        #
        # ms = {"input": mX, "output": mY}
        # sds = {"input": sdX, "output": sdY}
        return ms, sds, nrows

    def _normalize_aux(self, path, size, fn):
        """
        Compute some statistics of the file fn.
        fn should be a csv file with a semi-colon separator, copntaining only float objects, with a single header line.
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
        m = {k:v.astype(DTYPE_NPY) for k, v in m}
        std = {k:v.astype(DTYPE_NPY) for k, v in std}
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

        nrows = None
        prev_files = set()
        for key in sizes.keys():
            nrows_tmp = self._countlines_aux(path, fn=fns[key])
            if nrows is not None:
                if nrows_tmp != nrows:
                    error_str = "ExpCSVDataReader._normalize: The files {} and {} (located at {}) "
                    error_str += "does not count the same number of lines."
                    raise RuntimeError(error_str.format(fns[key], prev_files, path))
            nrows = nrows_tmp
            prev_files.add(fns[key])
        ms = {k: np.zeros(v, dtype=DTYPE_NPY) for k, v in sizes.items()}
        sds = {k: np.ones(v, dtype=DTYPE_NPY) for k, v in sizes.items()}
        return ms, sds, nrows

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
        row = tf.decode_csv(csv_row,
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

class ExpTFrecordsDataReader(ExpDataReader):
    def __init__(self, train, batch_size, donnotcenter={},
                 pathdata=".",
                 filename="data.tfrecord",
                 sizes={"input":1, "output":1},
                 num_thread=4,
                 ms=None, sds=None,
                 fun_preprocess=None,
                 add_noise={},
                 dtypes={}):
        """
        ms (and sds) should be None or dictionnaries with at least the keys in vars, and tensorflow float32 tensors as values
        
        :param train: if true concern the training set
        :param batch_size: number of data to unpack each time
        :param donnotcenter: iterable: variable that will not be centered / reduced
        :param pathdata: path where the data are stored
        :param filenames: names of the files with input data and output data
        :param sizes: number of columns of the data in X and Y
        :param num_thread: number of thread to read the data
        :param ms: means of data set (used for validation instead -- of recomputing the mean). 
        :param sds: standard deviation of data set (used for validation instead -- of recomputing the mean)
        :param fun_preprocess: fun use to preprocess data (before centering / reducing), (dictionnary with variable name as key)
        :param add_noise: iterable: in which data do you add noise 
        """
        if type(filename) == type(""):
            filename = {filename}
        ExpDataReader.__init__(self, train=train, batch_size=batch_size)
        self.sizes = sizes
        self.num_thread = num_thread
        self.batch_size = batch_size
        self.dtypes = dtypes
        # pdb.set_trace()
        self.features = {k: tf.FixedLenFeature((val,), tf.float32 if k not in dtypes else dtypes[k]
                                               # ,default_value=[0.0 for _ in range(val)]
                                               )
                    for k, val in sizes.items()}
        self.funs_preprocess = {k: tf.identity for k in sizes.keys()}
        if fun_preprocess is not None:
            for k, fun in fun_preprocess.items():
                self.funs_preprocess[k] = fun
        self.donnotcenter = donnotcenter
        # TODO optimization: do not parse the file if you know nrows (training set parsed 2 times)

        # add noise when training, with customizable variance
        if len(add_noise) and train:
            self.sigma_noise = tf.get_variable(name="noise_std", trainable=False)
            self.amount_noise_ph = tf.placeholder(dtype=DTYPE_USED, shape=(), name="skip_conn")
            self.assign_noise = tf.assign(self.sigma_noise, self.amount_noise_ph, name="assign_noise_std")
        else:
            self.amount_noise_ph = tf.placeholder(dtype=DTYPE_USED, shape=(), name="skip_conn")
            self.assign_noise = tf.no_op(name="donothing_noise_std")

        # count the number of lines
        if (ms is None) or (sds is None):
            fun_process = self._normalize
        else:
            fun_process = self._countlines

        ms__, sds__, self.nrows = fun_process(
            path=pathdata, fn=filename, sizes=sizes)
        ms_ = {}
        sds_ = {}
        for k in ms__.keys():
            if k in donnotcenter:
                ms_[k] = np.zeros(ms__[k].shape, dtype=DTYPE_NPY)
                sds_[k] = np.ones(sds__[k].shape, dtype=DTYPE_NPY)
            else:
                ms_[k] = ms__[k]
                sds_[k] = sds__[k]

        # if "cali_tempo.tfrecord" in filename :# or 'val.tfrecord' in filename:
        #     self._normalize(path=pathdata, fn=filename, sizes=sizes)
        #     pdb.set_trace()

        self.ms = self._shape_properly(ms_, name="means") if ms is None else ms
        self.sds = self._shape_properly(sds_, name="stds") if sds is None else sds

        self.dataset = tf.data.TFRecordDataset(
            [os.path.join(pathdata, fn) for fn in filename]).map(
            lambda line: self._parse_function(example_proto=line, sizes=sizes, ms=self.ms, stds=self.sds),
            num_parallel_calls=num_thread,
            # output_buffer_size=num_thread * 5
        ).prefetch(num_thread * 5)
        # self.dataset = self.dataset.shard(10, 2)
        if train:
            self.dataset = self.dataset.repeat(-1)
            self.dataset = self.dataset.shuffle(buffer_size=10000)
        else:
            self.dataset = self.dataset.repeat(1)
        self.dataset = self.dataset.batch(batch_size=batch_size)

    def _countlines(self, path, fn, sizes):
        """
        :param path: the path where data are located
        :param fn: the file names
        :return: the number of lines of the files (must iterate through it line by line) 
        """
        ms = {el: np.zeros(1, dtype=DTYPE_NPY) for el in sizes}
        sds = {el: np.ones(1, dtype=DTYPE_NPY) for el in sizes}
        nb_total = 0
        for fn_ in [os.path.join(path, el) for el in fn]:
            nb = 0
            for nb, record in enumerate(tf.python_io.tf_record_iterator(fn_)):
                pass
            if nb != 0:
                # if the file is empty, we don't have to add one line...
                nb_total += nb+1
                # don't forget to add 1 because python start at 0!
        return ms, sds, nb_total

    def _parse_function(self, example_proto, sizes, ms, stds):
        """
        :param example_proto: 
        :param sizes: 
        :param ms: 
        :param stds: 
        :return: 
        """
        parsed_features = tf.parse_single_example(example_proto, self.features)
        # TODO faster if I batch first! (use tf.pase_example instead)
        for k in sizes.keys():
            parsed_features[k] = self.funs_preprocess[k](parsed_features[k])
            parsed_features[k] = tf.cast(parsed_features[k], dtype=DTYPE_USED)
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

        acc = { k:np.zeros(shape=(v), dtype=np.float64) for k,v in sizes.items() }
        acc2 = { k:np.zeros(shape=(v), dtype=np.float64) for k,v in sizes.items() }
        msg_displayed = {k: 0 for k, _ in sizes.items()}
        with tf.variable_scope("datareader_compute_means_vars"):
            ms = {k: tf.constant(0.0, name="fake_means", dtype=DTYPE_USED) for k,_ in sizes.items()}
            sds = {k: tf.constant(1.0, name="fake_stds", dtype=DTYPE_USED) for k,_ in sizes.items()}
            dataset = tf.data.TFRecordDataset(
                [os.path.join(path, el) for el in fn]).map(
                lambda line: self._parse_function(example_proto=line, sizes=sizes, ms=ms, stds=sds),
                num_parallel_calls=self.num_thread
            ).prefetch(self.num_thread * 5).batch(self.num_thread).repeat(1)
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
                        # print(np.sum(np.abs(pf["conso_X"])))
                        # pdb.set_trace()
                        for k in sizes.keys():
                            vect = pf[k].astype(np.float64)
                            if np.any(~np.isfinite(vect)):
                                if msg_displayed[k] == 0:
                                    msg = "W Datareader : there are infinite or nan values in the dataset named {}"
                                    msg += ". We replaced it with the current average (by column)"
                                    print(msg.format(k))
                                msg_displayed[k] += 1
                                # pdb.set_trace()
                                # import numpy.ma as ma
                                vect = np.where(np.isfinite(vect), vect, acc[k]/(count*self.num_thread))
                                # pdb.set_trace()
                                # vect[~np.isfinite(vect)] = acc[k]/(count*self.num_thread)
                            acc[k] += np.nansum(vect, axis=0)
                            acc2[k] += np.nansum(vect*vect, axis=0)
                            # if np.any(~np.isfinite(acc2[k])):
                            #     pdb.set_trace()
                    except tf.errors.OutOfRangeError:
                        break
            for k in sizes.keys():
                if msg_displayed[k] != 0:
                    msg = "W Datareader : there are at least {} lines where infinite or nan values "
                    msg += " were encounteredin the dataset named {}"
                    msg += ". They were replaced with the current average (by column) at the time of computation"
                    print(msg.format(msg_displayed[k], k))
            acc = {k: v/(count*self.num_thread) for k,v in acc.items()}
            acc2 = {k: v/(count*self.num_thread) for k,v in acc2.items()}

            ms = acc
            # pdb.set_trace()
            stds = {k: np.sqrt(acc2[k] - v * v) for k,v in acc.items()}
            for k,v in stds.items():
                stds[k][stds[k] <= 1e-3] = 1.0

            ms = {k: v.astype(DTYPE_NPY) for k, v in ms.items()}
            stds = {k: v.astype(DTYPE_NPY) for k, v in stds.items()}
        return ms, stds, count

    def _nrows(self, array):
        """
        :param array: unused
        :return: the number of rows in the training set
        """
        return self.nrows


class ExpData:
    def __init__(self, batch_size=50, sizemax=int(1e4),
                 pathdata=".", path_exp=".",
                 classData=ExpDataReader,
                 sizes={"input":1, "output":1},
                 argsTdata=(), kwargsTdata={},
                 argsVdata=(), kwargsVdata={},
                    otherdsinfo = {},
                 donnotcenter={},
                 fun_preprocess=None,
                 dtypes={}
                 ):
        """ The base class for every 'data' subclasses, depending on the problem
        :param batch_size: the size of the minibatch
        :param pathdata: the path where data are stored
        :param path_exp: the path where the experiment is saved
        :param sizemax: maximum size of data chunk that will be "fed" to the computation graph
        :param classData: the class of data (should derive from 'ExpDataReader')
        :param sizes: the sizes (number of columns) of each dataset
        :param argsTdata: default arguments to build an instance of class 'expDataReader' (build the training data set)
        :param kwargsTdata: keywords arguments to build an instance of class 'expDataReader' (build the training data set)
        :param argsVdata: default arguments to build an instance of class 'expDataReader' (build the validation data set)
        :param kwargsVdata: keywords arguments to build an instance of class 'expDataReader' (build the validation data set)
        :param otherdsinfo : dictionnaries of keys = dataset name, values = dictionnaries of keys: "argsdata" : tuple, kwargsdata: dictionnaries
        :param donnotcenter: data that won't be centered/reduced
        :param fun_preprocess: fun use to preprocess data (before centering / reducing), not apply for variable in donnotcenter (pairs: fun to preprocess, fun to "un preprocess")
        """

        # subdirectory name of the experiment where means and std will be stored
        self.means_vars_directory = "means_vars"
        self.path_exp = path_exp
        self.sizes = sizes
        self.donnotcenter = donnotcenter
        ms, sds = self._load_npy_means_stds(classData)
        self.classData = classData

        self.funs_preprocess = {varname: (tf.identity, lambda x :x ) for varname in self.sizes.keys()}
        if fun_preprocess is not None:
            for varname, fun in fun_preprocess.items():
                self.funs_preprocess[varname]= fun
        fun_preprocess = {k: v[0] for k, v in self.funs_preprocess.items()}
        # pdb.set_trace()
        # the data for training (fitting the models parameters)
        self.trainingData = classData(*argsTdata,
                                      donnotcenter=donnotcenter,
                                      pathdata=pathdata,
                                      sizes=sizes,
                                      train=True,
                                      batch_size=batch_size,
                                      ms=ms,
                                      sds=sds,
                                      fun_preprocess=fun_preprocess,
                                      dtypes=dtypes,
                                      **kwargsTdata)
        self.sizes = sizes
        # get the values of means and standard deviation of the training set,
        # to be use in the others sets
        self.ms = self.trainingData.ms
        self.sds = self.trainingData.sds

        # the data for training (only used when reporting error on the whole
        # set)
        self.trainData = classData(*argsTdata,
                                   donnotcenter=donnotcenter,
                                   pathdata=pathdata,
                                   sizes=sizes,
                                   train=False,
                                   batch_size=sizemax,
                                   ms=self.ms,
                                   sds=self.sds,
                                   fun_preprocess=fun_preprocess,
                                      dtypes=dtypes,
                                   **kwargsTdata)
        # the data for validation set (fitting the models hyper parameters --
        # only used when reporting error on the whole set)
        self.valData = classData(*argsVdata,
                                 donnotcenter=donnotcenter,
                                 pathdata=pathdata,
                                 sizes=sizes,
                                 train=False,
                                 batch_size=sizemax,
                                 ms=self.ms,
                                 sds=self.sds,
                                 fun_preprocess=fun_preprocess,
                                      dtypes=dtypes,
                                 **kwargsVdata)
        self.sizemax = sizemax # size maximum of a "minibatch" eg the maximum number of examples that will be fed
        # at once for making a single forward computation

        self.iterator = tf.data.Iterator.from_structure(
            output_types=self.trainingData.dataset.output_types,
            output_shapes=self.trainingData.dataset.output_shapes)

        self.true_data = self.iterator.get_next(
            name="true_data")
        # pdb.set_trace()
        self.training_init_op = self.iterator.make_initializer(
            self.trainingData.dataset)
        self.train_init_op = self.iterator.make_initializer(
            self.trainData.dataset)
        self.validation_init_op = self.iterator.make_initializer(
            self.valData.dataset)

        self.otherdatasets = {}
        self.otheriterator_init = {}
        for otherdsname, values in otherdsinfo.items():
            self.otherdatasets[otherdsname] = classData(*values["argsdata"],
                                                        pathdata=pathdata,
                                                        sizes=sizes,
                                                        train=False,
                                                        batch_size=sizemax,
                                                        ms=self.ms,
                                                        sds=self.sds,
                                                        fun_preprocess=fun_preprocess,
                                                        dtypes=dtypes,
                                                        **values["kwargsdata"]
                                                        )
            self.otheriterator_init[otherdsname] = self.iterator.make_initializer(self.otherdatasets[otherdsname].dataset)

    def activate_val_set(self):
        dataset = self.valData
        initop = self.validation_init_op
        return dataset, initop

    def activate_trainining_set(self):
        dataset = self.trainingData
        initop = self.training_init_op
        return dataset, initop

    def activate_dataset(self, dataset_name):
        dataset = self.otherdatasets[dataset_name]
        initop = self.otheriterator_init[dataset_name]
        return dataset, initop

    def activate_trainining_set_sameorder(self):
        dataset = self.trainData
        initop = self.train_init_op
        return dataset, initop

    def _load_npy_means_stds(self, classData):
        """
        If the means and variance have already been computed, it will load them from the hard drive
        :param classData: the data class used
        :return: ms, sds with
        ms = None if data does not exists, otherwise the dictionnary of means for each variable in "sizes"
        """

        if not os.path.exists(os.path.join(self.path_exp, self.means_vars_directory)):
            return None, None
        else:
            isOk = True
            for k in self.sizes.keys():
                mE = os.path.exists(os.path.join(self.path_exp, self.means_vars_directory, "ms-{}.npy".format(k)))
                sE = os.path.exists(os.path.join(self.path_exp, self.means_vars_directory, "sds-{}.npy".format(k)))
                if not mE or not sE:
                    isOk = False
                    break
            if not isOk:
                return None, None
            else:
                ms = {}
                sds = {}
                for k in self.sizes.keys():
                    m = np.load(os.path.join(self.path_exp, self.means_vars_directory, "ms-{}.npy".format(k)))
                    s = np.load(os.path.join(self.path_exp, self.means_vars_directory, "sds-{}.npy".format(k)))
                    if k in self.donnotcenter:
                        m = np.zeros(m.shape, dtype=DTYPE_NPY)
                        s = np.ones(s.shape, dtype=DTYPE_NPY)
                    ms[k] = m
                    sds[k] = s
                # pdb.set_trace()
                if classData.ms_tensor:
                    ms = self._shape_properly(ms, name="means")
                    sds = self._shape_properly(sds, name="stds")

                return ms, sds

    def _shape_properly(self, ms, name):
        """
        TODO copy paste from TFDataReader
        Transform a dictionnary of numpy array in a dictionnary of tensorflow tensor
        :param name
        :param ms: 
        :return: 
        """
        return {k: tf.convert_to_tensor(v, name="{}_{}".format(name, k), dtype=DTYPE_USED) for k, v in ms.items()}

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
        acc_loss = 0.
        error_nan = False
        while True:
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
            dtype=DTYPE_USED)
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
        if self.classData.ms_tensor:
            # the data class represents the means and standard deviation as tensors
            self.sds = sess.run(self.sds)
            self.ms = sess.run(self.ms)
        if not os.path.exists(os.path.join(self.path_exp, self.means_vars_directory)):
            os.mkdir(os.path.join(self.path_exp, self.means_vars_directory))
        # pdb.set_trace()
        for k in self.sizes.keys():
            np.save(file=os.path.join(self.path_exp, self.means_vars_directory, "ms-{}.npy".format(k)),
                    arr=self.ms[k])
            np.save(file=os.path.join(self.path_exp, self.means_vars_directory, "sds-{}.npy".format(k)),
                    arr=self.sds[k])


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


class ExpInMemoryDataReader(ExpDataReader):
    ms_tensor = False # is the "self.ms" (or "self.sds") a tensor (True) or a numpy array (False)
    def __init__(self, train, batch_size, pathdata=".",
                 filename={"input": "X.npy" , "output": ("Y.npy",)},
                 sizes=None,
                 num_thread=4, donnotcenter={},
                 fun_preprocess=lambda x: x,
                 dtypes={},
                 ms=None, sds=None,
                 numpy_=True,
                 panda_=False,
                 sep=";"):
        """
        Load data in memory, and then use an iterator to process it.
        Data can be read from numpy array (numpy_ = True) or from pandas data frame (numpy_ = False, panda_=True).
        In the later case, dataframe are converted to numpy array
        :param train: 
        :param batch_size: 
        :param pathdata: 
        :param filenames: 
        :param sizes: 
        :param num_thread: 
        :param donnotcenter: 
        :param ms: 
        :param sds: 
        """
        self.train = train

        if numpy_:
            mmap_mode = None if train else "c"  # "c" stand for: data are kept on the hard drive, but can be modified in memory
            self.datasets = {k: np.load(os.path.join(pathdata, v), mmap_mode=mmap_mode) for k, v in filename.items()}
        elif panda_:
            import pandas as pd
            self.datasets = {k: pd.read_csv(os.path.join(pathdata, v), sep=sep).values for k, v in filename.items()}
        else:
            raise RuntimeError("For now only panda dataframe and numpy array are supported in ")

        self.datasets = {k: v.astype(DTYPE_NPY) for k, v in self.datasets.items() }
        sizes = {k: el.shape[1] for k, el in self.datasets.items()}
        self.sizes = sizes
        # pdb.set_trace()
        if ms is None:
            ms_ = {k: np.mean(v, axis=0) for k,v in self.datasets.items()}
            self.ms = {k: v for k,v in ms_.items() if not k in donnotcenter}
            for el in donnotcenter:
                self.ms[el] = np.zeros(ms_[el].shape, dtype=DTYPE_NPY)
        else:
            self.ms = ms
        if sds is None:
            sds_ = {k: np.std(v, axis=0) for k,v in self.datasets.items()}
            self.sds = {k: v for k,v in sds_.items() if not k in donnotcenter}
            for el in donnotcenter:
                self.sds[el] = np.ones(sds_[el].shape, dtype=DTYPE_NPY)
        else:
            self.sds = sds

        self.datasets = {k: (v-self.ms[k])/self.sds[k] for k, v in self.datasets.items()}
        # self.placeholders = {k: tf.placeholder(shape=(None, v), dtype=tf.float32) for k,v in sizes.items()}

        self.batch_size = batch_size
        self.nrows = self.datasets[next(iter(filename.keys()))].shape[0]
        self.indexDataMinibatch = list(range(self.nrows))
        if self.train:
            random.shuffle(self.indexDataMinibatch)
        self.lastIndexDataMinibatch = 0
        self.train = train
        self.features = {k: tf.FixedLenFeature(val, tf.float32)#, default_value=[0.0 for _ in range(val)])
                            for k, val in sizes.items()}

        self.dataset = tf.data.Dataset.from_generator(generator=self.generator,
                                                      output_types={k: v.dtype for k, v in self.features.items()},
                                                      output_shapes={k: v.shape for k, v in self.features.items()})
        # pdb.set_trace()
        if train:
            self.dataset = self.dataset.repeat(-1)
            # self.dataset = self.dataset.shuffle(buffer_size=10000)
        else:
            self.dataset = self.dataset.repeat(1)
        self.dataset = self.dataset.batch(batch_size=batch_size)
        # pdb.set_trace()

    def generator(self):
        """
        :return: the data of one line of the dataset
        """
        new_epoch = False
        while not new_epoch:
            new_epoch, indx = self.getnextindexes()
            yield {k: v[indx, :].flatten() for k, v in self.datasets.items()}
        raise StopIteration

    def getnextindexes(self):
        """
        :return: the next index to be considered 
        """
        size = 1
        new_epoch = False
        if self.lastIndexDataMinibatch+size <= self.nrows:
            prev = self.lastIndexDataMinibatch
            self.lastIndexDataMinibatch += size
            res = self.indexDataMinibatch[prev:self.lastIndexDataMinibatch]
        else:
            new_epoch = True
            prev = self.lastIndexDataMinibatch
            res = self.indexDataMinibatch[prev:]
            if self.train:
                random.shuffle(self.indexDataMinibatch)
            self.lastIndexDataMinibatch = size-len(res)
            res += self.indexDataMinibatch[:self.lastIndexDataMinibatch]
        return new_epoch, res

    def init(self, sess):
        """
        :param sess: 
        :return: 
        """
        # tf.train.start_queue_runners(sess=sess)
        pass

    def nrowsX(self):
        """
        :return: The number of rows / examples of the input data set
        """
        return self.nrows


class ExpNpyDataReader(ExpInMemoryDataReader):
    def __init__(self, args, kwargs):
        ExpInMemoryDataReader.__init__(self, *args, **kwargs)

if __name__=="__main__":
    test_tf_nb_lines = ExpTFrecordsDataReader(train=True, batch_size=1)
    print("The number of lines computed is {}".format(test_tf_nb_lines._countlines(path="/home/bdonnot/Documents/PyHades2/tfrecords_118_5000",
                                                                                   fn="neighbours-test.tfrecord",
                                                                                   sizes={"prod_q": 54})))
    print("Test lines parsing {}".format(test_tf_nb_lines._normalize(path="/home/bdonnot/Documents/PyHades2/tfrecords_118_5000",
                                                                     fn="neighbours-test.tfrecord",
                                                                     sizes={"prod_q":54})))
