from .ANN import DenseLayer, NNFully, ResidualBlock, DenseBlock, DTYPE_USED

from .DataHandler import ExpDataReader, ExpCSVDataReader, ExpData, ExpTFrecordsDataReader, ExpNpyDataReader, ExpInMemoryDataReader  # reading and handling data

from .Graph import ExpGraphOneXOneY, ExpGraph, ComplexGraph  # the tensorflow "main" comptuation graph

from .Experiment import Path
from .Experiment import TFWriters, ExpLogger  # logging and saving data
from .Experiment import ExpParam, ExpSaverParam  # parameters for an experiment
from .Experiment import ExpModel  # model you want to use
from .Experiment import Exp #, ExpManager # the experience

from .Losses import l2, rmse, pinball, pinball_multi_q, sigmoid_cross_entropy_with_logits, softmax_cross_entropy

from .Emulator import EmulLog

def isok_arg(arg, argname=""):
    """
    Test if argument is "True" or "False", with improve compatibility for "t"/"f", "T"/"F", "1"/"0"
    :param arg:
    :param argname:
    :return:
    """
    if isinstance(arg, type(True)):
        return arg
    if isinstance(arg, type("")):
        arg = arg.strip("\"")

    res = False
    if arg == "True" or arg == "T" or arg == "true" or arg == "t" or str(arg) == "1":
        res = True
    elif arg == "False" or arg == "F" or arg == "false" or arg == "f" or str(arg) == "0":
        res = False
    else:
        msg = "You enter a argument {} which should be a boolean."
        msg += " Please check argument named \"{}\""
        msg += " and change its value to \"True\" or \"False\""
        raise RuntimeError(msg.format(arg, argname))
    return res

