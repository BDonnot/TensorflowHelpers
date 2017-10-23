import sys

from TensorflowHelpers.Experiment import ExpData, Exp, ExpSaverParam
from TensorflowHelpers.DefGraph import ExpGraphOneXOneY, ExpGraph, ComplexGraph
from TensorflowHelpers.DefDataHandler import ExpCSVDataReader, ExpTFrecordsDataReader
from TensorflowHelpers.DefANN import ResidualBlock, DenseBlock

if __name__ == "__main__":
    #TODO set the seeds, generate random data and industrialize this test

    # test multiple input and multiple output
    # for now only tfrecords
    pathdata = "/save/prev_conso_locale"
    path_exp = "/home/bdonnot/Documents/PyHades2/Test"

    # define the experiment parameters
    parameters = ExpSaverParam(name_exp="firstTestConso_small",
                               path=path_exp,
                               pathdata=pathdata,
                               num_epoch=10,
                               num_savings=1,
                               num_savings_minibatch=5,
                               num_savings_model=1,
                               continue_if_exists=True)
    var_x_name = {"x_veille", "x_sem", "meteo", "x_cat"}
    var_y_name = {"y"}
    sizes = {"x_veille": 24, "x_sem": 24, "meteo": 24, "x_cat": 15, "y": 24}
    sizes_out = {"x_veille": 300, "x_sem": 300, "meteo": 300, "x_cat": 300}
    kwargsTdata = {"filename": "conso_locale_train_small.tfrecord",
                   "num_thread": 2}
    kwargsVdata = {"filename": "conso_locale_val_small.tfrecord",
                   "num_thread": 2}
    datakwargs = {"classData": ExpTFrecordsDataReader,
                  "kwargsTdata": kwargsTdata,
                  "kwargsVdata": kwargsVdata,
                  "sizes": sizes
                  }
    my_exp = Exp(parameters=parameters,
                 dataClass=ExpData, datakwargs=datakwargs,
                 graphType=ComplexGraph,
                 graphkwargs={"kwargsNN": {"layersizes": [100, 100], "weightnorm": True, "layerClass": ResidualBlock},
                              "var_x_name": var_x_name,
                              "var_y_name": var_y_name,
                              "sizes": sizes_out,
                              "outputsize": 150,
                              "kwargs_enc_dec": {"layerClass": DenseBlock, "kwardslayer": {"nblayer":5}}
                              },
                 otherdsinfo={"test": {"argsdata": (), "kwargsdata": {"filename": "conso_locale_test_small.tfrecord"}} },
                 startfromscratch=True,
                 modelkwargs={"optimizerkwargs": {"lr": 1e-4}}
                 )
    my_exp.start()
    sys.exit("done for now")

    testmultipledata = True
    testcsv = False
    if not testmultipledata:
        # test single data input and single data output
        # for now simply csv and tfrecords
        if testcsv:
            #Test the launching of a classical experiments with csv
            pathdata = "/home/bdonnot/Documents/RTEConsoChallenge/data/newData/valData/forTF"
            path_exp = "/home/bdonnot/Documents/PyHades2/Test"

            # pathdata = "D:\\Documents\\Conso_RTE\\data\\forTF"
            # path_exp = "D:\\Documents\\PyHades2\\Test"

            # define the experiment parameters
            parameters = ExpSaverParam(name_exp="firstTestTFrecords", path=path_exp,
                                       pathdata=pathdata)

            kwargsTdata = {"filenames": {"input": "conso_X.csv", "ourput": "conso_Y.csv"}, "num_thread": 2}
            datakwargs = {"classData": ExpCSVDataReader,
                                "kwargsTdata": kwargsTdata,
                                "kwargsVdata": kwargsTdata,
                           "sizes": {"input":1690, "output": 845}
            }
            # optional for standard graph
            var_x_name = "input"
            var_y_name = "output"
        else:
            pathdata = "/home/bdonnot/Documents/PyHades2/tfrecords_118_5000"
            path_exp = "/home/bdonnot/Documents/PyHades2/Test"

            # define the experiment parameters
            parameters = ExpSaverParam(name_exp="firstTestTFrecords",
                                       path=path_exp,
                                       pathdata=pathdata,
                                       num_epoch=1,
                                       num_savings=1,
                                       num_savings_minibatch=5,
                                       num_savings_model=1)
            var_x_name = "prod_p"
            var_y_name = "prod_q"
            kwargsTdata = {"filename": "neighbours-test.tfrecord", "num_thread": 2}
            datakwargs = {"classData": ExpTFrecordsDataReader,
                                "kwargsTdata": kwargsTdata,
                                "kwargsVdata": kwargsTdata,
                          "sizes": {var_x_name: 54, var_y_name: 54}
                          }


        my_exp = Exp(parameters=parameters,
                     dataClass=ExpData, datakwargs=datakwargs,
                     graphType=ExpGraphOneXOneY, graphkwargs={"kwargsNN": {"layersizes": [50, 50],
                                                                   "weightnorm": True},
                                                              "var_x_name": var_x_name,
                                                              "var_y_name": var_y_name
                                                              }
                     )
        my_exp.start()
    else:
        # test multiple input and multiple output
        # for now only tfrecords
        pathdata = "/home/bdonnot/Documents/PyHades2/tfrecords_118_5000"
        path_exp = "/home/bdonnot/Documents/PyHades2/Test"

        # define the experiment parameters
        parameters = ExpSaverParam(name_exp="firstTestTFrecords",
                                   path=path_exp,
                                   pathdata=pathdata,
                                   num_epoch=1,
                                   num_savings=1,
                                   num_savings_minibatch=5,
                                   num_savings_model=1)
        var_x_name = {"prod_p", "prod_v", "loads_p"}
        var_y_name = {"prod_q", "loads_v"}
        sizes = {"prod_p": 54, "prod_q": 54, "loads_p":99, "prod_v":54, "loads_v": 99}
        kwargsTdata = {"filename": "N1-train.tfrecord",
                       "num_thread": 2}
        kwargsVdata = {"filename": "N1-val.tfrecord",
                       "num_thread": 2}
        datakwargs = {"classData": ExpTFrecordsDataReader,
                      "kwargsTdata": kwargsTdata,
                      "kwargsVdata": kwargsVdata,
                      "sizes": sizes
        }
        my_exp = Exp(parameters=parameters,
                     dataClass=ExpData, datakwargs=datakwargs,
                     graphType=ExpGraph, graphkwargs={"kwargsNN": {"layersizes": [100, 100], "weightnorm": True},
                                                              "var_x_name": var_x_name,
                                                              "var_y_name": var_y_name
                                                              },
                     otherdsinfo={ # "Test": {"argsdata": (), "kwargsdata": {"filename": "N1-test.tfrecord"}}, # dataset corrupted
                                  "n2_neighbours": {"argsdata": (),
                                                    "kwargsdata": {"filename": "neighbours-train.tfrecord"}},
                                  "n2_random": {"argsdata": (),
                                                 "kwargsdata": {"filename": "random-train.tfrecord"}},
                                  },
                     startfromscratch=True
                     )
        my_exp.start()
