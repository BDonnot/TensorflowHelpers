from TensorflowHelpers.Experiment import ExpData, Exp, ExpSaverParam, ExpGraph
from TensorflowHelpers.DefDataHandler import ExpCSVDataReader, ExpTFrecordsDataReader

if __name__ == "__main__":
    #TODO set the seeds, generate random data and industrialize this test

    testcsv = False

    if testcsv:
        #Test the launching of a classical experiments with csv
        pathdata = "/home/bdonnot/Documents/RTEConsoChallenge/data/newData/valData/forTF"
        path_exp = "/home/bdonnot/Documents/PyHades2/Test"

        # pathdata = "D:\\Documents\\Conso_RTE\\data\\forTF"
        # path_exp = "D:\\Documents\\PyHades2\\Test"

        # define the experiment parameters
        parameters = ExpSaverParam(name_exp="firstTestTFrecords", path=path_exp,
                                   pathdata=pathdata)

        kwargsTdata = {"filenames": {"input": "conso_X.csv", "ourput": "conso_Y.csv"},
                       "sizes": {"input":1690, "output": 845}, "num_thread": 2}
        datakwargs = {"classData": ExpCSVDataReader,
                            "kwargsTdata": kwargsTdata,
                            "kwargsVdata": kwargsTdata
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
        var_x_name = "prod_q"
        var_y_name = "prod_p"
        kwargsTdata = {"filename": "neighbours-test.tfrecord", "sizes": {var_x_name:54, var_y_name:54}, "num_thread": 2}
        datakwargs = {"classData": ExpTFrecordsDataReader,
                            "kwargsTdata": kwargsTdata,
                            "kwargsVdata": kwargsTdata
        }


    my_exp = Exp(parameters=parameters,
                 dataClass=ExpData, datakwargs=datakwargs,
                 graphType=ExpGraph, graphkwargs={"kwargsNN": {"layersizes": [50, 50],
                                                               "weightnorm": True},
                                                          "var_x_name": var_x_name,
                                                          "var_y_name": var_y_name
                                                          }
                 )
    my_exp.start()
