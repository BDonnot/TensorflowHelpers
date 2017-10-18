from TensorflowHelpers.Experiment import ExpData, Exp, ExpSaverParam, ExpCSVDataReader

if __name__ == "__main__":
    #TODO set the seeds, generate random data and industrialize this test

    #Test the launching of a classical experiments
    pathdata = "/home/bdonnot/Documents/RTEConsoChallenge/data/newData/valData/forTF"
    path_exp = "/home/bdonnot/Documents/PyHades2/Test"

    # pathdata = "D:\\Documents\\Conso_RTE\\data\\forTF"
    # path_exp = "D:\\Documents\\PyHades2\\Test"

    # define the experiment parameters
    parameters = ExpSaverParam(name_exp="firstTest", path=path_exp,
                               pathdata=pathdata)
    kwargsTdata = {"filenames": ["conso_X.csv", "conso_Y.csv"], "sizes": [1690, 845], "num_thread": 2}
    datakwargs = {"classData": ExpCSVDataReader,
                        "kwargsTdata": kwargsTdata,
                        "kwargsVdata": kwargsTdata
    }
    my_exp = Exp(parameters=parameters,
                 dataClass=ExpData, datakwargs=datakwargs,
                 graphkwargs={"kwargsNN": {"layersizes":[50,50], "weightnorm": True}}
                 )
    my_exp.start()

