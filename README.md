# TensorflowHelpers

This package provide some utility function to help training model using the tensorflow framework.
It helps in many ways:
- easy to compute errors on the complete training set, or validation set, or test set etc.
- easy logging of data, using tensorboard and a text logger (quicker acces)
- the computation graph will be formatted in different part:
	- data: where data are read and parsed, possibly pre-processed
	- graph: the most important part of the computation graph, where the predictions are done
	- training_loss: different loss that will be optimized during training
	- optimizer: the optimizer used
	- summaries: regroup all the summaries that will be exported in tensorboard

It also offers the possibility to quickly setup an experiment (training of a neural network) without having to bother with the definitions of annex stuff, such as reporting training loss etc.
