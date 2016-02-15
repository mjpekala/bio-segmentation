"""  Trains a CNN for dense (i.e. per-pixel) classification problems.

"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2016, JHU/APL"
__license__ = "Apache 2.0"


import sys, os.path
import argparse
import logging

from keras.optimizers import SGD

import emlib
import models as emm



def _train_mode_args():
    """Parameters for training a CNN.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x-train', dest='emTrainFile', 
		    type=str, required=True,
		    help='Filename of the training volume (train mode)')
    parser.add_argument('--y-train', dest='labelsTrainFile', 
		    type=str, required=True,
		    help='Filename of the training labels (train mode)')
    parser.add_argument('--train-slices', dest='trainSlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y train')

    parser.add_argument('--x-valid', dest='emValidFile', 
		    type=str, required=True,
		    help='Filename of the validation volume (train mode)')
    parser.add_argument('--y-valid', dest='labelsValidFile', 
		    type=str, required=True,
		    help='Filename of the validation labels (train mode)')
    parser.add_argument('--valid-slices', dest='validSlices', 
		    type=str, default='', 
		    help='(optional) limit to a subset of X/Y validation')

    parser.add_argument('--omit-labels', dest='omitLabels', 
		    type=str, default='', 
		    help='(optional) list of class labels to omit from training')
    
    parser.add_argument('--n-epochs', dest='nEpochs', 
		    type=int, default=30,
		    help='number of training epochs')

    parser.add_argument('--gpu', dest='gpu', 
		    type=int, default=-1, 
		    help='GPU ID to use')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='', 
		    help='directory where the trained file should be placed')

    args = parser.parse_args()

    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)

    # map strings into python objects.  A little gross to use eval, but life is short.
    str_to_obj = lambda x: eval(x) if x else []
    
    args.trainSlices = str_to_obj(args.trainSlices)
    args.validSlices = str_to_obj(args.validSlices)
    args.omitLabels = str_to_obj(args.omitLabels)
    
    return args



def _train_one_epoch(model, Xtrain, Ytrain):
    pass
    


if __name__ == "__main__":
    args = _train_mode_args()
    
    # whoever designed the logging module must have been a java programmer...
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)

    # load training and validation volumes
    Xtrain = emlib.load_cube(args.emTrainFile)
    Ytrain = emlib.load_cube(args.labelsTrainFile)
    if args.trainSlices:
        Xtrain = Xtrain[args.trainSlices,:,:]
        Ytrain = Ytrain[args.trainSlices,:,:]
        
    Xvalid = emlib.load_cube(args.emValidFile)
    Yvalid = emlib.load_cube(args.labelsValidFile)
    if args.validSlices:
        Xvalid = Xvalid[args.validSlices,:,:]
        Yvalid = Yvalid[args.validSlices,:,:]

    logger.info('training volume dimensions:   %s' % str(Xtrain.shape))
    logger.info('validation volume dimensions: %s' % str(Xvalid.shape))

    # create and configure CNN
    # XXX: make configurable via command line?
    model = emm.ciresan_n3()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    for epoch in range(args.nEpochs):
        logger.info('starting training epoch %d' % epoch);
    
        #model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
