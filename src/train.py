"""  Trains a CNN for dense (i.e. per-pixel) classification problems.

Note: currently assumes input image data is single channel (i.e. grayscale)

"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2016, JHU/APL"
__license__ = "Apache 2.0"


import sys, os.path
import time
import random
import argparse
import logging
import numpy as np
import pdb

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



def _xform_minibatch(X):
    """Implements synthetic data augmentation by randomly appling
    an element of the group of symmetries of the square toa single mini-batch
    of data.

    The default set of data augmentation operations correspond to
    the symmetries of the square (a non abelian group).  The
    elements of this group are:

      o four rotations (0, pi/2, pi, 3*pi/4)
        Denote these by: R0 R1 R2 R3

      o two mirror images (about y-axis or x-axis)
        Denote these by: M1 M2

      o two diagonal flips (about y=-x or y=x)
        Denote these by: D1 D2

    This page has a nice visual depiction:
      http://www.cs.umb.edu/~eb/d4/


    Parameters: 
       X := Mini-batch data (#examples, #channels, rows, colums) 
    """

    def R0(X):
        return X  # this is the identity map

    def M1(X):
        return X[:,:,::-1,:]

    def M2(X): 
        return X[:,:,:,::-1]

    def D1(X):
        return np.transpose(X, [0, 1, 3, 2])

    def R1(X):
        return D1(M2(X))   # = rot90 on the last two dimensions

    def R2(X):
        return M2(M1(X))

    def R3(X): 
        return D2(M2(X))

    def D2(X):
        return R1(M1(X))


    symmetries = [R0, R1, R2, R3, M1, M2, D1, D2]
    op = random.choice(symmetries) 
        
    # For some reason, the implementation of row and column reversals, 
    #     e.g.      X[:,:,::-1,:]
    # break PyCaffe.  Numpy must be doing something under the hood 
    # (e.g. changing from C order to Fortran order) to implement this 
    # efficiently which is incompatible w/ PyCaffe.  
    # Hence the explicit construction of X2 with order 'C' below.
    X2 = np.zeros(X.shape, dtype=np.float32, order='C') 
    X2[...] = op(X)

    return X2



def _train_one_epoch(logger, model, X, Y, omitLabels, batchSize=100):
    # Pre-allocate some variables & storage.

    # assuming square tiles with odd dimension
    rows, cols = model.input_shape[2:4]
    assert(rows == cols)
    assert(np.mod(rows,2) == 1)
    tileRadius = int(rows/2)

    nChannels = 1  # could relax this assumption later
    Xi = np.zeros((batchSize, nChannels, rows, cols), dtype=np.float32)

    # with the Theano backend, it seems the codes are expecting
    # one hot vectors as class labels.
    yAll = np.unique(Y).astype(np.int32)
    assert(np.min(yAll) == 0);  assert(np.max(yAll) == (len(yAll)-1))
    yi = np.zeros((batchSize, len(yAll)), dtype=np.float32)

    # some variables we'll use for reporting progress    
    lastChatter = -2
    startTime = time.time()
    accBuffer = np.nan*np.ones(10)
    lossBuffer = np.nan*np.ones(accBuffer.shape)

    it = emlib.stratified_interior_pixel_generator(Y,
                                                   tileRadius,
                                                   batchSize,
                                                   omitLabels=omitLabels) 

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        # Map the indices Idx -> tiles Xi and labels yi 
        # 
        # Note: if Idx.shape[0] < batchDim[0] (last iteration of an epoch) 
        # a few examples from the previous minibatch will be "recycled" here. 
        # This is intentional (to keep batch sizes consistent even if data 
        # set size is not a multiple of the minibatch size). 
        # 
        for jj in range(Idx.shape[0]): 
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]
            yj = Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ] 
            yi[jj,:] = 0;  yi[jj,yj] = 1

        # label-preserving data transformation (synthetic data generation)
        Xi = _xform_minibatch(Xi)

        assert(not np.any(np.isnan(Xi)))
        assert(not np.any(np.isnan(yi)))

        # do training
        loss, acc = model.train_on_batch(Xi, yi, accuracy=True)
        accBuffer[np.mod(mbIdx, len(accBuffer))] = acc
        lossBuffer[np.mod(mbIdx, len(lossBuffer))] = loss

        #----------------------------------------
        # Some events occur on regular intervals.
        # Address these here.
        #----------------------------------------
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  # notify progress every 2 min
            lastChatter = elapsed
            logger.info("just completed mini-batch %d" % mbIdx)
            logger.info("we are %0.2f%% complete with this epoch" % (100.*epochPct))
            logger.info("recent accuracy, loss: %0.2f, %0.2f" % (np.mean(accBuffer), np.mean(lossBuffer)))
                


if __name__ == "__main__":
    args = _train_mode_args()
    
    # setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load training and validation volumes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # rescale features to live in [0 1]
    xMin = np.min(Xtrain);  xMax = np.max(Xtrain)
    Xtrain = (Xtrain - xMin) / (xMax - xMin)
    Xvalid = (Xvalid - xMin) / (xMax - xMin)

    logger.info('training volume dimensions:   %s' % str(Xtrain.shape))
    logger.info('training values min/max:      %g, %g' % (np.min(Xtrain), np.max(Xtrain)))
    logger.info('validation volume dimensions: %s' % str(Xvalid.shape))
    logger.info('validation values min/max:    %g, %g' % (np.min(Xvalid), np.max(Xvalid)))

    # remap class labels to consecutive natural numbers
    Ytrain = emlib.number_classes(Ytrain, args.omitLabels)
    Yvalid = emlib.number_classes(Yvalid, args.omitLabels)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create and configure CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: make network configurable via command line
    logger.info('creating CNN')
    model = emm.ciresan_n3()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', class_mode='binary', optimizer=sgd)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do training
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for epoch in range(args.nEpochs):
        logger.info('starting training epoch %d' % epoch);
        _train_one_epoch(logger, model, Xtrain, Ytrain, args.omitLabels)
    
        logger.info('epoch %d complete. validating...' % epoch);
        #model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
        
        logger.info('TODO')


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
