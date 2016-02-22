"""  Trains a CNN for dense (i.e. per-pixel) classification problems.

 Notes: 
   o Currently assumes image data volumes have dimensions:
            #slices x #channels x rows x colums

     while label volumes have dimensions:
            #slices x rows x colums

   o When using caffe, it was straightforward to set the gpu id here.  
     I believe for keras this is controlled by the backend (e.g. theano).
     So, for the time being, you need to set the gpu id to use externally
     from this script.

"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2016, JHU/APL"
__license__ = "Apache 2.0"


import sys, os
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
		    type=str, default='[-1,]', 
		    help='(optional) list of class labels to omit from training')
    
    parser.add_argument('--model', dest='model', 
		    type=str, default='ciresan_n3',
		    help='name of CNN model to use (python function)')
    parser.add_argument('--num-epochs', dest='nEpochs', 
		    type=int, default=30,
		    help='number of training epochs')
    parser.add_argument('--num-mb-per-epoch', dest='nBatches', 
		    type=int, default=sys.maxint,
		    help='maximum number of mini-batches to process each epoch')

    parser.add_argument('--out-dir', dest='outDir', 
		    type=str, default='', 
		    help='directory where the trained file should be placed')

    args = parser.parse_args()

    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)


    # Map strings into python objects.  
    # A little gross to use eval, but life is short.
    str_to_obj = lambda x: eval(x) if x else []
    
    args.trainSlices = str_to_obj(args.trainSlices)
    args.validSlices = str_to_obj(args.validSlices)
    args.omitLabels = str_to_obj(args.omitLabels)
    
    return args



def _xform_minibatch(X):
    """Implements synthetic data augmentation by randomly appling
    an element of the group of symmetries of the square to a single 
    mini-batch of data.

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




def _train_one_epoch(logger, model, X, Y, 
                     omitLabels=[], 
                     batchSize=100,
                     nBatches=sys.maxint):
    """Trains the model for <= one epoch.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    nChannels, nRows, nCols = model.input_shape[1:4]
    assert(nRows == nCols)
    ste = emlib.SimpleTileExtractor(nRows, X, Y, omitLabels=omitLabels)

    # some variables we'll use for reporting progress    
    lastChatter = -2
    startTime = time.time()
    gpuTime = 0
    accBuffer = []
    lossBuffer = []

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = emlib.stratified_interior_pixel_generator(Y, 0, batchSize,
                                                   omitLabels=omitLabels,
                                                   stopAfter=nBatches*batchSize) 

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        Xi, Yi = ste.extract(Idx)

        # label-preserving data transformation (synthetic data generation)
        Xi = _xform_minibatch(Xi)

        assert(not np.any(np.isnan(Xi)))
        assert(not np.any(np.isnan(Yi)))

        # do training
        tic = time.time()
        loss, acc = model.train_on_batch(Xi, Yi, accuracy=True)
        gpuTime += time.time() - tic

        accBuffer.append(acc);  lossBuffer.append(loss)

        #----------------------------------------
        # Some events occur on regular intervals.
        # Address these here.
        #----------------------------------------
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  
            # notify progress every 2 min
            lastChatter = elapsed

            if len(accBuffer) < 10:
                recentAcc = np.mean(accBuffer)
                recentLoss = np.mean(lossBuffer)
            else:
                recentAcc = np.mean(accBuffer[-10:])
                recentLoss = np.mean(lossBuffer[-10:])

            logger.info("  just completed mini-batch %d" % mbIdx)
            logger.info("  we are %0.2g%% complete with this epoch" % (100.*epochPct))
            logger.info("  recent accuracy, loss: %0.2f, %0.2f" % (recentAcc, recentLoss))
            fracGPU = (gpuTime/60.)/elapsed
            logger.info("  pct. time spent on CNN ops.: %0.2f%%" % (100.*fracGPU))
            logger.info("")

    # return statistics
    return accBuffer, lossBuffer



def _evaluate(logger, model, X, Y, omitLabels=[], batchSize=100):
    """Evaluate model on held-out data.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    nChannels, tileRows, tileCols = model.input_shape[1:4]
    tileRadius = int(tileRows/2)
    ste = emlib.SimpleTileExtractor(tileRows, X)

    numClasses = model.output_shape[-1]
    [numZ, numChan, numRows, numCols] = X.shape
    Prob = np.nan * np.ones([numZ, numClasses, numRows, numCols],
                            dtype=np.float32)

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = emlib.interior_pixel_generator(X, tileRadius, batchSize)

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        n = Idx.shape[0]         # may be < batchSize on final iteration
        Xi = ste.extract(Idx)
        prob = model.predict_on_batch(Xi)
        Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[0][:n,:]

    # Evaluate accuracy only on the subset of pixels that:
    #   o were actually provided to the CNN (not downsampled)
    #   o have a label that should be evaluated
    #
    # The mask tensor M will indicate which pixels to consider.
    M = np.all(np.isfinite(Prob), axis=1)
    for om in omitLabels:
        M[Y==om] = False
    Yhat = np.argmax(Prob, axis=1)  # probabilities -> class labels
    acc = 100.0 * np.sum(Yhat[M] == Y[M]) / np.sum(M)

    return Prob, acc



#-------------------------------------------------------------------------------
if __name__ == "__main__":
    args = _train_mode_args()
    
    # setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)

    logger.info(str(args))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load training and validation volumes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Xtrain = emlib.load_cube(args.emTrainFile, addChannel=True)
    Ytrain = emlib.load_cube(args.labelsTrainFile, addChannel=False)
    if args.trainSlices:
        Xtrain = Xtrain[args.trainSlices,:,:,:]
        Ytrain = Ytrain[args.trainSlices,:,:]
        
    Xvalid = emlib.load_cube(args.emValidFile, addChannel=True)
    Yvalid = emlib.load_cube(args.labelsValidFile, addChannel=False)
    if args.validSlices:
        Xvalid = Xvalid[args.validSlices,:,:,:]
        Yvalid = Yvalid[args.validSlices,:,:]

    # rescale features to live in [0 1]
    # XXX: technically, should probably use scale factors from
    #      train volume on validation data...
    Xtrain = emlib.rescale_01(Xtrain, perChannel=True)
    Xvalid = emlib.rescale_01(Xvalid, perChannel=True)

    logger.info('training volume dimensions:   %s' % str(Xtrain.shape))
    logger.info('training values min/max:      %g, %g' % (np.min(Xtrain), np.max(Xtrain)))
    logger.info('training class labels:        %s' % str(np.unique(Ytrain)))
    logger.info('')
    logger.info('validation volume dimensions: %s' % str(Xvalid.shape))
    logger.info('validation values min/max:    %g, %g' % (np.min(Xvalid), np.max(Xvalid)))
    logger.info('validation class labels:      %s' % str(np.unique(Ytrain)))

    # Remap class labels to consecutive natural numbers.
    # Note that any pixels that should be omitted from the 
    # analysis are mapped to -1 by this function.
    Ytrain = emlib.number_classes(Ytrain, args.omitLabels)
    Yvalid = emlib.number_classes(Yvalid, args.omitLabels)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create and configure CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # XXX: make the learning rate parameters configurable.
    logger.info('creating CNN')
    model = getattr(emm, args.model)() 
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', 
            class_mode='categorical', 
            optimizer=sgd)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do training
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for epoch in range(args.nEpochs):
        logger.info('starting training epoch %d' % epoch);
        acc, loss = _train_one_epoch(logger, model, Xtrain, Ytrain, 
                                     omitLabels=[-1,],
                                     nBatches=args.nBatches)

        # save a snapshot of current model weights
        weightFile = os.path.join(args.outDir, "weights_epoch_%03d.h5" % epoch)
        if os.path.exists(weightFile):
            os.remove(weightFile)
        model.save_weights(weightFile)

        # also save accuracies (for diagnostic purposes)
        accFile = os.path.join(args.outDir, 'acc_epoch_%03d.npy' % epoch)
        np.save(accFile, acc)

        # Evaluate performance on validation data.
        logger.info('epoch %d complete. validating...' % epoch)
        Prob, acc = _evaluate(logger, model, Xvalid, Yvalid, omitLabels=[-1,])
        logger.info('accuracy on validation data: %0.2f%%' % acc)
        estFile = os.path.join(args.outDir, "validation_epoch_%03d.npy" % epoch)
        np.save(estFile, Prob)

        logger.info('Finished!')


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
