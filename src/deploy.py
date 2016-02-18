"""Deploys a previously trained CNN on image data.

 See train.py for an example of how to train a CNN.


 Notes: 
   o Currently assumes input image data is single channel (i.e. grayscale).
   o Output probability values of NaN indicate pixels that were not 
     evaluated (e.g. due to boundary conditions or downsampling)

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
import scipy
import pdb

import emlib
import models as emm
from train import _minibatch_setup

from sobol_lib import i4_sobol_generate as sobol




def _deploy_mode_args():
    """Parameters for deploying a CNN.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x', dest='emFile', 
		    type=str, required=True,
		    help='Filename of volume to evaluate')

    parser.add_argument('--model', dest='model', 
		    type=str, default='ciresan_n3',
		    help='name of CNN model to use (python function)')
    parser.add_argument('--weight-file', dest='weightFile', 
		    type=str, required=True,
		    help='CNN weights to use')

    parser.add_argument('--slices', dest='slices', 
		    type=str, default='', 
		    help='(optional) subset of slices to evaluate')
    parser.add_argument('--eval-pct', dest='evalPct', 
		    type=float, default=1.0, 
		    help='(optional) Percent of pixels to evaluate (in [0,1])')

    parser.add_argument('--out-file', dest='outFile', 
		    type=str, required=True,
		    help='Ouput file name (will contain probability estimates)')

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.outFile)):
        os.makedirs(os.path.dirname(args.outFile))

    if not args.outFile.endswith('.npy'):
        args.outFile += '.npy'

    # Map strings into python objects.  
    # A little gross to use eval, but life is short.
    str_to_obj = lambda x: eval(x) if x else []
    
    args.slices = str_to_obj(args.slices)
    
    return args



def _downsample_mask(X, pct):
    """ Create a boolean mask indicating which subset of X should be 
    evaluated.
    """
    if pct < 1.0: 
        Mask = np.zeros(X.shape, dtype=np.bool)
        m = X.shape[-2]
        n = X.shape[-1]
        nToEval = np.round(pct*m*n).astype(np.int32)
        idx = sobol(2, nToEval ,0)
        idx[0] = np.floor(m*idx[0])
        idx[1] = np.floor(n*idx[1])
        idx = idx.astype(np.int32)
        Mask[:,idx[0], idx[1]] = True
    else:
        Mask = np.ones(X.shape, dtype=np.bool)

    return Mask



def _evaluate(logger, model, X, batchSize=100, evalPct=1.0):
    """Evaluate model on held-out data.

    Returns:
      Prob : a tensor of per-pixel probability estimates with dimensions:
         (#layers, #classes, width, height)

    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    Xi, unused, tileRadius = _minibatch_setup(model, batchSize)

    lastChatter = -2
    startTime = time.time()

    # identify subset of volume to evaluate
    Mask = _downsample_mask(X, evalPct)
    logger.info('after masking, will evaluate %0.2f%% of data' % (100.0*np.sum(Mask)/Mask.size))

    # Mirror edges (so that we can evaluate the entire volume)
    Xm = emlib.mirror_edges(X, tileRadius)

    Mask = emlib.mirror_edges(Mask, tileRadius)
    Mask[:,:,0:tileRadius,:] = False
    Mask[:,:,-tileRadius:,:] = False
    Mask[:,:,:,0:tileRadius] = False
    Mask[:,:,:,-tileRadius:] = False

    # Create storage for class probabilities.
    # Note that we store all class probabilities, even if this
    # is a binary classification problem (in which case p(1) = 1 - p(0)).
    # We do this to support multiclass classification seamlessly.
    [numZ, numChan, numRows, numCols] = Xm.shape
    numClasses = model.output_shape[-1]
    Prob = np.nan * np.ones([numZ, numClasses, numRows, numCols], 
                            dtype=np.float32)

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = emlib.interior_pixel_generator(Xm, tileRadius, batchSize, mask=Mask)

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        n = Idx.shape[0] # may be < batchSize on final iteration
        assert(Idx.shape[1] == 3)

        # Map pixel indices to tiles
        for jj in range(n):
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, :, :, :] = Xm[ Idx[jj,0], :, a:b, c:d ]

        prob = model.predict_on_batch(Xi)
        Prob[Idx[:,0], :, Idx[:,1], Idx[:,2]] = prob[0][:n,:]

        # notify user re. progress
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  
            lastChatter = elapsed
            logger.info("  last pixel %s (%0.2f%% complete)" % (str(Idx[-1,:]), 100.*epochPct))

    # discard mirrored portion before returning
    Prob = Prob[:, :, tileRadius:-tileRadius, tileRadius:-tileRadius]
    return Prob



#-------------------------------------------------------------------------------
if __name__ == "__main__":
    args = _deploy_mode_args()

    # setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s:%(name)s:%(levelname)s]  %(message)s'))
    logger.addHandler(ch)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load data volume
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = emlib.load_cube(args.emFile)
    if args.slices:
        X = X[args.slices,:,:]

    # rescale features to live in [0, 1]
    X = emlib.rescale_01(X, perChannel=True)

    logger.info('volume dimensions:   %s' % str(X.shape))
    logger.info('values min/max:      %g, %g' % (np.min(X), np.max(X)))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info('initializing CNN...')
    model = getattr(emm, args.model)() 
    model.compile(optimizer='sgd',   # not used, but required by keras
                  loss='categorical_crossentropy',
                  class_mode='categorical')
    model.load_weights(args.weightFile)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do it
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info('evaluating volume...')
    Prob = _evaluate(logger, model, X, evalPct=args.evalPct)
    np.save(args.outFile, Prob)
    scipy.io.savemat(args.outFile.replace('.npy', '.mat'), {'P' : Prob})

    logger.info('Complete!  Probabilites stored in file %s' % args.outFile)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
