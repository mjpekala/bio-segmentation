"""Deploys a previously trained CNN on image data.

 See train.py for an example of how to train a CNN.


 Notes: 
   o currently assumes input image data is single channel (i.e. grayscale)

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

import emlib
import models as emm
from train import _minibatch_setup




def _deploy_mode_args():
    """Parameters for deploying a CNN.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--x', dest='emFile', 
		    type=str, required=True,
		    help='Filename of volume to evaluate')

    parser.add_argument('--weight-file', dest='weightFile', 
		    type=str, required=True,
		    help='CNN weights to use')

    parser.add_argument('--slices', dest='slices', 
		    type=str, default='', 
		    help='(optional) subset of slices to evaluate')
    parser.add_argument('--eval-pct', dest='evalPct', 
		    type=float, default=1, 
		    help='(optional) Percent of pixels to evaluate (in [0,1])')

    parser.add_argument('--out-file', dest='outFile', 
		    type=str, required=True,
		    help='Ouput file name (will contain probability estimates)')

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.outFile)):
        os.makedirs(os.path.dirname(args.outFile))

    # Map strings into python objects.  
    # A little gross to use eval, but life is short.
    str_to_obj = lambda x: eval(x) if x else []
    
    args.slices = str_to_obj(args.slices)
    
    return args




def _evaluate(logger, model, X, batchSize=100):
    """Evaluate model on held-out data.
    """
    #----------------------------------------
    # Pre-allocate some variables & storage.
    #----------------------------------------
    Xi, unused, tileRadius = _minibatch_setup(model, batchSize)
    Prob = np.nan * np.ones(X.shape, dtype=np.float32)

    lastChatter = -2
    startTime = time.time()

    #----------------------------------------
    # Loop over mini-batches
    #----------------------------------------
    it = emlib.interior_pixel_generator(X, tileRadius, batchSize)

    for mbIdx, (Idx, epochPct) in enumerate(it): 
        n = Idx.shape[0] # may be < batchSize on final iteration

        # Map pixel indices to tiles
        for jj in range(n):
            a = Idx[jj,1] - tileRadius 
            b = Idx[jj,1] + tileRadius + 1 
            c = Idx[jj,2] - tileRadius 
            d = Idx[jj,2] + tileRadius + 1 
            Xi[jj, 0, :, :] = X[ Idx[jj,0], a:b, c:d ]

        prob = model.predict_on_batch(Xi)
        Prob[Idx[:,0], Idx[:,1], Idx[:,2]] = prob[0][:n,1]

        # notify user re. progress
        elapsed = (time.time() - startTime) / 60.0
        if (lastChatter+2) < elapsed:  
            lastChatter = elapsed
            logger.info("  last pixel %s (%0.2f%% complete)" % (str(Idx[-1,:]), 100.*epochPct))


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
    xMin = np.min(X);  xMax = np.max(X)
    X = (X - xMin) / (xMax - xMin)

    logger.info('volume dimensions:   %s' % str(X.shape))
    logger.info('values min/max:      %g, %g' % (np.min(X), np.max(X)))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initialize CNN
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info('initializing CNN...')
    model = emm.ciresan_n3()
    model.compile(optimizer='sgd',   # not used, but required by keras
                  loss='categorical_crossentropy', 
                  class_mode='binary')
    model.load_weights(args.weightFile)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Do it
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info('evaluating volume...')
    Prob = _evaluate(logger, model, X)
    estFile = os.path.join(args.outDir, "validation_epoch_%03d.npy" % epoch)
    np.save(args.outFile, Prob)

    logger.info('Complete!  Probabilites stored in file %s' % estFile)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
