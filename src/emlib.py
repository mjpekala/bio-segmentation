""" Functions for working with image data volumes.

The functions in this module all assume an image volume has dimensions:
      #slices x width x height
      
The choice to have the z dimension first is for compatability with numpy's slicing,
which implicitly squeezes 3d to 2d when dimensions are ordered this way.
This feels a little backwards if you are used to working in Matlab and if you
use matplotlib.

"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2016, JHU/APL"
__license__ = "Apache 2.0"


import os, sys, re
import pdb

import numpy as np
from PIL import Image

from scipy.signal import convolve2d
from scipy.io import loadmat
import h5py



def load_cube(dataFile, dtype='float32'):
    """ Loads a data volume.  This could be image data or per-pixel class labels.

    Uses the file extension to determine the underlying data format.
    Note that the Matlab data format currently assumes you saved using
    the -v7.3 flag (hdf5 under the hood).

    dataFile  := the full filename containing the data volume
    dtype     := data type that should be used to represent the data
    """
    
    # Raw TIFF data
    if dataFile.endswith('.tif') or dataFile.endswith('.tiff'):
        return load_tiff_data(dataFile, dtype)

    # Matlab data 
    elif dataFile.endswith('.mat'):
        # currently assumes matlab 7.3 format files - i.e. hdf
        # 
        # Note: matlab uses fortran ordering, hence the permute/transpose here.
        d = h5py.File(dataFile, 'r')
        if len(d.keys()) > 1:
            raise RuntimeError('mat file has more than one key - not yet supported!')
        X = (d.values()[0])[:]
        X = np.transpose(X, (0,2,1))
        return X.astype(dtype)

    # Numpy file 
    else:
        # assumpy numpy serialized object
        return np.load(dataFile).astype(dtype)
 


def load_tiff_data(dataFile, dtype='float32'):
    """ Loads data from a multilayer .tif file.

    dataFile := the tiff file name
    dtype    := data type to use for the returned tensor
    
    Returns result as a numpy tensor with dimensions (layers, width, height).
    """
    if not os.path.isfile(dataFile):
        raise RuntimeError('could not find file "%s"' % dataFile)
    
    # load the data from multi-layer TIF files
    dataImg = Image.open(dataFile)
    X = [];
    for ii in xrange(sys.maxint):
        Xi = np.array(dataImg, dtype=dtype)
        Xi = np.reshape(Xi, (1, Xi.shape[0], Xi.shape[1]))  # add a slice dimension
        X.append(Xi)
        try:
            dataImg.seek(dataImg.tell()+1)
        except EOFError:
            break # this just means hit end of file (not really an error)

    X = np.concatenate(X, axis=0)  # list -> tensor
    return X



def save_tiff_data(X, outDir, baseName='X_'):
    """Unfortunately, it appears PIL can only load multi-page .tif files
    (i.e. it cannot create them).  So the approach is to create them a
    slice at a time, using this function, and then combine them after
    the fact using convert, e.g.

    convert slice??.tif -define quantum:format=floating-point combinedF.tif
    """
    assert(len(X.shape) == 3)
    for ii in range(X.shape[0]):
        fn = os.path.join(outDir, baseName + "%02d" % ii + ".tif")
        im = Image.fromarray(X[ii,:,:].astype('float32'))
        im.save(fn)



def label_epsilon(Y, epsilon=3, n=9, targetClass=255, verbose=True):
    """Given a tensor of per-pixel class labels, return a new tensor of labels
    where the class is 1 iff at least n pixels in an epsilon ball are the target class.

    Note: for the moment, a "epsilon ball" is a rectangle of radius epsilon.

    Example: given a tensor of per-pixel binary class labels, we want to create
    a new training set where the class label of a pixel is 1 if there are enough
    pixels nearby with class 1.  However, if there are other +1 pixels nearby but
    not enough to be considered a positive class, we also want to flag this as
    an instance we don't want to train on (so negative examples have no +1 pixels
    nearby).

       Y = emlib.load_cube('/Users/pekalmj1/Data/SynapseData3/Y_train2.mat')
       eps=15
       Y1 = emlib.label_epsilon(Y, epsilon=eps, n=15, targetClass=1)
       Yany = emlib.label_epsilon(Y, epsilon=eps, n=1, targetClass=1)  # Y may have labels other that {+1,0}
       Yeps = np.zeros(Y.shape, dtype=np.int32)  
       Yeps[Yany==1] = -1 
       Yeps[Y1==1] = 1 
       scipy.io.savemat("Yeps.mat", {'Y' : np.transpose(Yeps, (1,2,0))})
    
    """
    if len(Y.shape) != 3:
        raise RuntimeError('Sorry - Y must be a 3d tensor')

    Y = (Y == targetClass).astype(np.int32)
    d = 2*epsilon+1
    W = np.ones((d,d), dtype=bool)
    Yeps = np.zeros(Y.shape)
    for ii in range(Y.shape[0]):
        if verbose: print('[label_epsilon]: processing slice %d (of %d)' % (ii, Y.shape[0]))
        tmp = convolve2d(Y[ii,...], W, boundary='symm')
        Yeps[ii,...] = tmp[epsilon:-epsilon, epsilon:-epsilon]
    return (Yeps >= n)



def number_classes(Yin, omitLabels=[]):
    """Class labels must be contiguous natural numbers starting at 0.
    This is because they are mapped to indices at the output of the CNN.
    This function remaps the input y values if needed.

    Any pixels that should be ignored will have class label of -1.
    """
    if Yin is None: return None

    yAll = np.sort(np.unique(Yin))
    yAll = [y for y in yAll if y not in omitLabels]

    Yout = -1*np.ones(Yin.shape, dtype=Yin.dtype)
    for yIdx, y in enumerate(yAll):
        Yout[Yin==y] = yIdx

    return Yout



def mirror_edges(X, nPixels):
    """Given an (s x m x n) tensor X, generates a new
          s x (m+2*nPixels) x (n+2*nPixels)
    tensor with an "outer border" created by mirroring pixels along
    the outer border of X
    """
    assert(nPixels > 0)
    
    s,m,n = X.shape
    Xm = np.zeros((s, m+2*nPixels, n+2*nPixels), dtype=X.dtype)
    
    Xm[:, nPixels:m+nPixels, nPixels:n+nPixels] = X

    # a helper function for dealing with corners
    flip_corner = lambda X : np.fliplr(np.flipud(X))

    for ii in range(s):
        # top left corner
        Xm[ii, 0:nPixels,0:nPixels] = flip_corner(X[ii, 0:nPixels,0:nPixels])

        # top right corner
        Xm[ii, 0:nPixels,n+nPixels:] = flip_corner(X[ii, 0:nPixels,n-nPixels:])

        # bottom left corner
        Xm[ii, m+nPixels:,0:nPixels] = flip_corner(X[ii, m-nPixels:,0:nPixels])

        # bottom right corner
        Xm[ii, m+nPixels:,n+nPixels:] = flip_corner(X[ii, m-nPixels:,n-nPixels:])

        # top border
        Xm[ii, 0:nPixels, nPixels:n+nPixels] = np.flipud(X[ii, 0:nPixels,:])

        # bottom border
        Xm[ii, m+nPixels:, nPixels:n+nPixels] = np.flipud(X[ii, m-nPixels:,:])

        # left border
        Xm[ii, nPixels:m+nPixels,0:nPixels] = np.fliplr(X[ii, :,0:nPixels])

        # right border
        Xm[ii, nPixels:m+nPixels,n+nPixels:] = np.fliplr(X[ii, :,n-nPixels:])

    return Xm



def stratified_interior_pixel_generator(Y, borderSize, batchSize,
                                        mask=None,
                                        omitSlices=[],
                                        omitLabels=[]):
    """An iterator over pixel indices with the property that pixels of different
    class labels are represented in equal proportions.

    Warning: this is fairly memory intensive (pre-computes the entire list of indices).
    An alternative (an approxmation) might have been random sampling...

    Parameters:
      Y          := a (# slices x width x height) class label tensor

      borderSize := Specifies a border width - all pixels in this exterior border
                    will be excluded from the return value.
                    
      batchSize  := The number of pixels that should be returned each iteration.
      
      mask       := a boolean tensor the same size as X where 0/false means omit
                    the corresponding pixel
    """
    [s,m,n] = Y.shape
    yAll = np.unique(Y)
    yAll = [y for y in yAll if y not in omitLabels]
    assert(len(yAll) > 0)

    # Used to restrict the set of pixels under consideration.
    bitMask = np.ones(Y.shape, dtype=bool)
    bitMask[omitSlices,:,:] = 0

    bitMask[:, 0:borderSize, :] = 0
    bitMask[:, (m-borderSize):m, :] = 0
    bitMask[:, :, 0:borderSize] = 0
    bitMask[:, :, (n-borderSize):n] = 0

    if mask is not None:
        bitMask = bitMask & mask

    # Determine how many instances of each class to report
    # (the minimum over the total number)
    cnt = [np.sum( (Y==y) & bitMask ) for y in yAll]
    print('[emlib]: num. pixels per class label is: %s' % str(cnt))
    cnt = min(cnt)
    print('[emlib]: will draw %d samples from each class' % cnt)

    # Stratified sampling
    Idx = np.zeros((0,3), dtype=np.int32)  # three columns because there are 3 dimensions in the tensor
    for y in yAll:
        tup = np.nonzero( (Y==y) & bitMask )
        Yi = np.column_stack(tup)
        np.random.shuffle(Yi)
        Idx = np.vstack((Idx, Yi[:cnt,:]))

    # one last shuffle to mix all the classes together
    np.random.shuffle(Idx)   # note: modifies array in-place

    # return in subsets of size batchSize
    for ii in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0] - ii)
        yield Idx[ii:(ii+nRet)], (1.0*ii)/Idx.shape[0]


 
def interior_pixel_generator(X, borderSize, batchSize,
                             mask=None,
                             omitSlices=[]):
    """An iterator over pixel indices in the interior of an image.

    Warning: this is fairly memory intensive (pre-computes the entire list of indices).

    Note: we could potentially speed up the process of extracting subtiles by
    creating a more efficient implementation; however, some simple timing tests
    indicate we are spending orders of magnitude more time in CNN operations so
    there is no pressing need to optimize tile extraction at the moment.

    Parameters:
      X          := a (# slices x width x height) image tensor
      
      borderSize := Specifies a border width - all pixels in this exterior border
                    will be excluded from the return value.
                    
      batchSize  := The number of pixels that should be returned each iteration.
      
      mask       := a boolean tensor the same size as X where 0/false means omit
                    the corresponding pixel
    """
    [s,m,n] = X.shape
        
    # Used to restrict the set of pixels under consideration.
    bitMask = np.ones(X.shape, dtype=bool)
    bitMask[omitSlices,:,:] = 0

    bitMask[:, 0:borderSize, :] = 0
    bitMask[:, (m-borderSize):m, :] = 0
    bitMask[:, :, 0:borderSize] = 0
    bitMask[:, :, (n-borderSize):n] = 0
    
    if mask is not None:
        bitMask = bitMask & mask

    Idx = np.column_stack(np.nonzero(bitMask))

    # return in subsets of size batchSize
    for ii in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0] - ii)
        yield Idx[ii:(ii+nRet)], (1.0*ii)/Idx.shape[0]



def metrics(Y, Yhat, display=False): 
    """
    PARAMETERS:
      Y    :  a numpy array of true class labels
      Yhat : a numpy array of estimated class labels (same size as Y)

    o Assumes any class label <0 should be ignored in the analysis.
    o Assumes all non-negative class labels are contiguous and start at 0.
      (so for binary classification, the class labels are {0,1})
    """
    assert(len(Y.shape) == 3)
    assert(len(Yhat.shape) == 3)

    # create a confusion matrix
    # yAll is all *non-negative* class labels in Y
    yAll = np.unique(Y);  yAll = yAll[yAll >= 0] 
    C = np.zeros((yAll.size, yAll.size))
    for yi in yAll:
        est = Yhat[Y==yi]
        for jj in yAll:
            C[yi,jj] = np.sum(est==jj)

    # works for arbitrary # of classes
    acc = 1.0*np.sum(Yhat[Y>=0] == Y[Y>=0]) / np.sum(Y>=0)

    # binary classification metrics (only for classes {0,1})
    nTruePos = 1.0*np.sum((Y==1) & (Yhat==1))
    precision = nTruePos / np.sum(Yhat==1)
    recall = nTruePos / np.sum(Y==1)
    f1 = 2.0*(precision*recall) / (precision+recall)

    if display: 
        for ii in range(C.shape[0]): 
            print('  class=%d    %s' % (ii, C[ii,:]))
        print('  accuracy:  %0.3f' % (acc))
        print('  precision: %0.3f' % (precision))
        print('  recall:    %0.3f' % (recall))
        print('  f1:        %0.3f' % (f1))

    return C, acc, precision, recall, f1


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
