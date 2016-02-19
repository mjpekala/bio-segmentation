""" Functions for working with image data volumes.

 Data volumes
 ------------
 This module assumes that all image data volumes (by convention, 
 have the variable name X) have dimensions:

          (z-slices, #channels, rows, cols) 

  For example, a grayscale EM volume with 10 slices each of
  512x512 pixels will have shape
          (10, 1, 512, 512)

  This module assumes that class label volumes (which conventionally will
  be named Y) will only every have a single class label for each pixel. 
  Hence, the #channels dimension is unncessary.
  Therefore, these volumes will have dimension

          (z-slices, rows, cols)
 
  This particular ordering of dimensions was chosen to maximize
  compatibility with the CNN frameworks (and also works well with
  numpy's slicing, which implicitly squeezes from the left).

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



def load_cube(dataFile, dtype='float32', addChannel=True):
    """ Loads a data volume.  This could be image data or per-pixel 
    class labels.

    Uses the file extension to determine the underlying data format.
    Note that the Matlab data format currently assumes you saved using
    the -v7.3 flag (hdf5 under the hood).

      dataFile   : the full filename containing the data volume
      dtype      : data type that should be used to represent the data
      d4         : if true, adds the fourth channel dimension to the tensor
    """
    
    # Raw TIFF data
    if dataFile.lower().endswith('.tif') or dataFile.lower().endswith('.tiff'):
        X = load_tiff_data(dataFile, dtype)

    # Matlab data 
    elif dataFile.lower().endswith('.mat'):
        # currently assumes matlab 7.3 format files - i.e. hdf
        # 
        # Note: matlab uses fortran ordering, hence the permute/transpose here.
        d = h5py.File(dataFile, 'r')
        if len(d.keys()) > 1:
            raise RuntimeError('mat file has more than one key - not yet supported!')
        X = (d.values()[0])[:]
        X = np.transpose(X, (0,2,1))

    # Numpy file 
    else:
        # otherwise assumpy numpy serialized object. 
        X = np.load(dataFile)

    # No matter the source, make sure the type and dimensions are right.
    X = X.astype(dtype)
    if addChannel and X.ndim == 3:
        X = X[:, np.newaxis, :, :]

    return X


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
        if Xi.ndim == 2:
            Xi = Xi[np.newaxis, ...] # add slice dimension
        X.append(Xi)
        try:
            dataImg.seek(dataImg.tell()+1)
        except EOFError:
            break # this just means hit end of file (not really an error)

    X = np.concatenate(X, axis=0)  # list of 2d -> tensor
    return X




def number_classes(Yin, omitLabels=[]):
    """Remaps class labels to contiguous natural numbers starting at 0.
    In many frameworks (e.g. caffe) class labels are mapped to indices at
    the output of the CNN; hence this remapping.

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
    """Given a tensor X with dimension 
         (z, c, row, col) 

    produces a new tensor with dimensions
         (z, c, row+2*nPixels, row+2*nPixels)

    tensor with an "outer border" created by mirroring pixels along
    the outer border of X
    """
    assert(nPixels > 0)

    z,c,m,n = X.shape

    Xm = np.zeros((z, c, m+2*nPixels, n+2*nPixels), dtype=X.dtype)
   
    # the interior of Xm is just X
    Xm[:, :, nPixels:m+nPixels, nPixels:n+nPixels] = X

    # Note we do *not* replicate the pixel on the outer edge of the original image.
    for ii in range(z):
        for jj in range(c):
            # left edge 
            Xm[ii,jj, :, 0:nPixels] = np.fliplr(Xm[ii,jj, :, (nPixels+1):(2*nPixels+1)])

            # right edge
            Xm[ii,jj, :, -nPixels:] = np.fliplr(Xm[ii,jj, :, (-2*nPixels-1):(-nPixels-1)])

            # top edge (fills in corners)
            Xm[ii,jj, 0:nPixels, :] = np.flipud(Xm[ii,jj, (nPixels+1):(2*nPixels+1), :])

            # bottom edge (fills in corners)
            Xm[ii,jj, -nPixels:, :] = np.flipud(Xm[ii,jj, (-2*nPixels-1):(-nPixels-1), :])

    return Xm



def rescale_01(X, perChannel=True):
    """Rescales all values to live in [0,1].
    """
    if not perChannel: 
        xMin = np.min(X);  xMax = np.max(X)
        return (X - xMin) / (xMax - xMin)
    else:
        Xout = np.zeros(X.shape, dtype=X.dtype)
        for c in range(X.shape[1]):
            xMin = np.min(X[:,c,...])
            xMax = np.max(X[:,c,...])
            Xout[:,c,:,:] = (X[:,c,...] - xMin) / (xMax - xMin)
        return Xout



#-------------------------------------------------------------------------------
# Functions for extracting tiles from images
#-------------------------------------------------------------------------------

def stratified_interior_pixel_generator(Y, borderSize, batchSize,
                                        mask=None,
                                        omitSlices=[],
                                        omitLabels=[],
                                        stopAfter=-1):
    """An iterator over pixel indices with the property that pixels of 
    different class labels are represented in equal proportions.

    Warning: this is fairly memory intensive (pre-computes the 
    entire list of indices).
    An alternative (an approxmation) might have been random sampling...
    """
    [s,m,n] = Y.shape
    yAll = np.unique(Y)
    yAll = [y for y in yAll if y not in omitLabels]
    assert(len(yAll) > 0)

    # Used to restrict the set of pixels under consideration.
    bitMask = np.ones(Y.shape, dtype=bool)
    bitMask[omitSlices,:,:] = 0

    if borderSize > 0: 
        bitMask[:, 0:borderSize, :] = 0
        bitMask[:, (m-borderSize):m, :] = 0
        bitMask[:, :, 0:borderSize] = 0
        bitMask[:, :, (n-borderSize):n] = 0

    if mask is not None:
        bitMask = bitMask & mask

    # Determine how many instances of each class to report
    # (the minimum over the total number)
    cnt = [np.sum( (Y==y) & bitMask ) for y in yAll]
    #print('[emlib]: num. pixels per class label is: %s' % str(cnt))
    cnt = min(cnt)
    #print('[emlib]: will draw %d samples from each class' % cnt)

    # Stratified sampling
    Idx = np.zeros((0,3), dtype=np.int32)  # three columns because there are 3 dimensions in the tensor
    for y in yAll:
        tup = np.nonzero( (Y==y) & bitMask )
        Yi = np.column_stack(tup)
        np.random.shuffle(Yi)
        Idx = np.vstack((Idx, Yi[:cnt,:]))

    # one last shuffle to mix all the classes together
    np.random.shuffle(Idx)   # note: modifies array in-place

    # (optional) implement early stopping
    if (stopAfter > 0) and (stopAfter <= Idx.shape[0]):
        Idx = Idx[:stopAfter,...]

    # return in subsets of size batchSize
    for ii in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0] - ii)
        yield Idx[ii:(ii+nRet)], (1.0*ii+nRet)/Idx.shape[0]


 
def interior_pixel_generator(X, borderSize, batchSize,
                             mask=None,
                             omitSlices=[]):
    """An iterator over pixel indices in the interior of an image.

    Warning: this is fairly memory intensive (pre-computes the entire 
    list of indices).

    Note: we could potentially speed up the process of extracting subtiles by
    creating a more efficient implementation; however, some simple timing tests
    indicated are spending orders of magnitude more time doing CNN operations 
    so there is no pressing need to optimize tile extraction at the moment.

    Parameters:
      X          := An image tensor with *either* dimensions:
                       (#slices, #channels, width, height) 
                    or
                       (#slices, width, height) 
      
      borderSize := Specifies a border width - all pixels in this exterior 
                    border will be excluded from the return value.
                    
      batchSize  := The number of pixels that should be returned each iteration.
      
      mask       := a boolean tensor the same size as X where 0/false means 
                    omit the corresponding pixel
    """
    if X.ndim == 4:
        [s,c,m,n] = X.shape
        # if the mask has a channel dimension, collapse it
        if mask is not None and mask.ndim == 4: 
            mask = np.all(mask, axis=1)
    else:
        [s,m,n] = X.shape

    # Used to restrict the set of pixels under consideration.
    # Note that the number of channels plays no role here.
    bitMask = np.ones([s,m,n], dtype=bool)
    bitMask[omitSlices,:,:] = False

    bitMask[:, 0:borderSize, :] = False
    bitMask[:, (m-borderSize):m, :] = False
    bitMask[:, :, 0:borderSize] = False
    bitMask[:, :, (n-borderSize):n] = False
    
    if mask is not None:
        assert(np.all(mask.shape == bitMask.shape))
        bitMask = bitMask & mask

    Idx = np.column_stack(np.nonzero(bitMask))

    # return in subsets of size batchSize
    for ii in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0] - ii)
        yield Idx[ii:(ii+nRet)], (1.0*ii+nRet)/Idx.shape[0]




class SimpleTileExtractor:
    """ Encapsulates the process of extracting tiles/windows centered at
    provided pixel locations.  This includes issues with mirroring 
    to handle edge conditions.

    Makes a copy of X under the hood, so may be inappopriate for large
    data volumes.
    """

    def __init__(self, tileWidth, X, Y=None):
        # The tile dimension must be odd and > 1
        assert(np.mod(tileWidth,2) == 1)
        assert(tileWidth > 1)

        tileRadius = int(tileWidth/2)
        self._X = mirror_edges(X, tileRadius)

        # Note: we do not (yet) know how many tiles will be in the batch.
        #       Defer actually allocating memory until later
        nChannels = X.shape[1]
        self._Xb = np.zeros([0, nChannels, tileWidth, tileWidth], dtype=np.float32)


        if (Y is not None) and (Y.size > 0):
            # class labels will be indices into a one-hot vector; make sure
            # the labels are suitable for this purpose.
            yAll = np.unique(Y).astype(np.int32)
            nClasses = yAll.size
            assert(np.min(yAll) == 0)
            assert(np.max(yAll) == nClasses-1)

            self._Yb = np.zeros([0, nClasses], dtype=np.float32)
            self._Y = Y
        else:
            self._Yb = np.zeros([0,0])


    def extract(self, Idx):
        """
        Idx : an (n x 3) matrix, where the columns correspond to pixel
              depth, row and column.
        """
        assert(Idx.shape[1] == 3)  
        n = Idx.shape[0]   # n := batch size
        tileWidth = self._Xb.shape[2]

        # (re)allocate memory, if needed
        # Note that if n is less than the previous batch size, old examples
        # will be reused.  This is intentional.
        if n > self._Xb.shape[0]:
            self._Xb = np.zeros( (n,) + self._Xb.shape[1:], dtype=np.float32)
            self._Yb = np.zeros( (n, self._Yb.shape[1]), dtype=np.float32)

        # Map pixel indices to tiles (and possibly class labels)
        for jj in range(n):
            # Note: Idx refers to coordinates in X, so we must account for
            #       the fact that _X has mirrored edges
            # Note: the code below is correcting for the 
            a = Idx[jj,1]
            b = Idx[jj,1] + tileWidth
            c = Idx[jj,2]
            d = Idx[jj,2] + tileWidth

            self._Xb[jj, :, :, :] = self._X[ Idx[jj,0], :, a:b, c:d ]

            if self._Yb.size > 0: 
                yj = self._Y[ Idx[jj,0], Idx[jj,1], Idx[jj,2] ] 
                # store the class label as a 1 hot vector
                self._Yb[jj,:] = 0
                self._Yb[jj,yj] = 1

        if self._Yb.size:
            return self._Xb, self._Yb
        else:
            return self._Xb



#-------------------------------------------------------------------------------

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
