This repository provides simple examples of using deep networks to segment electron microscropy and other biological image data.  The basic approach is based on the sliding-window approach described in [this](http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images) paper by Dan Ciresan and others at IDSIA.


Note: this code is in an experimental state and subject to change.
Furthermore, no attempt has (yet) been made to optimize these CNN models.


## Quick start

### Prerequisites

This code requires you install [Keras](http://keras.io/) along with a suitable backend; we are using [Theano](http://deeplearning.net/software/theano/).  Fortunately these packages are quite easy to install using pip; instructions are availble on the respective websites.


### Running Experiments

An example is provided for the [ISBI 2012 membrane segmentation challenge problem](http://brainiac2.mit.edu/isbi_challenge/).  Before running this example you will need to download the ISBI 2012 data set (using the link above) and place the .tif files in the directory ./data/ISBI2012. At this point you are ready to train and deploy a CNN.  To work with different data volumes or change the experimental parameters, see the [Makefile](./Makefile).

The code makes a few assumptions as of this writing; some of these are straightforward to relax (and hopefully will be soon)

- Assumes input data volumes are one channel (grayscale)
- Assumes you want to solve a dense (i.e. per-pixel) binary classification problme
- We have not yet done any sophisticated hyper-parameter optimization (although we have used techniques from the Gaussian process community on other efforts, and may incorporate this at some point in he future).

