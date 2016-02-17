This repository provides simple examples of using deep networks to segment electron microscropy and other biological image data.  The basic approach is based on the sliding-window approach described in [this](http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images) NIPS 2013 paper by Dan Ciresan and others at IDSIA.


Note: this code is in an experimental state and subject to change.


## Quick start

### Prerequisites

This code requires you install [Keras](http://keras.io/) along with a suitable backend; we are using [Theano](http://deeplearning.net/software/theano/).  Fortunately these packages are quite easy to install using pip; instructions are availble on the respective websites.


### Running Experiments

An example is provided for the [ISBI 2012 membrane segmentation challenge problem](http://brainiac2.mit.edu/isbi_challenge/).  Before running this example you will need to download the ISBI 2012 data set (using the link above) and place the .tif files in the directory ./data/ISBI2012. At this point you are ready to train and deploy a CNN.  The provided [Makefile](./Makefile) automates this process:

```
Targets supported by this makefile:
   train-isbi  : trains a model using ISBI 2012 data set
   deploy-isbi : evaluate ISBI 2012 test data
   test        : runs unit tests
   clean       : deletes any previously trained models

Training can take a long time.  You may want to do e.g.
  (nohup make train-isbi &) > train.out
```


### Notes
The code makes a few assumptions as of this writing; some of these are straightforward to relax (and hopefully will be soon)

- Assumes input data volumes are one channel (grayscale)
- Assumes you want to solve a dense (i.e. per-pixel) binary classification problme
- We have not yet done any sophisticated hyper-parameter optimization (although we have used techniques from the Gaussian process community on other efforts, and may incorporate this at some point in he future).


The sliding window approach to generating dense predictions worked well for the ISBI 2012 challenge problem; however, it is somewhat computationally expensive.  There are more sophisticated techniques for generating dense prediction problems that one might want to consider.   Also, other more recent papers have improved upon the ISBI 2012 results somewhat and may be of interest if task performance is paramount.  A non-exhaustive list of potential references along these lines is provided below:

o TODO
o TODO
