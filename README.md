This repository provides simple examples of using deep networks to segment electron microscropy and other biological image data.  The basic approach is based on the sliding-window technique described in [this](http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images) NIPS 2013 paper by Dan Ciresan and others at IDSIA.


Note: this code is in an experimental state and subject to change.


## Quick start

### Prerequisites

This code requires you install [Keras](http://keras.io/) along with a suitable backend; we are using [Theano](http://deeplearning.net/software/theano/).  Fortunately these packages are quite easy to install using pip; instructions are availble on the respective websites.


### ISBI 2012 Example

An example is provided for the [ISBI 2012 membrane segmentation challenge problem](http://brainiac2.mit.edu/isbi_challenge/).  Before running this example you will need to download the ISBI 2012 data set (using the link above) and place the .tif files in the directory ./data/ISBI2012. At this point you are ready to train and deploy a CNN.  The provided [Makefile](./Makefile) automates this process.

To train a CNN do:
```
    make -f Makefile.isbi train
```

Once training is complete (or, alternately, using the provided weights we generated) you can evaluate (subsets of) the ISBI test volume via one of:
```
    make -f Makefile.isbi deploy 
    make -f Makefile.isbi deploy-slice0
    make -f Makefile.isbi deploy-sub
```

Assuming you have data volumes with similar structure (grayscale with dense labels) it should be straightforward to modify the Makefile for your problem of interest.


## Comments About Performance
The sliding window approach to generating dense predictions worked well for the ISBI 2012 challenge problem; however, it is somewhat computationally expensive.  There are more sophisticated techniques for solving dense prediction problems that one might want to consider if you wish to do things "at scale".  We provide some simple downsampling capabilities that, in conjunction with some fast interpolation or inpainting techniques, can speed things up with relatively mild degradation of task performance (at least on ISBI 2012).  A non-exhaustive list of papers that provide some alternative approaches to solving dense prediction problems is given below:


- Giusti, Alessandro, et al. "Fast image scanning with deep max-pooling convolutional neural networks." arXiv preprint arXiv:1302.1700 (2013).

- Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

- Tschopp, Fabian. "Efficient Convolutional Neural Networks for Pixelwise Classification on Heterogeneous Hardware Systems." arXiv preprint arXiv:1509.03371 (2015).

In terms of this particular implementation, as of this writing we have not (a) done any kind of sophisticated CNN hyperparameter optimization and (b) we are only using a single CNN (vs. the ensemble described in the primary reference).  Additionally, more recent efforts have improved upon the task performance of ISBI 2012.   You may want to take a look at papers written by those currently atop the ISBI 2012 leader board for inspiration.   At a minimum, you should be aware that the code provided here delivers a reasonable starting baseline as opposed to a cutting edge result.  
