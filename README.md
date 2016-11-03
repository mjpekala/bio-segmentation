# Convolutional Neural Nets for Dense (per-pixel) Image Classification
This repository provides simple examples of using deep networks to segment electron microscropy and other biological image data.  The basic approach is based on the sliding-window technique described in [this](http://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images) NIPS 2013 paper by Dan Ciresan and others at IDSIA.  Note that while the sliding-window technique can provide reasonable results, it is not the only approach for this class of problems (and is fairly computationally inefficient; see notes below for more details).

Note: this code is in an experimental state and subject to change; it is made available on an "as is" basis in the hope that it may be useful to others.  Use at your own risk :)


## Quick start

**Update**: This latest version of the code has been updated based on changes to the Keras API.  The version in master should now be compatible with Keras 1.1.0.  However, at this point in time, you are probably better off using more modern methods for this type of problem (see references below).

### Prerequisites

This code requires you install [Keras](http://keras.io/) along with a suitable backend; we are using [Theano](http://deeplearning.net/software/theano/).  These packages are quite easy to install using pip; instructions are available on the respective websites.


### ISBI 2012 Example

An example is provided for the [ISBI 2012 membrane segmentation challenge problem](http://brainiac2.mit.edu/isbi_challenge/).  Before running this example you will need to download the ISBI 2012 data set (using the link above) and place the .tif files in the directory ./data/ISBI2012. At this point you are ready to train and deploy a CNN.  The provided [Makefile](./Makefile) automates this process.

**To train a CNN**:
```
    make -f Makefile.isbi train
```

Note that training can take a long time; if you are running the code on a remote system you may prefer to do something like:
```
(nohup make -f Makefile.isbi GPU=2 train &) > train.large.isbi.out
```

**To deploy a CNN**, once training is complete you can evaluate (subsets of) the ISBI test volume via one of:
```
    make -f Makefile.isbi deploy 
    make -f Makefile.isbi deploy-slice0
    make -f Makefile.isbi deploy-sub
```

Alternatively, there is a [simple example](./examples/isbi2012_deploy.ipynb) in the form of a Jupyter notebook.


## Comments About Performance
The sliding window approach to generating dense predictions worked well for the ISBI 2012 challenge problem; however, it is somewhat computationally expensive.  There are more sophisticated techniques for solving dense prediction problems that one might want to consider if you wish to do things "at scale".  We provide some simple downsampling capabilities that, in conjunction with some fast interpolation or inpainting techniques, can speed things up with relatively mild degradation of task performance (at least on ISBI 2012).  A (non-exhaustive) list of papers that provide some alternative (and more sophisticated) approaches to solving dense prediction problems is given below:


- Sermanet, Pierre et al. "OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks." arXiv preprint arXiv:1312.6229 (2013).

- Giusti, Alessandro, et al. "Fast image scanning with deep max-pooling convolutional neural networks." arXiv preprint arXiv:1302.1700 (2013).

- Iandola et al. "DenseNet: Implementing Efficient ConvNet Descriptor Pyramids," Technical Report UC Berkeley, 2014.

- Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

- Tschopp, Fabian. "Efficient Convolutional Neural Networks for Pixelwise Classification on Heterogeneous Hardware Systems." arXiv preprint arXiv:1509.03371 (2015).

- Ronneberger, Fischer, Brox "U-Net: Convolutional Networks for Biomedical Image Segmentation." 2015. [link](https://arxiv.org/abs/1505.04597)

In terms of this particular implementation, as of this writing we have not (a) done any kind of sophisticated CNN design or hyperparameter optimization and (b) we are only using a single CNN (vs. the ensemble described in the primary reference).  Additionally, more recent efforts have improved upon the task performance of ISBI 2012.   You may want to take a look at papers written by those currently atop the ISBI 2012 leader board for inspiration.   At a minimum, you should be aware that the code in this repository provides a reasonable starting baseline as opposed to a state-of-the-art result.

mjp, March 2016
