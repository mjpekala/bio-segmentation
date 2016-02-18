"""Unit test for emlib.py

To run:
    PYTHONPATH=../src python test_emlib.py
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2015, JHU/APL"
__license__ = "Apache 2.0"


import unittest
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as smetrics

import emlib



class TestEmlib(unittest.TestCase):
    def test_metrics(self):
        Y = np.random.randint(0,2,size=(2,5,5))
        Yhat = np.random.randint(0,2,size=(2,5,5))

        C,acc,prec,recall,f1 = emlib.metrics(Y, Yhat, display=False)
        prec2, recall2, f12, supp = smetrics(np.reshape(Y, (Y.size,)), 
                np.reshape(Yhat, (Yhat.size,)))

        self.assertAlmostEqual(prec, prec2[1])
        self.assertAlmostEqual(recall, recall2[1])
        self.assertAlmostEqual(f1, f12[1])

        
    def test_number_classes(self):
        y = np.array([-10, 10, 1, 3])
        yn = emlib.number_classes(y, [-10])
        self.assertTrue(np.all(yn == [-1, 2, 0, 1]))
        yn = emlib.number_classes(y)
        self.assertTrue(np.all(yn == [0, 3, 1, 2]))


    def test_mirror_edges(self):
        X = np.random.rand(10,2,3,3);
        b = 2  # b := border size
        Xm = emlib.mirror_edges(X,b)

        # make sure the result has the proper size
        assert(Xm.shape[0] == X.shape[0]);
        assert(Xm.shape[1] == X.shape[1]);
        assert(Xm.shape[2] == X.shape[2]+2*b);
        assert(Xm.shape[3] == X.shape[3]+2*b);

        # make sure the data looks reasonable
        self.assertTrue(np.all(Xm[:,:,:,b-1] == Xm[:,:,:,b]))
        self.assertTrue(np.all(Xm[:,:, b:-b, b:-b] == X))

        ## another test case
        X = np.zeros((1,1,10,10))
        X[0,0,2,:] = 1
        Xm = emlib.mirror_edges(X, 3)
        self.assertTrue(np.all(Xm[0,0,0,:] == 1))
        self.assertTrue(np.all(Xm[0,0,1,:] == 0))
        self.assertTrue(np.all(Xm[0,0,2,:] == 0))
        self.assertTrue(np.all(Xm[0,0,3,:] == 0))
        self.assertTrue(np.all(Xm[0,0,4,:] == 0))
        self.assertTrue(np.all(Xm[0,0,5,:] == 1))
        self.assertTrue(np.all(Xm[0,0,6,:] == 0))


    def test_rescale_01(self):
        X = np.random.rand(10,3,3,3)
        Xr = emlib.rescale_01(X, perChannel=False)
        self.assertTrue(np.max(Xr) <= 1.0)
        self.assertTrue(np.min(Xr) >= 0.0)
        Xr = emlib.rescale_01(X, perChannel=True)
        self.assertTrue(np.max(Xr) <= 1.0)
        self.assertTrue(np.min(Xr) >= 0.0)

        X = np.random.rand(10,3,100,100)
        X[:,0,...] *= 1000.0
        Xr = emlib.rescale_01(X, perChannel=False)
        self.assertTrue(np.max(Xr[:,1,...]) <= .1)
        Xr = emlib.rescale_01(X, perChannel=True)
        self.assertTrue(np.max(Xr[:,1,...]) > .8)



    def test_interior_pixel_generator(self):
        b = 10  # b := border size
        Z = np.zeros((2,100,100), dtype=np.int32)
        for idx, pct  in emlib.interior_pixel_generator(Z,b,30):
            Z[idx[:,0],idx[:,1],idx[:,2]] += 1

        self.assertTrue(np.all(Z[:,b:-b,b:-b]==1))
        Z[:,b:-b,b:-b] = 0
        self.assertTrue(np.all(Z==0))

        # Make sure it works with 4d tensors as well
        Z = np.zeros((2,3,100,100), dtype=np.int32)
        for idx, pct  in emlib.interior_pixel_generator(Z,b,30):
            Z[idx[:,0],:,idx[:,1],idx[:,2]] += 1

        self.assertTrue(np.all(Z[:,:,b:-b,b:-b]==1))
        Z[:,:,b:-b,b:-b] = 0
        self.assertTrue(np.all(Z==0))

        
    def test_stratified_interior_pixel_generator(self):
        b = 10  # b := border size

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For a 50/50 split of pixels in the interior, the generator
        # should reproduce the entire interior.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Y = np.zeros((2,100,100))
        Y[:,0:50,:] = 1

        Z = np.zeros(Y.shape)
        for idx,pct in emlib.stratified_interior_pixel_generator(Y,b,30):
            Z[idx[:,0],idx[:,1],idx[:,2]] += 1

        self.assertTrue(np.all(Z[:,b:-b,b:-b]==1))
        Z[:,b:-b,b:-b] = 0
        self.assertTrue(np.all(Z==0))

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For a random input, should see a 50/50 split of class
        # labels, but not necessarily hit the entire interior.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Y = np.random.rand(2,100,100) > 0.5
        nOne=0; nZero=0;
        for idx,pct in emlib.stratified_interior_pixel_generator(Y,b,30):
            slices = idx[:,0];  rows = idx[:,1];  cols = idx[:,2]
            nOne += np.sum(Y[slices,rows,cols] == 1)  
            nZero += np.sum(Y[slices,rows,cols] == 0)
        self.assertTrue(nOne == nZero) 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For an input tensor with "no-ops", the sampler should only
        # return pixels with a positive or negative label.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Y = np.zeros((2,100,100))
        Y[:,0:20,0:20] = 1      
        Y[:,50:70,50:70] = -1
        Z = np.zeros(Y.shape)
        nPos=0; nNeg=0; nTotal=0;
        for idx,pct in emlib.stratified_interior_pixel_generator(Y,0,10,omitLabels=[0]):
            slices = idx[:,0];  rows = idx[:,1];  cols = idx[:,2]
            Z[slices,rows,cols] = Y[slices,rows,cols]
            nPos += np.sum(Y[slices,rows,cols] == 1)
            nNeg += np.sum(Y[slices,rows,cols] == -1)
            nTotal += len(slices)
            
        self.assertTrue(nPos == 20*20*2);
        self.assertTrue(nNeg == 20*20*2);
        self.assertTrue(nTotal == 20*20*2*2);
        self.assertTrue(np.all(Y == Z))




if __name__ == "__main__":
    unittest.main()


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
