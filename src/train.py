"""
"""

__author__ = "Mike Pekala"
__copyright__ = "Copyright 2016, JHU/APL"
__license__ = "Apache 2.0"


from keras.optimizers import SGD

import models as emm



if __name__ == "__main__":
    model = emm.ciresan_n3()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    #model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
