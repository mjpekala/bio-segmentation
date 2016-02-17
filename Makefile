# Example of training/deploying CNN for ISBI 2012 data set.
# 
# mjp Feb, 2016

PY=python

ISBI=./data/ISBI2012
OUT_DIR=./Models/ISBI2012


# Pick one of TRAIN_QUICK or TRAIN_FULL depending on how much time and GPU
# you have available.
# 
TRAIN_QUICK=--num-batches-per-epoch 10000 --num-epochs 3
TRAIN_FULL=--num-epochs 30
TRAIN_FLAGS=$(TRAIN_QUICK)


#-------------------------------------------------------------------------------
train-isbi :
	$(PY) ./src/train.py \
		--x-train $(ISBI)/train-volume.tif \
		--train-slices "range(0,27)" \
		--y-train $(ISBI)/train-labels.tif \
		--x-valid $(ISBI)/train-volume.tif \
		--y-valid $(ISBI)/train-labels.tif \
		--valid-slices "[29,]"  \
		--out-dir $(OUT_DIR) \
		$(TRAIN_FLAGS)


test :
	PYTHONPATH=./src $(PY) ./tests/test_emlib.py


clean :
	\rm -rf $(OUT_DIR)
