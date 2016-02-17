# Example of training/deploying CNN for ISBI 2012 data set.
# 
# mjp Feb, 2016

PY=PYTHONPATH=./src:./src/thirdparty python

ISBI=./data/ISBI2012
OUT_DIR=./Models/ISBI2012


# Use TRAIN_QUICK to make sure training runs end-to-end; use TRAIN_FULL
# if you want a reasonable model.
# 
TRAIN_QUICK=--num-batches-per-epoch 100000 --num-epochs 3
TRAIN_FULL=--num-epochs 30
TRAIN_FLAGS=$(TRAIN_QUICK)


#-------------------------------------------------------------------------------
default :
	@echo ""
	@echo "Targets supported by this makefile:"
	@echo "   train-isbi  : trains a model using ISBI 2012 data set"
	@echo "   deploy-isbi : evaluate ISBI 2012 test data"
	@echo "   test        : runs unit tests"
	@echo "   clean       : deletes any previously trained models"
	@echo ""


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


deploy-isbi :
	$(PY) ./src/deploy.py \
		--x $(ISBI)/test-volume.tif \
		--slices "range(0,5)" \
		--weight-file $(OUT_DIR)/weights_epoch_002.h5 \
		--out-file $(OUT_DIR)/Yhat_test.npy


test :
	$(PY) ./tests/test_emlib.py


clean :
	\rm -rf $(OUT_DIR)
