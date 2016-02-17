# Example showing how to use the provided codes to train/deploy on ISBI 2012.


#PY=ipython -i --
PY=python

ISBI=./data/ISBI2012
OUT_DIR=./Models/ISBI2012


train-isbi :
	$(PY) ./src/train.py \
		--x-train $(ISBI)/train-volume.tif \
		--train-slices "range(0,27)" \
		--y-train $(ISBI)/train-labels.tif \
		--x-valid $(ISBI)/train-volume.tif \
		--y-valid $(ISBI)/train-labels.tif \
		--valid-slices "[27, 28, 29]"  \
		--num-batches-per-epoch 1000 \
		--out-dir $(OUT_DIR)

test :
	PYTHONPATH=./src $(PY) ./tests/test_emlib.py


clean :
	\rm -rf $(OUT_DIR)
