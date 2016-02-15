


ISBI=./data/ISBI2012


train-isbi :
	python ./src/train.py \
		--x-train $(ISBI)/train-volume.tif \
		--train-slices "range(0,27)" \
		--y-train $(ISBI)/train-labels.tif \
		--x-valid $(ISBI)/train-volume.tif \
		--y-valid $(ISBI)/train-labels.tif \
		--valid-slices "[27, 28, 29]"  \
		--out-dir ./Models/ISBI2012

