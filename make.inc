# Common make macros.
# 
# mjp Feb, 2016


# Assuming you are using Theano as the backend, you can control
# the gpu id in this manner (helpful if running multiple instances).
GPU=2
THEANO_FLAGS='floatX=float32,device=gpu$(GPU)'

PY=PYTHONPATH=./src:./src/thirdparty THEANO_FLAGS=$(THEANO_FLAGS) python
