#!/bin/bash 

wget https://www.dropbox.com/s/be6jmue7zpfgb89/model_14.pkl?dl=1 -O 'cnn_model.pkl'

python3 hw3_test.py $1 $2 cnn_model.pkl