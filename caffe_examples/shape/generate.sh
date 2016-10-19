#!/usr/bin/env bash

rm -r shape_lmdb_test
rm -r shape_lmdb_train

../../bin/generate_random_shape_training_data --backend=lmdb --split=1 --shuffle=true --balance=true shape_lmdb