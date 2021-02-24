#!/bin/bash
# installation steps for FEAT on LPC. 
# set our conda environment
conda env create -f environment.yml
conda activate feat-env
#clone feat
git clone https://github.com/lacava/feat/
cd feat
#checkout version 0.4.2
git checkout d8a13d241b782fef2d53b49e045be266d7922b71
#install feat
export SHOGUN_LIB=$CONDA_PREFIX/lib/
export SHOGUN_DIR=$CONDA_PREFIX/include/
export EIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3/
./configure lpc
./install lpc y

