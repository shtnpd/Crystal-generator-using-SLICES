#!/bin/bash

mkdir -p data/raw/sAlex

cd data/raw/sAlex
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz
tar -xzf train.tar.gz