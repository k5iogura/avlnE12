#!/bin/bash

jupyter-nbconvert --to python train.ipynb
python3 train.py
