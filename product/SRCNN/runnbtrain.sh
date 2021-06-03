#!/bin/bash

jupyter-nbconvert --to python train.ipynb
echo edit args --cuda option!
echo run python3 train.py
