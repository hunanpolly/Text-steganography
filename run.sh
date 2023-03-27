#!/bin/bash

python HC.py -dataset 'Twitter' -generate-num 100000 -idx-gpu 1
python HC.py -dataset 'IMDB' -generate-num 100000 -idx-gpu 1
python AC.py -dataset 'Twitter' -generate-num 100000 -idx-gpu 1
python AC.py -dataset 'IMDB' -generate-num 100000 -idx-gpu 1
python ADG.py -dataset 'IMDB' -generate-num 100000 -idx-gpu 1
python ADG.py -dataset 'Twitter' -generate-num 100000 -idx-gpu 1
