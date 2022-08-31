#!/usr/bin/env sh

python experiment.py --file-name=e2.csv --domains-path=../../data/domains --ao=MiCRONegotiator --ao=Atlas3 --reserved0=0.0 --reserved1=0.1 --reserved1=0.5 --reserved1=0.9 $@
