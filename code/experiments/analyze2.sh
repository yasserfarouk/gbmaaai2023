#!/usr/bin/env sh
python helpers/clean.py .
python eval4.py --files=e2.csv --output=e4everything $@
