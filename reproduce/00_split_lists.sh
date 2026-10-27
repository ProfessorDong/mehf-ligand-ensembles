#!/bin/bash
# Frozen val / test example lists (pure function of the corpus and the split seed).
source "$(dirname "$0")/common.sh"
$PY scripts/make_split_example_lists.py
