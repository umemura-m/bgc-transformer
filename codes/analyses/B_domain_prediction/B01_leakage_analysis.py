#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Execute in each model directory

import csv,os,json,pickle,hashlib,argparse
from pathlib import Path
from typing import Dict,List,Tuple,Iterable
from functools import lru_cache
from leakage_index import build_leakage_index,tag_eval_index,load_leakage_index,tag_mibig_file

# argument setting
ap = argparse.ArgumentParser()
ap.add_argument("--target",type=str,required=True,help="Target name")
ap.add_argument("--idx_exist",dest="idx_exist",action="store_true")
ap.add_argument("--no_idx_exist", dest="idx_exist", action="store_false")
ap.set_defaults(idx_exist=False)
ap.add_argument("--input_file",type=str,required=True,help="MIBiG file for evaluation")
args = ap.parse_args()

train_tsv = f"./indices.{args.target}/train_index.tsv"

# Build or read leakage idx file
if args.idx_exist:
    idx = load_leakage_index(f"./leakage.{args.target}/train_leakage_idx.pkl")
else:
    idx = build_leakage_index(
        train_index_tsv=train_tsv,
        token_col="Domain_Name",
        W=5,         # window half-width (window size = 11)
        k=5,         # near-dup shingle length
        out_pickle=f"./leakage.{args.target}/train_leakage_idx.pkl"
    )

# Tag your MIBiG eval lines file
tags = tag_mibig_file(
    mibig_txt_path=args.input_file,
    idx=idx,
    out_csv=f"./leakage.{args.target}/mibig_eval_leakage_tags.csv"
)

print("meta:",idx["meta"])
print("example tags:",tags[:5])
print("Tagged rows:",len(tags)," example:",tags[:5])

#---end---
