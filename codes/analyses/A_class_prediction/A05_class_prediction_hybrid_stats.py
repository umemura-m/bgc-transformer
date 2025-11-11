#!/usr/bin/env python3

import os,torch,argparse,torch
import pandas as pd, numpy as np
from tqdm import tqdm

ifile = 'o01.class_prediction_results.top10.csv'
ofile = 'o05.class_prediction_hybrid_stats.csv'

df = pd.read_csv(ifile)

# keep rows where ID contains "gbk_", which is hybrid
#mask = df["ID"].astype("string").str.contains("gbk_",case=False,na=False,regex=False)
#df = df.loc[mask].copy()

s = df["ID"].astype("string")
mask_hybrid = s.str.contains("gbk_",case=False,na=False,regex=False)
stems = (s[mask_hybrid]
         .str.replace(r'(?i)^(.*?gbk)_.*$',r'\1',regex=True)
         .str.lower().unique())
mask_stem = s.str.lower().isin(stems)
mask = mask_hybrid | mask_stem
df = df.loc[mask].copy()
nstem = sum(mask_stem)

# Normalize class names to be case-insensitive
df["true"] = df["Class"].astype(str).str.strip().str.lower().str.replace("_","-",regex=False)
df["pred"] = df["#1"].astype(str).str.strip().str.lower().str.replace("_","-",regex=False)

# Group key: strip a trailing ".gbk_\d" from ID
#df["base"] = df["ID"].astype(str).str.replace(r"\.gbk_\d+$","", regex=True)
df["base"] = df["ID"].astype(str).str.replace(r"\.gbk(?:_\d+)?$","", regex=True)

classes_true = sorted(df["true"].unique())
bases = sorted(df["base"].unique())

# For each base, collect the set of predictions in that group
pred_lookup = df.groupby("base")["pred"].agg(lambda s: set(s.dropna()))

# Per-row: does this row's TRUE appear in any PRED within the same group?
df["hit_as_entry"] = df["true"].eq(df["pred"])
df["hit_in_group"] = [t in pred_lookup.get(b,set()) for t,b in zip(df["true"],df["base"])]
df["hit_as_group"] = df.groupby("base")["hit_in_group"].transform("any")

# Per-group summary: True if any row in the group hits
group_hit = df.groupby("base")["hit_as_group"].any()
n_all_bases = int(group_hit.size)
n_true_bases = int(group_hit.sum())
ptrue = n_true_bases / n_all_bases

# output
cols_keep = ["base","ID","true","pred","hit_as_entry","hit_in_group","hit_as_group"]
cols_sort = ["base","hit_in_group","ID"]
ascending = [True,True,True]
out = df.sort_values(by=cols_sort,ascending=ascending,na_position="last")[cols_keep].reset_index(drop=True)

out.to_csv(ofile,index=False)
print(f"% of hybrid BGCs hit as BGC: {ptrue:.1%} ({n_true_bases}/{n_all_bases})")
print(f"#stem: {nstem}")
#print(f"Result was saved in {ofile}.")

#---end---

