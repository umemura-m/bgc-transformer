#!/usr/bin/env python3

import numpy as np, pandas as pd
import csv,torch,random,argparse,re
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from typing import Dict,Iterable,List

ap = argparse.ArgumentParser(description="Evaluate statistics per predicted domain")
ap.add_argument("--ifile",type=str,required=True,help="Predicted result for evaluation")
ap.add_argument("--efile",type=str,required=True,help="File for eval")
ap.add_argument("--tdir",type=str,required=True,help="Directory for training index file")
args = ap.parse_args()

name = Path(args.ifile).name
m = re.match(r"^every_domain_prediction\.(.+)\.csv$",name)
title = m.group(1)

df = pd.read_csv(args.ifile)

# Normalization helper
def norm_token(t: str) -> str:
    return t.strip()

# Read training corpus
class DomainTokenCache:
    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, file_path: str) -> np.ndarray:
        arr = self._cache.get(file_path)
        if arr is not None:
            return arr
        df_src = pd.read_csv(file_path, usecols=["Domain_Name"])
        arr = df_src["Domain_Name"].astype(str).map(norm_token).values
        self._cache[file_path] = arr
        return arr

def iter_tokens_for_index(index_df: pd.DataFrame, cache: DomainTokenCache) -> Iterable[str]:
    # assumes start_token is 0-based over rows of source CSV
    for file_path, grp in index_df.groupby("file_path", sort=False):
        arr = cache.get(file_path)
        starts = grp["start_token"].astype(int).values
        lens   = grp["length"].astype(int).values
        for s, L in zip(starts, lens):
            if s < 0 or L < 0 or s + L > len(arr):
                raise IndexError(f"Slice OOB for {file_path}: start={s}, len={L}, n={len(arr)}")
            yield from arr[s:s+L]

def train_token_counts(index_df: pd.DataFrame) -> pd.Series:
    cache = DomainTokenCache()
    c = Counter(iter_tokens_for_index(index_df, cache))
    s = pd.Series(c, dtype="int64").sort_values(ascending=False)
    s.index.name = "token"
    s.name = "train_count"
    return s

# Read evaluation corpus
SEP_PATTERN = re.compile(r"\s*\[SEP\]\s*")

def eval_lines_token_counts(lines: List[str]) -> pd.Series:
    tokens = []
    for line in lines:
        line = line.strip()
        if not line: continue
        # split first two fields (bgc_id, compound_class), remainder is domains
        try:
            _, _, remainder = line.split(maxsplit=2)
        except ValueError:
            continue
        sections = SEP_PATTERN.split(remainder)
        for sec in sections:
            toks = [norm_token(t) for t in sec.split() if t]
            tokens.extend(toks)
    s = pd.Series(tokens).value_counts()
    s.index.name = "token"
    s.name = "eval_count"
    return s.astype("int64")

# Statistics in the predicted result
def per_token_eval_stats(pred_df: pd.DataFrame) -> pd.DataFrame:
    # coerce numeric
    pred_df = pred_df.copy()
    pred_df["Original token"] = pred_df["Original token"].astype(str).map(norm_token)
    pred_df["True token rank"] = pd.to_numeric(pred_df["True token rank"], errors="coerce")
    pred_df["True token probability"] = pd.to_numeric(pred_df["True token probability"], errors="coerce")
    pred_df["Probability 1"] = pd.to_numeric(pred_df["Probability 1"], errors="coerce")

    g = pred_df.groupby("Original token",as_index=False).agg(
        avg_true_rank=("True token rank","mean"),
        med_true_rank=("True token rank","median"),
        std_true_rank=("True token rank","std"),
        mean_top1_prob=("Probability 1","mean"),
        n_rows=("Original token","size"),
    )
    g = g.rename(columns={"Original token":"token"})
    return g

# Bring everything together
def build_domain_summary_table(train_index_df: pd.DataFrame,
                               eval_prediction_df: pd.DataFrame,
                               eval_lines: List[str]) -> pd.DataFrame:
    train_counts = train_token_counts(train_index_df)
    eval_counts  = eval_lines_token_counts(eval_lines)
    eval_stats = per_token_eval_stats(eval_prediction_df)
    out = (eval_stats
           .merge(train_counts.to_frame(),on="token",how="outer")
           .merge(eval_counts.to_frame(), on="token",how="outer"))

    # fill and tidy
    out[["n_rows","train_count","eval_count"]] = out[["n_rows","train_count","eval_count"]].fillna(0).astype("int64")
    # avg/med ranks & mean_top1_prob can stay NaN if no eval stats for that token
    out = out.sort_values(["med_true_rank","avg_true_rank","std_true_rank","n_rows","train_count"],ascending=[True,True,True,False,False],na_position="last").reset_index(drop=True)
    cols = ["token","med_true_rank","avg_true_rank","std_true_rank","mean_top1_prob","train_count","eval_count","n_rows"]
    out = out[cols]
    return out

# Load and output
train_index_df = pd.read_csv(args.tdir+"/train_index.tsv",sep="\t")
eval_prediction_df = pd.read_csv(args.ifile)
with open(args.efile,"r",encoding="utf-8") as f:
    eval_lines = f.readlines()

summary = build_domain_summary_table(train_index_df,eval_prediction_df,eval_lines)
summary.to_csv("o03."+title+".domain_summary_table.csv", index=False)
print(summary.head(15))

#---end---

