#!/usr/bin/env python3

import os,torch
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

ifile = 'o01.class_prediction_results.top10.csv'
ohead = 'o04.confusion_matrix'

df = pd.read_csv(ifile)

# Normalize class names to be case-insensitive
df["true"] = df["Class"].astype(str).str.strip().str.lower().str.replace("_","-",regex=False)
df["pred"] = df["#1"].astype(str).str.strip().str.lower().str.replace("_","-",regex=False)

classes = sorted(df["true"].dropna().unique())

# Counts confusion matrix
cm = pd.crosstab(df["true"],df["pred"])
cm = cm.reindex(index=classes,columns=classes,fill_value=0)

# Row-normalized (per-true-class %)
cm_row_norm = cm.div(cm.sum(axis=1).replace(0,np.nan),axis=0)

# Output
cm.to_csv(ohead+"_counts.csv")
cm_row_norm.to_csv(ohead+"_row_normalized.csv")

#---end---

