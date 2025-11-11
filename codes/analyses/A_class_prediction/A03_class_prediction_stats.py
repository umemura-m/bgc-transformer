#!/usr/bin/env python3

import numpy as np, pandas as pd

#ifile = 'o01.BGC_class_prediction_results.top10.csv'
#ofile = 'o03.class_prediction_stats.transformer.csv'
#ifile = 'o02.bod_stump.csv'
#ofile = 'o03.class_prediction_stats.bod_stump.csv'
ifile = 'o02.bod_tree3.csv'
ofile = 'o03.class_prediction_stats.bod_tree3.csv'

df = pd.read_csv(ifile)

# Normalize class names (case-insensitive; "_" -> "-")
df["true"] = df["Class"].astype(str).str.strip().str.lower().str.replace("_","-",regex=False)
df["pred"] = df["#1"].astype(str).str.strip().str.lower().str.replace("_","-",regex=False)

classes_true = sorted(df["true"].unique())

# counts
support_true    = df["true"].value_counts().reindex(classes_true,fill_value=0)
predicted_count = df["pred"].value_counts().reindex(classes_true,fill_value=0)

# TP per true class
tp = (df.groupby("true")
        .apply(lambda g: (g["pred"] == g.name).sum())
        .reindex(classes_true, fill_value=0))

# metrics
recall    = (tp / support_true.replace(0, np.nan)).rename("TP/all_class_members")
precision = (tp / predicted_count.replace(0, np.nan)).rename("TP/all_predicted_as_class")

# per-class F1 (set NaN to 0 when precision/recall undefined)
f1 = (2 * precision * recall / (precision + recall)).fillna(0.0).rename("F1")

# table
out = (pd.concat([support_true.rename("count"),
                  tp.rename("TP"),
                  predicted_count.rename("predicted_count"),
                  recall,precision,f1], axis=1)
         .reset_index().rename(columns={"index":"class"}))

out = out.sort_values(["TP/all_class_members","count","class"],
                      ascending=[False,False,True]).reset_index(drop=True)

# aggregate scores (macro over classes present in y_true)
macro_f1    = f1.mean()
weighted_f1 = (f1 * (support_true / support_true.sum())).sum()
accuracy    = (df["true"] == df["pred"]).mean()  # micro-F1 for single-label multiclass
micro_f1    = accuracy

print(f"Macro-F1:   {macro_f1:.6f}")
print(f"Micro-F1:   {micro_f1:.6f}  (equals accuracy)")
print(f"Weighted-F1:{weighted_f1:.6f}")
print(f"Accuracy:   {accuracy:.6f}")

# save per-class table
out.to_csv(ofile,index=False)

# optional: write a small summary file
with open(ofile.replace(".csv",".scores.out"),"w",encoding="utf-8") as f:
    f.write(f"Macro-F1:   {macro_f1:.6f}\n")
    f.write(f"Micro-F1:   {micro_f1:.6f}\n")
    f.write(f"Weighted-F1:{weighted_f1:.6f}\n")
    f.write(f"Accuracy:   {accuracy:.6f}\n")

#---end---
