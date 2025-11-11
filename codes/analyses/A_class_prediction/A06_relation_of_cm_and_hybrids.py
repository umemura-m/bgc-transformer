#!/usr/bin/env python3

import csv, pandas as pd, numpy as np

# Confusion matrix (tab-separated, row = true class)
cm = pd.read_csv("o04.confusion_matrix_row_normalized.csv")
# Hybrid list
hy = pd.read_csv("o05.class_prediction_hybrid_stats.csv")

# Melt confusion matrix into long format (TRUE,PRED,VALUE)
true = cm.columns[0].strip()
cm = cm.rename(columns=str.strip)
cm[true] = cm[true].astype(str).str.strip()

# Set true class as index; ensure numeric values
cm = cm.set_index(true).rename_axis("true").apply(pd.to_numeric,errors="coerce")

# -> rows: true class, cols: predicted class
cm_long = cm.reset_index().melt(id_vars="true",var_name="pred",value_name="value")

# Select pairs with confusion > 0.3 but excluding diagonal (same class)
high_conf_pairs = cm_long.query("value > 0.3 and true != pred")
#high_conf_pairs.to_csv("o06.high_conf_pairs.csv")
print("High confusion pairs:\n#",high_conf_pairs)

# Ensure label normalization
for col in ["true","pred"]:
    cm_long[col] = cm_long[col].astype(str).str.strip()

for col in ["base","true","pred"]:
    hy[col] = hy[col].astype(str).str.strip()

# Wrong predictions among hybrids that are in high-conf set
hc_set = set(zip(high_conf_pairs["true"],high_conf_pairs["pred"]))
hy["is_wrong"] = hy["true"] != hy["pred"]
n_hy_wrong = len(hy["is_wrong"])
hy["pair"] = list(zip(hy["true"],hy["pred"]))
hy_wrong_hc = hy[hy["is_wrong"] & hy["pair"].isin(hc_set)].copy()

# Reverse confusion within the same base (pred->true exists)
rev = hy.rename(columns={"true":"true_rev","pred":"pred_rev","base":"base_rev"})
rev = rev[["base_rev","true_rev","pred_rev"]]

# Match rows where (true_rev, pred_rev) == (row.pred, row.true) and same base, different row
a_merge = (hy_wrong_hc.merge(rev, left_on=["base","pred","true"],right_on=["base_rev","true_rev","pred_rev"],how="left",indicator=True))
hy_wrong_hc["has_reverse_within_base"] = a_merge["_merge"].eq("both")
count_A = hy_wrong_hc["has_reverse_within_base"].sum()

# Class 'pred' appears as a true class within the same base
# For each base, collect set of true classes
true_by_base = hy.groupby("base")["true"].apply(set)
hy_wrong_hc["pred_true_within_base"] = hy_wrong_hc.apply(
    lambda r: r["pred"] in true_by_base.get(r["base"], set()),
    axis=1
)

count_B = hy_wrong_hc["pred_true_within_base"].sum()

# Outputs
n_total_wrong_hc = len(hy_wrong_hc)
print(f"Wrong hybrid predictions / high-confusion pairs: {n_total_wrong_hc} / {n_hy_wrong}")
print(f"  A) With reverse confusion present within same base: {count_A}")
print(f"  B) With 'pred' present as true within same base:   {count_B}")

# If you also want the rows:
hy_wrong_hc.to_csv("o06.wrong_highconf_hybrids.csv", index=False)

#---end---

