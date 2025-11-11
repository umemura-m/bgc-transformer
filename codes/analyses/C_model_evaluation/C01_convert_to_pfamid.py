#!/usr/bin/env python3

import gzip,re, pandas as pd, numpy as np

# Change paths in your environment
pdat  = "/your/interproscan/data/pfam/35.0/pfam_a.dat"
mfile = "../../../data/MIBiG_domain_list.txt"
o1file = "o01.mibig_bgcs.dname.csv"
o2file = "o01.mibig_bgcs.pfamid.csv"

# Pfam Doname_name <-> Pfam ID converter
pfam_name2acc = {}
with open(pdat,"rt") as f:
    acc = name = None
    for line in f:
        if line.startswith("#=GF ID   "):
            name = line.split()[2]               # Pfam ID (short name)
        elif line.startswith("#=GF AC   "):
            acc = line.split()[2].split(".")[0]  # PFxxxxx (drop version)
        elif line.startswith("//"):
            if acc and name:
                pfam_name2acc[name.lower()] = acc
            acc = name = None

# generate mibig_bgc.pfamid.csv
row1,row2 = [],[]
with open(mfile,"rt") as f:
    for l in f:
        x = l.rstrip().split(maxsplit=2)
        cid,label,tokens = x[0],x[1],x[2]
        toks = tokens.split()
        tok1 = [t for t in toks if t != "[SEP]"]
        tok2 = [pfam_name2acc[t.lower()] for t in toks if t.lower() in pfam_name2acc]
        row1.append({
            "cluster_id": cid,
            "label": label,
            "domain_tokens": " ".join(tok1),
            "tokens": tok1,
        })
        row2.append({
            "cluster_id": cid,
            "label": label,
            "domain_tokens": " ".join(tok2),
            "tokens": tok2,
        })

df1 = pd.DataFrame(row1).drop_duplicates(subset=["cluster_id"]).reset_index(drop=True)
df2 = pd.DataFrame(row2).drop_duplicates(subset=["cluster_id"]).reset_index(drop=True)
df1.to_csv(o1file,index=False)
df2.to_csv(o2file,index=False)
print(f"Wrote {o1file} with {len(df1)} rows")
print(f"Wrote {o2file} with {len(df2)} rows")

