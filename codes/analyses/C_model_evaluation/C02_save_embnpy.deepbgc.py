#!/usr/bin/env python3

import re,ast, pandas as pd, numpy as np
from pathlib import Path
from typing import Dict,Union

mfile = "o01.mibig_bgcs.pfamid.csv"
pfile = "../../../data/pfam2vec.csv"

df_pfam = pd.read_csv(mfile,converters={"tokens": lambda s: ast.literal_eval(s) if isinstance(s,str) else s})

# Load pfam2vec.csv dict -> { "PFxxxxx": np.array(100,) }
PFAM_RE = re.compile(r"^PF\d{5}$")  # keep only real Pfam accessions

def load_pfam2vec_csv(path:Union[str,Path]) -> Dict[str,np.ndarray]:
    df = pd.read_csv(path)
    id_col = "pfam_id"
    # Keep only numeric 0..99 columns (some files may include extra metadata cols)
    num_cols = [c for c in df.columns if str(c).isdigit()]
    if len(num_cols) != 100:
        # if the columns are strings "0".."99" we’ve got them; otherwise try to coerce
        # This also handles cases where they're ints 0..99
        try:
            num_cols = sorted(num_cols, key=lambda x: int(x))
        except Exception:
            raise SystemExit(f"Expected 100 numeric columns 0..99, found {len(num_cols)}: {num_cols[:10]}")

    # Filter valid pfam ids
    df[id_col] = df[id_col].astype(str)
    df = df[df[id_col].apply(lambda x: bool(PFAM_RE.match(x)))].copy()

    # Build dict
    vecs = df[num_cols].to_numpy(dtype=np.float32)
    ids  = df[id_col].tolist()
    d = {pid:vecs[i] for i,pid in enumerate(ids)}
    print(f"Loaded pfam2vec: {len(d)} vectors, dim={vecs.shape[1]}")
    return d

pfam2vec = load_pfam2vec_csv(pfile)

#---vectorize---
# Compute the DeepBGC proxy vector for each row in df_pfam
def bgc_vec_from_accs(accs):
    vs = [pfam2vec[a] for a in accs if a in pfam2vec]
    if not vs: return None
    return np.mean(vs,axis=0,dtype=np.float32) if vs else None

# clusters with one token will be skipped
kidx,vecs,skipped = [],[],[]
for i,accs in enumerate(df_pfam["tokens"]):
    v = bgc_vec_from_accs(accs)
    if v is None or v.shape != (100,):
        skipped.append(i)
        continue
    kidx.append(i)
    vecs.append(v)

df_index = df_pfam.iloc[kidx].copy()
df_index["n_tokens"] = df_index["tokens"].apply(len)
df_index = df_index[["cluster_id","label","n_tokens"]]
df_index.to_csv("o02.embeddings_index.deepbgc.csv",index=False)

X_pfam = np.vstack(vecs).astype(np.float32)
np.save("./emb/o02.emb_deepbgc.npy",X_pfam)
print(f"X_pfam shape = {X_pfam.shape}  (skipped {len(skipped)} empty rows)")

#---end---

