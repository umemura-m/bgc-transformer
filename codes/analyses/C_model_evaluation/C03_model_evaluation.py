#!/usr/bin/env python3

import os,json,math,random, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List,Dict,Tuple
from sklearn.preprocessing import StandardScaler,LabelEncoder,normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE,trustworthiness

# %% Paths (edit these)
head = "bacfun"
titles = ["bacfun"]  # name list as long as you want
CSV_MIBIG = "o01.mibig_bgcs.pfamid.csv"  # cluster_id,label,domain_tokens
EMB_IDXp = "o02.embeddings_index.deepbgc.csv"
EMB_IDXe = "o02.embeddings_index.csv"
Xp = np.load("./emb/o02.emb_deepbgc.npy")

Xe = {}; Ls = ["L1","L2","L3"]
for t in titles:
    Xe.setdefault(t,{})
    for l in Ls:
        Xe[t][l] = np.load(f"./emb/o02.emb_{l}.{t}.npy")

# %% Load tables
df = pd.read_csv(CSV_MIBIG)  # columns: cluster_id,label,domain_tokens
for col in ("cluster_id","label","domain_tokens"):
    if col not in df.columns:
        raise SystemExit(f"Missing required column '{col}' in {CSV_MIBIG}")
df["label"] = df["label"].astype(str).str.strip().str.lower()

idxp = pd.read_csv(EMB_IDXp)[["cluster_id"]].copy()
idxp["_row"] = np.arange(len(idxp))
dfp = (df.merge(idxp,on="cluster_id",how="inner")
        .sort_values("_row")
        .drop_duplicates(subset=["cluster_id"])
        .reset_index(drop=True))
print(f"Loaded {len(dfp)} BGC rows for deepbgc")

idxe = pd.read_csv(EMB_IDXe)[["cluster_id"]].copy()
idxe["_row"] = np.arange(len(idxe))
dfe = (df.merge(idxe,on="cluster_id",how="inner")
        .sort_values("_row")
        .drop_duplicates(subset=["cluster_id"])
        .reset_index(drop=True))
print(f"Loaded {len(dfe)} BGC rows for ours")

mask_ok = ~np.isnan(Xp).any(axis=1)
n_drop = (~mask_ok).sum()
if n_drop:
    print(f"Dropping {n_drop} BGCs with no pfam2vec coverage")
    dfp = dfp.loc[mask_ok].reset_index(drop=True)
    Xp = Xp[mask_ok]

# %% Prepare labels
le = LabelEncoder()
yp = le.fit_transform(dfp["label"].values)
y1 = dfp["label"].values
ye = le.fit_transform(dfe["label"].values)
y2 = dfe["label"].values
classes = list(le.classes_)
print(f"{len(classes)} classes: {classes}")


#---linear-probe---
def linear_probe_cv(X:np.ndarray,y:np.ndarray,C=1.0,n_splits=5,seed=42,solver="lbfgs",max_iter=10000,tol=1e-3) -> dict:
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    f1_macros,accs = [],[]
    # standardize inside CV to avoid leakage
    for tr,te in skf.split(X,y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=5000,C=C,multi_class="auto",n_jobs=8)
        clf.fit(Xtr,y[tr])
        yhat = clf.predict(Xte)
        f1_macros.append(f1_score(y[te],yhat,average="macro"))
        accs.append(accuracy_score(y[te],yhat))
    return {
        "macroF1_mean": float(np.mean(f1_macros)),
        "macroF1_std": float(np.std(f1_macros,ddof=1)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs,ddof=1)),
        "folds": n_splits
    }

def report_linear_probe(name,X,y):
    if X is None:
        print(f"[{name}] skipped (X is None)")
        return None
    out = linear_probe_cv(X,y)
    print(f"[{name}] Linear probe (LogReg, {out['folds']}×CV): "
          f"macro-F1={out['macroF1_mean']:.3f}±{out['macroF1_std']:.3f}, "
          f"acc={out['acc_mean']:.3f}±{out['acc_std']:.3f}")
    return out

res_lp = {}
res_lp["pfam"] = report_linear_probe("deepbgc",Xp,yp)
for t in titles:
    for l in Ls:
        name = f"{t}_{l}"
        res_lp[name] = report_linear_probe(name,Xe[t][l],ye)


#---k-NN label purity (model-free)
def knn_purity(X: np.ndarray, y: np.ndarray, k=10, metric="cosine") -> float:
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
    dists,idxs = nbrs.kneighbors(X)
    # exclude self at column 0
    idxs = idxs[:,1:]
    same = (y[idxs] == y[:,None]).mean()
    return float(same)

def report_knn_purity(name,X,y,ks=(5,10,20)):
    if X is None:
        print(f"[{name}] k-NN purity skipped (X is None)")
        return {}
    out = {}
    # normalize to unit vectors for cosine
    Xn = normalize(X, axis=1)
    for k in ks:
        out[f"purity@{k}"] = knn_purity(Xn, y, k=k, metric="cosine")
    print(f"[{name}] k-NN purity: " + ", ".join(f"@{k}={out[f'purity@{k}']:.3f}" for k in ks))
    return out

res_knn = {}
res_knn["pfam"] = report_knn_purity("deepbgc",Xp,yp)
for t in titles:
    for l in Ls:
        name = f"{t}_{l}"
        res_knn[name] = report_knn_purity(name,Xe[t][l],ye)


#---Retrieval@k and mAP (label-aware)
def retrieval_metrics(X: np.ndarray, y: np.ndarray, ks=(1,5,10), metric="cosine") -> dict:
    Xn = normalize(X, axis=1)
    # pairwise cosine distance -> similarity
    D = pairwise_distances(Xn, Xn, metric=metric)   # shape (N,N)
    if metric == "cosine":
        S = 1.0 - D
    else:
        S = -D
    np.fill_diagonal(S, -np.inf)  # exclude self
    order = np.argsort(-S, axis=1)  # descending similarity
    y_rep = np.repeat(y[:,None], y.shape[0], axis=1)
    rel = (y_rep == y[order])

    out = {}
    N = X.shape[0]

    # Recall@k
    for k in ks:
        hits = rel[:,:k].any(axis=1).astype(float) if k==1 else rel[:,:k].sum(axis=1) / np.maximum(1, (y[:,None]==y[:,None]).sum(axis=1)-1)
        # For multi-class single-label, simpler: mean of at least one hit? But better: fraction of relevant retrieved per query:
        # To be robust, compute true relevant count per query (excluding self):
        n_rel = np.array([(y==yy).sum()-1 for yy in y], dtype=float)
        # recall@k per query:
        rec_k = np.array([rel[i,:k].sum()/n_rel[i] if n_rel[i]>0 else 0.0 for i in range(N)])
        out[f"recall@{k}"] = float(rec_k.mean())

    # mAP
    APs = []
    for i in range(N):
        r = rel[i]  # boolean array over ranks
        # precision at each hit
        ranks = np.where(r)[0]
        if ranks.size == 0:
            APs.append(0.0)
            continue
        precs = [(r[:(rnk+1)].sum()/(rnk+1)) for rnk in ranks]
        APs.append(float(np.mean(precs)))
    out["mAP"] = float(np.mean(APs))
    return out

def report_retrieval(name,X,y,ks=(1,5,10)):
    if X is None:
        print(f"[{name}] retrieval skipped (X is None)")
        return {}
    out = retrieval_metrics(X, y, ks=ks)
    print(f"[{name}] Retrieval: " + ", ".join([*(f"R@{k}={out[f'recall@{k}']:.3f}" for k in ks), f"mAP={out['mAP']:.3f}"]))
    return out

res_ret = {}
res_ret["pfam"] = report_retrieval("deepbgc",Xp,yp)
for t in titles:
    for l in Ls:
        name = f"{t}_{l}"
        res_ret[name] = report_retrieval(name,Xe[t][l],ye)


#---Unsupervised cluster structure (silhouette,CH,DB)
def cluster_quality(name,X,y):
    if X is None:
        print(f"[{name}] cluster metrics skipped (X is None)")
        return {}
    # Standardize helps distance-based metrics
    Xz = StandardScaler().fit_transform(X)
    out = {
        "silhouette": float(silhouette_score(Xz,y,metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(Xz,y)),
        "davies_bouldin": float(davies_bouldin_score(Xz,y)),
    }
    print(f"[{name}] Cluster metrics: sil={out['silhouette']:.3f}, CH={out['calinski_harabasz']:.1f}, DB={out['davies_bouldin']:.3f}")
    return out

res_clu = {}
res_clu["pfam"] = cluster_quality("deepbgc",Xp,yp)
for t in titles:
    for l in Ls:
        name = f"{t}_{l}"
        res_clu[name] = cluster_quality(name,Xe[t][l],ye)


#---t-SNE + trustworthiness
def tsne_save(name,X,labels,idx,perplexity=30,seed=42):
    lab = labels
    ids = idx.iloc[:,0]

    tsne = TSNE(n_components=2,perplexity=perplexity,learning_rate=200,init="pca",random_state=seed)
    Z = tsne.fit_transform(X)

    tw = trustworthiness(X,Z,n_neighbors=10)
    print(f"[{name}] t-SNE trustworthiness (k=10): {tw:.3f}")

    pd.DataFrame({"cluster_id":ids,"label":lab,"tsne1":Z[:,0],"tsne2":Z[:,1]}).to_csv(f"o03.tsne_{name}.csv",index=False)

# Draw pfam + your best layer
tsne_save("deepbgc",Xp,y1,idxp)
for t in titles:
    for l in Ls:
        name = f"{t}_{l}"
        tsne_save(name,Xe[t][l],y2,idxe)


#---Dump a compact comparison table
def collect_row(tag,lp,knn,ret,clu):
    row = {"model": tag}
    if lp:  row.update({ "lp_macroF1": lp["macroF1_mean"], "lp_acc": lp["acc_mean"] })
    if knn: row.update(knn)
    if ret:
        for k in (1,5,10):
            if f"recall@{k}" in ret: row[f"R@{k}"] = ret[f"recall@{k}"]
        row["mAP"] = ret.get("mAP", np.nan)
    if clu:
        row.update({
            "silhouette": clu["silhouette"],
            "CH": clu["calinski_harabasz"],
            "DB": clu["davies_bouldin"]
        })
    return row

summary = []
summary.append(collect_row("deepbgc",res_lp["pfam"],res_knn["pfam"],res_ret["pfam"],res_clu["pfam"]))
for t in titles:
    for l in Ls:
        name = f"{t}_{l}"
        summary.append(collect_row(name,res_lp.get(name),res_knn.get(name),res_ret.get(name),res_clu.get(name)))

tab = pd.DataFrame(summary)
with pd.option_context("display.max_columns",None):
    print(tab.round(3))
tab.to_csv(f"o03.model_evaluation_summary.{head}.csv",index=False)
print("Wrote comparison_summary.csv")

#---end---

