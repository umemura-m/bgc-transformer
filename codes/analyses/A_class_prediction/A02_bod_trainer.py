#!/usr/bin/env python3

# Train bag-of-domains baselines and export predictions CSV compatible with your evaluator.
# Train bag-of-domains baselines from lines like:
#   <ID>: [CLS] <class> [CLS] <tok1> [SEP] <tok2 tok2b> [SEP] <tok3> ...

import sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from scipy import sparse

train_txt = "../../../data/antismash_bac_bgcs_class_trainset.txt"
test_txt = "../../../data/antismash_bac_bgcs_class_testset.txt"
min_df = 5
topk = 10

def norm_label(s: str) -> str:
    return str(s).strip().lower().replace("_","-")

def parse_txt(path: str) -> pd.DataFrame:
    """
    Parse a text file where each non-empty line is:
      ID: [CLS] <class possibly multi-token> [CLS] <domains separated by [SEP] or spaces>
    Returns DataFrame with columns: ID, Class, domain_tokens (space-separated domain tokens).
    """
    ids,labels,domains = [],[],[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            line = ln.strip()
            if not line or line.startswith("#"): continue
            if ":" not in line:
                raise ValueError(f"Missing ':' in line: {line[:120]}")
            left,right = line.split(":",1)
            cid = left.strip()
            tokens = right.strip().split()

            # find [CLS] markers
            cls_idx = [i for i,t in enumerate(tokens) if t == "[CLS]"]
            if len(cls_idx) < 2:
                raise ValueError(f"Expected two [CLS] markers in line for ID={cid}")

            # class tokens are between first and second [CLS]
            c_from = cls_idx[0] + 1
            c_to   = cls_idx[1]
            class_tokens = tokens[c_from:c_to]
            if not class_tokens:
                raise ValueError(f"No class token(s) found for ID={cid}")
            label = " ".join(class_tokens)

            # domain tokens after second [CLS]; drop [SEP]/[CLS]
            raw_dom = [t for t in tokens[cls_idx[1]+1:] if t not in ("[SEP]","[CLS]")]
            # join by space (CountVectorizer token_pattern r"[^ ]+" will split on spaces)
            dom_str = " ".join(raw_dom).strip()

            ids.append(cid)
            labels.append(norm_label(label))
            domains.append(dom_str)
    return pd.DataFrame({"ID": ids, "Class": labels, "domain_tokens": domains})

def make_X(vec: CountVectorizer, series_tokens: pd.Series, fit: bool):
    return vec.fit_transform(series_tokens) if fit else vec.transform(series_tokens)

def train_and_predict(model_name:str,Xtr,ytr,Xte):
    """
    Train baseline and return top-k labels and probabilities for Xte.
    """
    if model_name == "majority":
        clf = DummyClassifier(strategy="most_frequent")
    elif model_name == "stump":
        clf = DecisionTreeClassifier(max_depth=1,random_state=42)
    elif model_name == "tree3":
        clf = DecisionTreeClassifier(max_depth=3,random_state=42)
    else:
        raise ValueError("model_name must be one of: majority, stump, tree3")

    clf.fit(Xtr,ytr)

    # class probabilities (or one-hot if not supported)
    if hasattr(clf,"predict_proba"):
        proba = clf.predict_proba(Xte)  # shape (N,n_classes_trained)
        cls_names = list(clf.classes_)
    else:
        yhat = clf.predict(Xte)
        cls_names = list(np.unique(ytr))
        name_to_idx = {c:i for i,c in enumerate(cls_names)}
        proba = np.zeros((Xte.shape[0],len(cls_names)),dtype=float)
        for i,c in enumerate(yhat):
            proba[i,name_to_idx[c]] = 1.0

    # Top-k = 5
    k = min(topk,proba.shape[1])
    order = np.argsort(-proba,axis=1)
    topk_idx = order[:,:k]
    topk_probs = np.take_along_axis(proba,topk_idx,axis=1)
    topk_labels = np.array([[cls_names[j] for j in row] for row in topk_idx], dtype=object)

    return topk_labels,topk_probs

def write_predictions_csv(path_out:str,ids:pd.Series,y_true:pd.Series,topk_labels,topk_probs):
    k = topk_labels.shape[1]
    data = {"ID":ids.tolist(),"Class":y_true.tolist()}
    for i in range(k):
        data[f"#%d"%(i+1)] = topk_labels[:, i]
    for i in range(k):
        data[f"prob#%d"%(i+1)] = np.round(topk_probs[:, i], 6)
    # pad to 5 if fewer classes
    for i in range(k,5):
        data[f"#%d"%(i+1)] = [""] * len(ids)
        data[f"prob#%d"%(i+1)] = [np.nan] * len(ids)
    pd.DataFrame(data).to_csv(path_out, index=False)

def count_tokens(s) -> int:
    if s is None: return 0
    s = str(s).strip()
    return 0 if not s else len(s.split())

def main():
    dtr = parse_txt(train_txt)
    dte = parse_txt(test_txt)

    # BoD features
    vec = CountVectorizer(token_pattern=r"[^ ]+",min_df=min_df)
    Xtr = make_X(vec,dtr["domain_tokens"],fit=True)
    Xte = make_X(vec,dte["domain_tokens"],fit=False)

    # save bag-of-domains embeddings as dense .npy
#   vocab_list = vec.get_feature_names_out()
#   np.save(f"o02.bod_vocab.npy",vocab_list)
#   np.save(f"o02.emb_bod.npy",Xtr.toarray().astype(np.float32))
#   sparse.save_npz("o02.emb_bod.npz",Xtr)
#   print(f"[ok] saved embeddings: o02.emb_bod.npy {Xtr.shape}")

    ytr = dtr["Class"].tolist()

    for name in ["majority","stump","tree3"]:
        topk_labels,topk_probs = train_and_predict(name,Xtr,ytr,Xte)
        out_csv = f"o02.bod_{name}.csv"
        write_predictions_csv(out_csv,dte["ID"],dte["Class"],topk_labels,topk_probs)
        print(f"[ok] wrote {out_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

#---end---

