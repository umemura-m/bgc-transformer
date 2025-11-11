#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv,os,json,pickle,hashlib
from pathlib import Path
from typing import Dict,List,Tuple,Iterable
from functools import lru_cache

# I/O
def load_index_tsv(path: str) -> List[Tuple[str,int,int,int]]:
    """
    Returns rows of (file_path, start_token, length, chunk_index_in_file).
    """
    out = []
    with open(path,"r",encoding="utf-8") as f:
        r = csv.DictReader(f,delimiter="\t")
        for row in r:
            out.append((
                row["file_path"],
                int(row["start_token"]),
                int(row["length"]),
                int(row.get("chunk_index_in_file",0)),
            ))
    return out

@lru_cache(maxsize=4096)
def read_token_file_csv(path:str,token_col:str="Domain_Name") -> List[str]:
    toks:List[str] = []
    with open(path,"r",encoding="utf-8",newline="") as f:
        reader = csv.DictReader(f,delimiter=",")
        if token_col not in reader.fieldnames:
            raise KeyError(f"{token_col} not found in {path}. Columns: {reader.fieldnames}")
        for row in reader:
            t = (row.get(token_col) or "").strip()
            if t:
                toks.append(t)
    return toks

def md5_str(s:str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def shingles(tokens:List[str],k:int) -> List[str]:
    # contiguous k-grams (order-sensitive)
    if len(tokens) < k: return []
    return [" ".join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]

def unordered_pairs(tokens:List[str]) -> List[Tuple[str,str]]:
    # all unordered token pairs (order-agnostic; captures co-occurrence)
    n = len(tokens)
    out = []
    for i in range(n):
        for j in range(i+1, n):
            a,b = tokens[i],tokens[j]
            if a == b:  # optional: skip identical tokens
                continue
            if a < b:
                out.append((a,b))
            else:
                out.append((b,a))
    return out

# Build leakage indices from TRAIN shards
def build_leakage_index(
    train_index_tsv:str,
    token_col:str = "Domain_Name",
    W:int=5,        # half-window (window len = 2W+1)
    k:int=5,        # shingle size for near-dup
    out_pickle:str=None,
) -> Dict:
    """
    Scans all windows (±W) inside each TRAIN chunk and builds:
      - exact_ctx: set(md5(window_str))
      - seen_shingles: set of k-gram strings
      - seen_pairs: set of unordered (tok_i, tok_j) co-occurrence pairs
      - meta: parameters & counts
    """
    rows = load_index_tsv(train_index_tsv)
    exact_ctx = set()
    seen_shingles = set()
    seen_pairs = set()

    n_windows = 0
    for fp,start,length,_ in rows:
        toks = read_token_file_csv(fp,token_col=token_col)
        block = toks[start:start+length]
        if len(block) != length:
            raise ValueError(f"Slice length mismatch in {fp} at start={start} (want {length}, got {len(block)})")
        # center windows inside the block
        for i in range(W,len(block)-W):
            win = block[i-W:i+W+1]
            # L1
            exact_ctx.add(md5_str(" ".join(win)))
            # L2 (order-sensitive local k-grams)
            for g in shingles(win,k):
                seen_shingles.add(g)
            # L3 (order-agnostic co-occurrence)
            for a,b in unordered_pairs(win):
                seen_pairs.add((a,b))
            n_windows += 1

    idx = {
        "exact_ctx": exact_ctx,
        "seen_shingles": seen_shingles,
        "seen_pairs": seen_pairs,
        "meta": {
            "W": W,
            "k": k,
            "train_index_tsv": os.path.abspath(train_index_tsv),
            "n_train_chunks": len(rows),
            "n_windows": n_windows,
            "n_exact": len(exact_ctx),
            "n_shingles": len(seen_shingles),
            "n_pairs": len(seen_pairs),
            "token_col": token_col,
        }
    }
    if out_pickle:
        Path(out_pickle).parent.mkdir(parents=True,exist_ok=True)
        with open(out_pickle,"wb") as f:
            pickle.dump(idx,f,protocol=pickle.HIGHEST_PROTOCOL)
        # also drop a small json meta for quick inspection
        with open(os.path.splitext(out_pickle)[0] + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(idx["meta"], f, indent=2)
    return idx

def load_leakage_index(pkl_path:str) -> Dict:
    with open(pkl_path,"rb") as f:
        return pickle.load(f)

# Tag eval positions with L1/L2/L3
def tag_window(
    win_tokens:List[str],
    idx:Dict,
    l2_thresh:float=0.8,     # fraction of k-grams seen in train
    l3_thresh:float=0.7,     # fraction of unordered pairs seen in train
) -> str:
    W = idx["meta"]["W"]; k = idx["meta"]["k"]
    # L1
    if md5_str(" ".join(win_tokens)) in idx["exact_ctx"]:
        return "L1_exact"

    # L2 (near-dup by k-grams presence)
    g = shingles(win_tokens, k)
    if g:
        frac = sum(1 for x in g if x in idx["seen_shingles"]) / len(g)
        if frac >= l2_thresh:
            return "L2_near"

    # L3 (family/remote by unordered pairs presence)
    pairs = unordered_pairs(win_tokens)
    if pairs:
        frac = sum(1 for p in pairs if p in idx["seen_pairs"]) / len(pairs)
        if frac >= l3_thresh:
            return "L3_family"

    return "Unseen"

def tag_eval_index(
    eval_index_tsv:str,idx:Dict,
    token_col:str="Domain_Name",out_csv:str=None,
) -> List[Tuple[str,int,int,str]]:
    """
    Returns a list of (file_path, chunk_index_in_file, pos_in_chunk, tag),
    and optionally writes a CSV with those columns.
    """
    W = idx["meta"]["W"]
    rows = load_index_tsv(eval_index_tsv)
    tags = []
    for fp,start,length,chunk_idx in rows:
        toks = read_token_file_csv(fp,token_col=token_col)
        block = toks[start:start+length]
        if len(block) != length:
            raise ValueError(f"Slice length mismatch in {fp} at start={start}")
        for i in range(W,len(block)-W):
            win = block[i-W:i+W+1]
            lab = tag_window(win,idx)
            tags.append((fp,chunk_idx,i,lab))

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True,exist_ok=True)
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_path","chunk_index_in_file","pos_in_chunk","tag"])
            w.writerows(tags)
    return tags


def parse_mibig_line(line:str):
    """
    Input (like your prediction file):
      BGC0000001 <Type> TOK1 [SEP] TOK2 [SEP] TOK3 ...

    Returns:
      cluster : str
      tokens  : list[str]          # full token list starting at parts[2:], includes "[SEP]"
      segments: list[dict]         # each: {"seg_id", "tokens", "orig_pos", "seg_pos"}
        - tokens   : tokens inside this segment (no [SEP])
        - orig_pos : absolute indices into `tokens` (match "Original position" in pred CSV)
        - seg_pos  : 0..len(segment)-1 (position within the segment)
    """
    parts = line.strip().split()
    if len(parts) < 3:
        return None  # malformed

    cluster = parts[0]
    tokens  = parts[2:]  # mirrors your prediction script

    segments = []
    seg_id = 0
    cur_tokens,cur_orig = [],[]

    for i, t in enumerate(tokens):
        if t == "[SEP]":
            if cur_tokens:
                segments.append({
                    "seg_id": seg_id,
                    "tokens": cur_tokens[:],
                    "orig_pos": cur_orig[:],                          # <-- absolute positions saved
                    "seg_pos": list(range(len(cur_tokens))),
                })
                seg_id += 1
                cur_tokens,cur_orig = [],[]
            continue
        cur_tokens.append(t)
        cur_orig.append(i)  # <-- this i is the "Original position" used in your pred CSV

    if cur_tokens:
        segments.append({
            "seg_id": seg_id,
            "tokens": cur_tokens[:],
            "orig_pos": cur_orig[:],
            "seg_pos": list(range(len(cur_tokens))),
        })

    return cluster,tokens,segments




def tag_mibig_file(
    mibig_txt_path: str,
    idx: dict,                      # leakage index from build_leakage_index(...)
    out_csv: str = None,
    l2_thresh: float = 0.8,
    l3_thresh: float = 0.7,
):
    """
    Read a MIBiG lines file and tag each maskable token position.
    Outputs rows: Cluster, Original position, Token, tag

    NOTE:
      - 'Original position' is the absolute index into `parts[2:]` (same as in predictions).
      - Supports segments as 2-tuples (tokens, orig_pos) or dicts {"tokens","orig_pos",...}.
    """
    W = idx["meta"]["W"]
    rows_out = []

    with open(mibig_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parsed = parse_mibig_line(line)  # must return: (cluster, tokens, segments)
            if parsed is None:
                continue
            cluster, tokens, segments = parsed

            for seg in segments:
                # accept either tuple or dict segment format
                if isinstance(seg, dict):
                    seg_tokens = seg.get("tokens")
                    seg_map    = seg.get("orig_pos")
                else:
                    seg_tokens, seg_map = seg

                if seg_tokens is None or seg_map is None:
                    raise ValueError("Segment missing 'tokens' or 'orig_pos'")

                # need at least 2W+1 tokens to form a window
                if len(seg_tokens) < 2 * W + 1:
                    continue

                for c in range(W, len(seg_tokens) - W):
                    abs_i = int(seg_map[c])  # absolute index into full `tokens`

                    # sanity check: mapped token matches
                    if tokens[abs_i] != seg_tokens[c]:
                        raise AssertionError(
                            f"Mapping mismatch: tokens[{abs_i}]={tokens[abs_i]!r} "
                            f"!= seg_tokens[{c}]={seg_tokens[c]!r}"
                        )

                    win = seg_tokens[c - W : c + W + 1]
                    tag = tag_window(win, idx, l2_thresh=l2_thresh, l3_thresh=l3_thresh)

                    center_tok = seg_tokens[c]          # the token at this position
                    rows_out.append((cluster, abs_i, center_tok, tag))

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Cluster", "Original position", "Token", "tag"])
            w.writerows(rows_out)

    return rows_out




def tag_mibig_filek(
    mibig_txt_path: str,
    idx: dict,                      # leakage index from build_leakage_index(...)
    out_csv: str = None,
    l2_thresh: float = 0.8,
    l3_thresh: float = 0.7,
):
    """
    Read a MIBiG lines file and tag each maskable token position.
    Outputs rows: Cluster, Original position, tag
    NOTE: 'Original position' is the absolute index into parts[2:] (same as in predictions).
    """
    W = idx["meta"]["W"]
    rows_out = []

    with open(mibig_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parsed = parse_mibig_line(line)   # must return: (cluster, tokens, segments)
            if parsed is None:
                continue
            cluster, tokens, segments = parsed  # 'tokens' includes "[SEP]" if your parser keeps it

            for seg in segments:
                # Support both formats:
                #  - tuple: (seg_tokens, seg_map)
                #  - dict : {"tokens": ..., "orig_pos": ...}
                if isinstance(seg, dict):
                    seg_tokens = seg.get("tokens")
                    seg_map    = seg.get("orig_pos")
                else:
                    seg_tokens, seg_map = seg  # unpack tuple

                if seg_tokens is None or seg_map is None:
                    raise ValueError("Segment missing 'tokens' or 'orig_pos'")

                # need at least 2W+1 tokens to form a window
                if len(seg_tokens) < 2 * W + 1:
                    continue

                for c in range(W, len(seg_tokens) - W):
                    abs_i = int(seg_map[c])  # absolute index into full tokens (== 'Original position')

                    # sanity: mapped token should match the segment token
                    try:
                        assert tokens[abs_i] == seg_tokens[c]
                    except AssertionError:
                        raise AssertionError(
                            f"Mapping mismatch: tokens[{abs_i}]={tokens[abs_i]!r} "
                            f"!= seg_tokens[{c}]={seg_tokens[c]!r}"
                        )

                    # build window inside the segment
                    win = seg_tokens[c - W : c + W + 1]
                    tag = tag_window(win, idx, l2_thresh=l2_thresh, l3_thresh=l3_thresh)

                    rows_out.append((cluster, abs_i, tag))

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Cluster", "Original position", "tag"])
            w.writerows(rows_out)

    return rows_out

#---end---
