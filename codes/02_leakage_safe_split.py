#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,os,sys,hashlib,random,csv
from collections import defaultdict,deque
from typing import List,Set,Tuple
import numpy as np

def read_domain_tokens_from_csv(path:str,token_col:str="Domain_Name") -> List[str]:
    """
    Read a CSV with a header and return the ordered list of domain tokens
    taken from the specified column (e.g., 'Domain_Name' or 'Domain_ID').
    """
    toks:List[str] = []
    with open(path,"r",encoding="utf-8",newline="") as f:
        reader = csv.DictReader(f,delimiter=",")
        if token_col not in reader.fieldnames:
            raise KeyError(f"{token_col} not found in {path}. Columns: {reader.fieldnames}")
        for row in reader:
            tok = (row.get(token_col) or "").strip()
            if tok:
                toks.append(tok)
    return toks

def iter_chunks_from_csv(path:str,block_size:int=512,token_col:str="Domain_Name"):
    """
    Yield consecutive NON-overlapping chunks (chunk_idx, List[str]) of domain tokens
    read from the CSV column 'token_col'.
    Tail shorter than block_size is discarded (matches your non-overlap behavior).
    """
    toks = read_domain_tokens_from_csv(path,token_col=token_col)
    n_full = len(toks) // block_size
    for ci in range(n_full):
        start = ci * block_size
        yield ci,toks[start:start + block_size]


#--- shingles + minhash + LSH ---
def shingles(tokens:List[str],n:int=6) -> Set[str]:
    if len(tokens) < n: return set()
    sep = "\x1f"
    return {sep.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def _h64(b:bytes) -> int:
    return int.from_bytes(hashlib.sha1(b).digest()[:8],"big",signed=False)

def minhash_signature(sh_set:Set[str],num_perm:int=128) -> Tuple[int, ...]:
    if not sh_set: return tuple([2**64-1]*num_perm)
    pre = [_h64(s.encode("utf-8")) for s in sh_set]
    out = []
    for i in range(num_perm):
        salt = i.to_bytes(4,"big")
        m = 2**64 - 1
        for h in pre:
            x = h ^ _h64(salt + h.to_bytes(8,"big"))
            if x < m: m = x
        out.append(m)
    return tuple(out)

def lsh_bands(signature:Tuple[int,...],bands:int=32):
    r = len(signature)//bands
    for b in range(bands):
        chunk = signature[b*r:(b+1)*r]
        key = hashlib.blake2b(b"".join(x.to_bytes(8,"big") for x in chunk),digest_size=8).hexdigest()
        yield b,key

def jaccard(a:Set[str],b:Set[str]) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

#--- graph ---
def connected_components(N:int,edges:defaultdict) -> List[List[int]]:
    seen,comps = set(),[]
    for n in range(N):
        if n in seen: continue
        q = deque([n]); seen.add(n); comp=[]
        while q:
            u = q.popleft(); comp.append(u)
            for v in edges[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        comps.append(comp)
    return comps

#--- split pipeline ---
def build_chunk_groups(file_list:List[str],
                       block_size:int=512, ngram:int=6,
                       num_perm:int=128, bands:int=32,
                       jaccard_thresh:float=0.80,
                       max_bucket:int=200, seed:int=42,
                       token_col:str="Domain_Name"):
    random.seed(seed)

    # Flatten all chunks across all files
    chunk_src: List[Tuple[str,int]] = []  # (file_path,chunk_index_in_file)
    chunk_sh:  List[Set[str]] = []
    chunk_sig: List[Tuple[int,...]] = []

    print(f"[1/5] Reading files and chunking (block={block_size},token_col={token_col}) ...")
    for path in file_list:
        ci = 0
        for ci,toks in iter_chunks_from_csv(path,block_size=block_size,token_col=token_col):
#       for ci,toks in iter_chunks_from_file(path,block_size):
            chunk_src.append((path,ci))
            sh = shingles(toks,n=ngram)
            chunk_sh.append(sh)
            chunk_sig.append(minhash_signature(sh,num_perm=num_perm))
    N = len(chunk_src)
    print(f"    total chunks: {N:,}")

    print(f"[2/5] LSH bucketing ...")
    buckets = defaultdict(list)  # (band,key) -> [chunk_ids]
    for cid,sig in enumerate(chunk_sig):
        for b,k in lsh_bands(sig,bands=bands):
            buckets[(b,k)].append(cid)

    print(f"[3/5] Candidate pairs ...")
    candidates = defaultdict(set)  # cid -> set(other_cid)
    for (_, _), ids in buckets.items():
        if len(ids) < 2: continue
        if len(ids) > max_bucket:
            ids = random.sample(ids,max_bucket)
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                a,b = ids[i],ids[j]
                candidates[a].add(b)

    print(f"[4/5] Verifying by exact Jaccard >= {jaccard_thresh:.2f} ...")
    edges = defaultdict(set)  # chunk graph
    checked=added=0
    for a,neighs in candidates.items():
        sha = chunk_sh[a]
        for b in neighs:
            shb = chunk_sh[b]
            checked += 1
            if jaccard(sha,shb) >= jaccard_thresh:
                edges[a].add(b); edges[b].add(a); added += 1
    for i in range(N): edges.setdefault(i, set())
    print(f"    checked pairs: {checked:,} | edges: {added:,}")

    print(f"[5/5] Connected components (chunk groups) ...")
    comps = connected_components(N,edges)  # list of lists of chunk_ids
    print(f"    groups: {len(comps)} | avg size: {sum(len(c) for c in comps)/max(1,len(comps)):.2f} chunks/group")
    return comps, chunk_src

def grouped_split(comps:List[List[int]],train_ratio=0.8,seed=42):
    rnd = random.Random(seed)
    order = list(range(len(comps))); rnd.shuffle(order)
    train_ids,eval_ids = [],[]
    total = sum(len(comps[i]) for i in order)
    tgt = int(round(total * train_ratio)); acc = 0
    for idx in order:
        grp = comps[idx]
        if acc < tgt:
            train_ids.extend(grp); acc += len(grp)
        else:
            eval_ids.extend(grp)
    return sorted(train_ids),sorted(eval_ids)

def write_index_tsv(path:str,sel_chunk_ids:List[int],chunk_src:List[Tuple[str,int]],block_size:int):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f,delimiter="\t")
        w.writerow(["file_path","start_token","length","chunk_index_in_file"])
        for cid in sel_chunk_ids:
            file_path,chunk_idx = chunk_src[cid]
            start = chunk_idx * block_size
            w.writerow([file_path,start,block_size,chunk_idx])

def read_file_list(list_path:str) -> List[str]:
    with open(list_path,"r",encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser(description="Build train/eval chunk index TSVs with chunk-level leakage control.")
    ap.add_argument("--file-list",required=True,help="Text file: one token-file path per line")
    ap.add_argument("--target",required=True,help="Directory indicator of output TSV for training chunks")
    ap.add_argument("--train-ratio",type=float,default=0.8)
    ap.add_argument("--block-size",type=int,default=512)
    ap.add_argument("--ngram",type=int,default=6)
    ap.add_argument("--num-perm",type=int,default=128)
    ap.add_argument("--bands",type=int,default=32)
    ap.add_argument("--jaccard-thresh",type=float,default=0.80)
    ap.add_argument("--max-bucket",type=int,default=200)
    ap.add_argument("--seed",type=int,default=42)
    args = ap.parse_args()

    files = read_file_list(args.file_list)
    if not files:
        print("Empty file list.",file=sys.stderr); sys.exit(1)

    comps,chunk_src = build_chunk_groups(
        files,
        block_size=args.block_size,
        ngram=args.ngram,
        num_perm=args.num_perm,
        bands=args.bands,
        jaccard_thresh=args.jaccard_thresh,
        max_bucket=args.max_bucket,
        seed=args.seed
    )
    train_ids,eval_ids = grouped_split(comps,train_ratio=args.train_ratio,seed=args.seed)
    print(f"Chunks → train: {len(train_ids)} | eval: {len(eval_ids)}")

    out_dir = f'./indices.{args.target}'
    write_index_tsv(os.path.join(out_dir,'train_index.tsv'),train_ids,chunk_src,args.block_size)
    write_index_tsv(os.path.join(out_dir,'eval_index.tsv'),eval_ids,chunk_src,args.block_size)
    print(f"Wrote:\n  {out_dir}/train_index.tsv\n  {out_dir}/eval_index.tsv")

if __name__ == "__main__":
    main()

#---end---

