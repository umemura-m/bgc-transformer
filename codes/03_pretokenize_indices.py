#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json,os
from collections import defaultdict
from typing import List,Tuple,Dict
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

#--- read TSV indices ---
def load_index_tsv(path:str):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        r = csv.DictReader(f,delimiter="\t")
        for row in r:
            rows.append((
                row["file_path"],
                int(row["start_token"]),
                int(row["length"])
            ))
    return rows  # list[(file_path,start,length)]

#--- token file loader (cache per file) ---
def read_token_file(path:str,token_col:str="Domain_Name") -> List[str]:
    toks:List[str] = []
    with open(path,"r",encoding="utf-8",newline="") as f:
        reader = csv.DictReader(f,delimiter=",")
        if token_col not in reader.fieldnames:
            raise KeyError(f"{token_col} not found in {path}. Columns: {reader.fieldnames}")
        for row in reader:
            t = (row.get(token_col) or "").strip()
            if t: toks.append(t)
    return toks

# Convert index rows -> memmap of input_ids (int32), shape [N,seq_len].
def pretokenize_to_memmap(index_tsv:str,out:str,
                          tokenizer,batch_size:int=1024,
                          add_special_tokens:bool=False,):
    rows = load_index_tsv(index_tsv)
    if not rows:
        raise RuntimeError(f"No rows in {index_tsv}")

    # Determine seq_len from first row (should be uniform, e.g., 512)
    seq_len = rows[0][2]
    for _,_,L in rows:
        if L != seq_len:
            raise ValueError(f"Mixed chunk lengths detected (got {L} vs {seq_len}).")

    os.makedirs(os.path.dirname(out),exist_ok=True)
    n = len(rows)
    mmap_path = f"{out}.ids.int32.mmap"
    meta_path = f"{out}.meta.json"

    # allocate memmap
    ids = np.memmap(mmap_path,dtype=np.int32,mode="w+",shape=(n,seq_len))

    # group rows by file to avoid re-reading token files
    by_file:Dict[str,List[Tuple[int,int,int]]] = defaultdict(list)
    for i, (fp,start,length) in enumerate(rows):
        by_file[fp].append((i,start,length))

    # process file by file
    pbar = tqdm(total=n,desc=f"Pretokenize {os.path.basename(index_tsv)}")
    for fp,items in by_file.items():
        toks = read_token_file(fp)  # list[str]
        n_tok = len(toks)
        for (row_idx,start,length) in items:
            if start < 0 or start + length > n_tok:
                raise ValueError(f"Slice OOB in {fp}: start={start}, len={length}, n_tok={n_tok}")

        # batch encode for speed
        for bi in range(0,len(items),batch_size):
            batch = items[bi:bi+batch_size]
            batch_tokens,order = [],[]
            for (row_idx,start,length) in batch:
                block = toks[start:start+length]
                if len(block) != length:
                    raise ValueError(f"Slice length mismatch in {fp} at start={start}")
                batch_tokens.append(block)
                order.append(row_idx)

            enc = tokenizer(
                batch_tokens,
                is_split_into_words=True,
#               add_special_tokens=add_special_tokens,
                add_special_tokens=False,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="np"
            )
            # enc["input_ids"] is np.int64 by default; cast to int32 to save space
            arr = enc["input_ids"].astype(np.int32,copy=False)
            if arr.shape[1] != seq_len:
                raise RuntimeError(f"Encoded width {arr.shape[1]} != seq_len {seq_len}")

            ids[order, :] = arr
            pbar.update(len(order))

    pbar.close()


#--probe--
#--- end-of-run verification (global indexing) ---
# Reopen via a fresh memmap view to be explicit
    mm = np.memmap(mmap_path, dtype=np.int32, mode="r", shape=(n,seq_len))

# sample a few global rows
    sample_is = [0, n//2, n-1] if n >= 3 else list(range(n))
    for i in sample_is:
        fp, start, length = rows[i]                  # <- use the correct file for this global row
        toks_i = read_token_file(fp)                 # reload tokens for *that* file
        block  = toks_i[start:start+length]          # list[str]

        enc = tokenizer(
            [block],
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="np",
        )["input_ids"][0].astype(np.int32)

        row_mm = np.array(mm[i], copy=True)
        if not np.array_equal(row_mm, enc):
            # print first diff to diagnose
            k = int(np.where(row_mm != enc)[0][0])
            print(f"❌ mismatch at global row {i} pos {k}  mm:{row_mm[k]}  enc:{enc[k]}")
            print("mm tok :", tokenizer.convert_ids_to_tokens([int(row_mm[k])])[0])
            print("enc tok:", tokenizer.convert_ids_to_tokens([int(enc[k])])[0])
            raise SystemExit("memmap row != tokenizer(list) encoding")

    print("✅ memmap == tokenizer(list) for sampled rows")
#--probe end--


    # flush memmap
    ids.flush()

    # write meta
    meta = {
        "num_examples": int(n),
        "seq_len": int(seq_len),
        "dtype": "int32",
        "pad_token_id": int(tokenizer.pad_token_id),
        "mmap_path": os.path.abspath(mmap_path),
        "index_tsv": os.path.abspath(index_tsv),
        "add_special_tokens": bool(add_special_tokens),
    }
    with open(meta_path,"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print(f"Saved:\n  {mmap_path}\n  {meta_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",required=True,help="Name of target; TSV (file_path,start_token,length ...)")
    ap.add_argument("--vocab-json",required=True,help="Path to your tokenizer JSON")
    args = ap.parse_args()

    if args.vocab_json:
        tok_obj = Tokenizer.from_file(args.vocab_json)
        tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj)
        tok.add_special_tokens({"mask_token":"[MASK]","pad_token":"[PAD]","sep_token":"[SEP]"})
    else:
        raise SystemExit("Please provide --vocab-json or adapt tokenizer init inside pretokenize_indices.py")

    index_tsv = './indices.'+args.target+'/train_index.tsv'
    out = './shards.'+args.target+'/train'
    pretokenize_to_memmap(index_tsv,out,tok)

    index_tsv = './indices.'+args.target+'/eval_index.tsv'
    out = './shards.'+args.target+'/eval'
    pretokenize_to_memmap(index_tsv,out,tok)

if __name__ == "__main__":
    main()

#---end---

