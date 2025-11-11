#!/usr/bin/env python3

import csv,torch,random,argparse
from tokenizers import Tokenizer
from transformers import RobertaConfig,RobertaForMaskedLM,RobertaTokenizerFast,PreTrainedTokenizerFast,AutoModelForMaskedLM
from tqdm import tqdm
from safetensors.torch import load_file

ap = argparse.ArgumentParser(description="Predict domains in MIBiG BGCs using a constructed model.")
ap.add_argument("--vocab-json",type=str,required=True,help="tokenizer json")
ap.add_argument("--ckpt-dir",type=str,required=True,help="Checkpoint directory of learned model")
ap.add_argument("--input-file",type=str,required=True,help="Domain list for eval")
ap.add_argument("--title",type=str,default="prediction_",help="Output file name header")
ap.add_argument("--max-len",type=int,default=512,help="Model context length used in training")
args = ap.parse_args()

# GPU setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenizer setting
try:
    tok = PreTrainedTokenizerFast.from_pretrained(args.ckpt_dir)
except Exception:
    tok_obj = Tokenizer.from_file(args.vocab_json)
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj)
    tok.add_special_tokens({"mask_token":"[MASK]","pad_token":"[PAD]","sep_token":"[SEP]"})

assert tok.mask_token_id is not None and tok.pad_token_id is not None

model = RobertaForMaskedLM.from_pretrained(args.ckpt_dir)
model.to(device).eval()

# helpers
def encode_tokens(domain_tokens):
    enc = tok(
        domain_tokens,
        is_split_into_words=True,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=args.max_len,
        return_tensors="pt",
    )
    return {k:v.to(device) for k,v in enc.items()}

def find_mask_index(input_ids):
    pos = (input_ids[0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    return None if pos.numel() == 0 else pos.item()

# read domain list for test
with open(args.input_file,"r",encoding="utf-8") as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]

out_path = f"every_domain_prediction.{args.title}.csv"
with open(out_path,"w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Cluster","Original position","Rank 1","Rank 2","Rank 3","Rank 4","Rank 5","Rank 6","Rank 7","Rank 8","Rank 9","Rank 10","Probability 1","Probability 2","Probability 3","Probability 4","Probability 5","Probability 6","Probability 7","Probability 8","Probability 9","Probability 10"
    ])

    for line in tqdm(lines, desc="Processing lines"):
        parts = line.split()
        if len(parts) < 3: continue
        title = parts[0]
        tokens = parts[2:]

        try:
            i = next(idx for idx, t in enumerate(tokens) if t == tok.mask_token or t == "[MASK]")
        except StopIteration:
            continue # no [MASK] in this line; skip

        masked = list(tokens)
        enc = encode_tokens(masked)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        mask_idx = find_mask_index(input_ids)
        if mask_idx is None: continue

        with torch.no_grad():
            out = model(input_ids=input_ids,attention_mask=attn_mask)
            logits = out.logits  # [1,L,V]

        # scores at the mask position
        scores = logits[0,mask_idx]        # [V]
        probs = torch.softmax(scores,dim=-1)

        # top-100 predictions
        topk_probs,topk_idx = torch.topk(probs,100)
        pred_tokens = tok.convert_ids_to_tokens(topk_idx.tolist())
        pred_probs  = topk_probs.tolist()

        writer.writerow([
            title,i,*pred_tokens,
            *[f"{p:.9f}" for p in pred_probs]
        ])

print(f"Done → {out_path}")

#---end---
