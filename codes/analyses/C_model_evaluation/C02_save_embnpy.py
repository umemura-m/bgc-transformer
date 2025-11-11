#!/usr/bin/env python3

import re,torch,ast,argparse, pandas as pd, numpy as np
from torch.utils.data import DataLoader,Dataset
from tokenizers import Tokenizer
from transformers import RobertaForMaskedLM,PreTrainedTokenizerFast

# argument setting
ap = argparse.ArgumentParser()
ap.add_argument("--title",type=str,required=True)
ap.add_argument("--dfile",type=str,required=True)
ap.add_argument("--vjson",type=str,required=True)
ap.add_argument("--ckdir",type=str,required=True)
args = ap.parse_args()

df_dname = pd.read_csv(args.dfile,converters={"tokens": lambda s: ast.literal_eval(s) if isinstance(s,str) else s})

df_inded = df_dname.copy()
df_inded["n_tokens"] = df_inded["tokens"].apply(len)
df_inded = df_inded[["cluster_id","label","n_tokens"]]
df_inded.to_csv("o02.embeddings_index.csv",index=False)

# load model
try:
    tok = PreTrainedTokenizerFast.from_pretrained(args.ckdir)
except Exception:
    tok_obj = Tokenizer.from_file(args.vjson)
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj)
    tok.add_special_tokens({"mask_token":"[MASK]","pad_token":"[PAD]","sep_token":"[SEP]"})

assert tok.mask_token_id is not None and tok.pad_token_id is not None
model = RobertaForMaskedLM.from_pretrained(args.ckdir)

# vector evaluator
def pool_mean(hidden,attn_mask,input_ids):
    mask = attn_mask.bool()
    for attr in ("cls_token_id","sep_token_id","bos_token_id","eos_token_id"):
        tid = getattr(tok,attr,None)
        if tid is not None:
            mask = mask & (input_ids != tid)
    masked = hidden * mask.unsqueeze(-1)
    lengths = mask.sum(1).clamp_min(1).unsqueeze(-1)
    return masked.sum(1) / lengths

seqs = df_dname["tokens"].tolist()

class BGCDS(Dataset):
    def __init__(self, seqs): self.seqs=seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self,i):
        enc = tok(self.seqs[i],is_split_into_words=True,
                  add_special_tokens=False,padding="max_length",
                  truncation=True,max_length=512,return_tensors="pt")
        return {k:v.squeeze(0) for k,v in enc.items()}

dl = DataLoader(BGCDS(seqs),batch_size=16,shuffle=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE).eval()

num_layers = model.config.num_hidden_layers
L1 = max(1, num_layers//3)
L2 = max(1, (2*num_layers)//3)
L3 = num_layers

def collect(layer_idx):
    outs = []
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(DEVICE)
            mask= batch["attention_mask"].to(DEVICE)
            hs  = model(ids,attention_mask=mask,output_hidden_states=True).hidden_states[layer_idx]
            outs.append(pool_mean(hs,mask,ids).cpu().numpy())
    return np.vstack(outs)

X_L1 = collect(L1); X_L2 = collect(L2); X_L3 = collect(L3)
np.save("./emb/o02.emb_L1."+args.title+".npy", X_L1); np.save("./emb/o02.emb_L2."+args.title+".npy", X_L2); np.save("./emb/o02.emb_L3."+args.title+".npy", X_L3)

#---end---

