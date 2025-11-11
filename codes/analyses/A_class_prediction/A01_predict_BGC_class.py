#!/usr/bin/env python3

import os,torch,argparse,torch
import pandas as pd
from tqdm import tqdm
from transformers import RobertaConfig,RobertaForMaskedLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from collections import defaultdict

ifile = '../../../data/antismash_bacterial_bgcs_class_testset.txt'
model_path = '../../../models/model_I_roberta_bgc/model_I_roberta_bgc_class.pth'
vocab_json = '../../../models/vocab_pfam_product_hmmer33.json'
topk = 10
ofile = "o01.class_prediction_results.top"+str(topk)+".csv"

# tokenizer setting for special-token awareness
tok_obj = Tokenizer.from_file(vocab_json)
tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj)
tok.add_special_tokens({"mask_token":"[MASK]","pad_token":"[PAD]","sep_token":"[SEP]"})

# model setting
config = RobertaConfig(
    vocab_size=19723,
    max_position_embeddings=514,
    hidden_size=1024,
    num_attention_heads=16,
    num_hidden_layers=8,
    type_vocab_size=1
)
model = RobertaForMaskedLM(config)
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))) # Load model to CPU, not to CUDA
model.eval()  # set to eval mode


def predict_single_class(line,topk=5):
    # line: "ID: [CLS] CLASS [CLS] rest ... [SEP] ..."  (CLASS is ONE token)
    rec_id,payload = line.split(":",1)
    toks = payload.split()

    # normalize placeholders to the model’s specials
    CLS = tok.cls_token or "[CLS]"
    SEP = tok.sep_token or "[SEP]"
    M   = tok.mask_token
    repl = {"[CLS]": CLS, "[SEP]": SEP}
    toks = [repl.get(t,t) for t in toks]

    # class token = immediately after the first CLS
    class_idx = toks.index(CLS) + 1
    org_cls = toks[class_idx]
    toks_masked = toks.copy()
    toks_masked[class_idx] = M

    # tokenize EXACTLY these tokens
    enc = tok(toks_masked,is_split_into_words=True,add_special_tokens=False,return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).logits[0]
        # find where the mask ended up after tokenization
        mask_pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=False).item()
        # get distribution for that position
        probs = torch.softmax(out[mask_pos],dim=-1)
        pred_id = probs.argmax().item()

    predicted_token = tok.decode([pred_id],skip_special_tokens=True)

    # return top tokens from vocab
    top = torch.topk(probs,k=topk)
    ranked = [(tok.decode([i.item()]).strip(),float(top.values[j].item())) for j,i in enumerate(top.indices)]

    return rec_id.strip(),org_cls,ranked

# Main
nlines = sum(1 for _ in open(ifile,encoding='utf-8'))
token_stats = {}; counts = defaultdict(int); rows = []
with open(ifile,'r',encoding='utf-8') as f:
    for line in tqdm(f,total=nlines,desc="Processing lines"):
        rec_id,cls,ranked = predict_single_class(line,topk)
        pred_ids = [x[0] for x in ranked]
        pred_probs = [x[1] for x in ranked]
        counts[rec_id] += 1
        id_uniq = f"{rec_id}_{counts[rec_id]:01d}"
        if rec_id in token_stats:
            token_stats[id_uniq] = [cls]+pred_ids+pred_probs
        else:
            token_stats[rec_id] = token_stats.get(rec_id,[])+[cls]+pred_ids+pred_probs

# save result
df = pd.DataFrame.from_dict(token_stats,orient='index')
df.reset_index(inplace=True)
#df.columns = ["Token","Total_Count","Correct_Count"]
df.columns = ['ID','Class'] + [f"#{i}" for i in range(1,topk+1)] + [f"prob#{i}" for i in range(1,topk+1)]
df.to_csv(ofile,index=False)

print(f"Result was saved in {ofile} .")

#---end---

